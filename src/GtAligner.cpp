// =====================================================================================
// GtAligner.cpp
// ---------------------------------------------------------------------------
// GtAligner.hpp icinde bildirilen fonksiyonlarin implementasyonu.
// =====================================================================================

#include "GtAligner.hpp"

namespace vio {

MatchedTrajectories GtAligner::match(const std::vector<TimestampNs>& vio_timestamps,
                                      const std::vector<Eigen::Vector3d>& vio_positions,
                                      const EurocDataLoader& loader) {
  MatchedTrajectories result;

  const auto& gt = loader.groundTruth();
  if (gt.empty() || vio_timestamps.empty()) return result;

  const TimestampNs gt_start = gt.front().timestamp_ns;
  const TimestampNs gt_end = gt.back().timestamp_ns;

  for (std::size_t i = 0; i < vio_timestamps.size(); ++i) {
    const TimestampNs t = vio_timestamps[i];
    // GT'nin GERCEKTEN kapsadigi araligin disindaki keyframe'leri atla -
    // aksi halde nearestGroundTruth hep ayni uc degere sabitlenir (bkz.
    // header'daki aciklama).
    if (t < gt_start || t > gt_end) continue;

    GroundTruthPose gt_pose;
    if (!loader.nearestGroundTruth(t, gt_pose)) continue;

    result.timestamps.push_back(t);
    result.vio_positions.push_back(vio_positions[i]);
    result.gt_positions.push_back(gt_pose.position);
  }

  return result;
}

AlignmentResult GtAligner::align(const MatchedTrajectories& matched, bool with_scale) {
  AlignmentResult result;
  const int n = static_cast<int>(matched.vio_positions.size());
  if (n < 3) return result;  // Umeyama en az 3 nokta gerektirir (rotasyonu belirlemek icin)

  Eigen::MatrixXd src(3, n), dst(3, n);
  for (int i = 0; i < n; ++i) {
    src.col(i) = matched.vio_positions[i];
    dst.col(i) = matched.gt_positions[i];
  }

  const Eigen::Matrix4d T = Eigen::umeyama(src, dst, with_scale);
  result.R = T.block<3, 3>(0, 0);
  result.t = T.block<3, 1>(0, 3);
  // Sim3'un sol-ust 3x3 blogu (R*scale) - rotasyon matrisi olculu oldugu
  // icin herhangi bir sutununun normu scale'i verir.
  result.scale = with_scale ? result.R.col(0).norm() : 1.0;
  if (with_scale) {
    result.R /= result.scale;  // R'i saf rotasyon matrisine geri dondur (R*scale -> R)
  }

  result.aligned_vio_positions.resize(n);
  for (int i = 0; i < n; ++i) {
    result.aligned_vio_positions[i] = result.scale * (result.R * matched.vio_positions[i]) + result.t;
  }

  return result;
}

}  // namespace vio
