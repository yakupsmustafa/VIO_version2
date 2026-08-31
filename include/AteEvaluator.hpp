// =====================================================================================
// AteEvaluator.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   GtAligner ile ZATEN hizalanmis VIO/GT pozisyon ciftlerinden ATE
//   (Absolute Trajectory Error) metrigini hesaplamak.
//
//   ATE NEDIR? Her keyframe icin, hizalanmis VIO pozisyonu ile GT pozisyonu
//   arasindaki OKLIDIT MESAFESI ("konum hatasi"). ATE-RMSE, bu hatalarin
//   KARELERININ ORTALAMASININ KAREKOKUDUR - VIO/SLAM literaturunde trajectory
//   dogrulugunu ozetlemek icin kullanilan STANDART tek sayidir. Kullaniciyla
//   birlikte hedeflenen aralik: 0.3-0.6 metre.
// =====================================================================================

#pragma once

#include <vector>

#include <Eigen/Core>

namespace vio {

struct AteResult {
  double rmse_m = 0.0;    // ana metrik - hedef: 0.3-0.6 m
  double mean_m = 0.0;
  double median_m = 0.0;
  double max_m = 0.0;
  std::vector<double> per_keyframe_errors_m;  // her keyframe'in kendi hatasi (m) - ileride grafik/analiz icin
};

class AteEvaluator {
 public:
  // aligned_vio_positions ve gt_positions AYNI SIRADA, AYNI SAYIDA nokta
  // icermeli (GtAligner::align()'in ciktisi + MatchedTrajectories::gt_positions).
  static AteResult evaluate(const std::vector<Eigen::Vector3d>& aligned_vio_positions,
                             const std::vector<Eigen::Vector3d>& gt_positions);
};

}  // namespace vio
