// =====================================================================================
// PoseGraphOptimizer.cpp
// ---------------------------------------------------------------------------
// PoseGraphOptimizer.hpp icinde bildirilen sinifin implementasyonu.
// =====================================================================================

#include "PoseGraphOptimizer.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

namespace vio {

using gtsam::symbol_shorthand::X;

namespace {
gtsam::SharedNoiseModel makeNoise(double rot_sigma, double trans_sigma) {
  return gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << rot_sigma, rot_sigma, rot_sigma, trans_sigma, trans_sigma, trans_sigma).finished());
}
}  // namespace

PoseGraphOptimizer::PoseGraphOptimizer(const PoseGraphParams& params) : params_(params) {}

void PoseGraphOptimizer::addKeyframe(int keyframe_id, const gtsam::Pose3& raw_pose) {
  values_.insert(X(keyframe_id), raw_pose);

  if (prev_keyframe_id_ < 0) {
    // Ilk keyframe: grafin capasi (anchor) - bu olmadan tum graf keyfi bir
    // cercevede kayar (gauge freedom), tipki FactorGraphBackend'deki gibi.
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
        X(keyframe_id), raw_pose, makeNoise(params_.prior_rot_sigma_rad, params_.prior_trans_sigma_m)));
  } else {
    // Ardisik keyframe'ler arasi "odometry" kisiti: FactorGraphBackend'in o
    // anki (henuz loop-closure'suz) tahmininden turetilen GORELI donusum.
    const gtsam::Pose3 prev_raw_pose = values_.at<gtsam::Pose3>(X(prev_keyframe_id_));
    // NOT: prev_raw_pose, graf ONCEKI bir loop-closure ile zaten duzeltilmis
    // olabilir (values_ guncel/duzeltilmis degerleri tutar) - bu YANLIS
    // DEGIL, TAM TERSINE ISTENEN davranistir: yeni eklenen odometry kisiti
    // grafin SU ANKI EN IYI bilgisine gore kurulur.
    const gtsam::Pose3 relative = prev_raw_pose.between(raw_pose);
    graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
        X(prev_keyframe_id_), X(keyframe_id), relative,
        makeNoise(params_.odometry_rot_sigma_rad, params_.odometry_trans_sigma_m)));
  }

  prev_keyframe_id_ = keyframe_id;
}

void PoseGraphOptimizer::addLoopClosure(int old_keyframe_id, int new_keyframe_id, const gtsam::Pose3& T_world_new) {
  const gtsam::Pose3 T_world_old = values_.at<gtsam::Pose3>(X(old_keyframe_id));
  // PnP'nin urettigi MUTLAK pozu, X(old)'un SU ANKI tahminine gore GORELI
  // bir kisita ceviriyoruz (bkz. header'daki aciklama).
  const gtsam::Pose3 relative = T_world_old.between(T_world_new);

  graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
      X(old_keyframe_id), X(new_keyframe_id), relative,
      makeNoise(params_.loop_rot_sigma_rad, params_.loop_trans_sigma_m)));

  has_loop_closure_ = true;
  ++loop_closure_count_;
  reoptimize();
}

void PoseGraphOptimizer::reoptimize() {
  gtsam::LevenbergMarquardtParams lm_params;
  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, values_, lm_params);
  values_ = optimizer.optimize();
}

gtsam::Pose3 PoseGraphOptimizer::correctedPoseAt(int keyframe_id) const {
  return values_.at<gtsam::Pose3>(X(keyframe_id));
}

}  // namespace vio
