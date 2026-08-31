// =====================================================================================
// AteEvaluator.cpp
// ---------------------------------------------------------------------------
// AteEvaluator.hpp icinde bildirilen fonksiyonun implementasyonu.
// =====================================================================================

#include "AteEvaluator.hpp"

#include <algorithm>
#include <cmath>

namespace vio {

AteResult AteEvaluator::evaluate(const std::vector<Eigen::Vector3d>& aligned_vio_positions,
                                  const std::vector<Eigen::Vector3d>& gt_positions) {
  AteResult result;
  const std::size_t n = aligned_vio_positions.size();
  if (n == 0 || n != gt_positions.size()) return result;

  result.per_keyframe_errors_m.resize(n);
  double sum = 0.0;
  double sum_sq = 0.0;
  double max_err = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    const double err = (aligned_vio_positions[i] - gt_positions[i]).norm();
    result.per_keyframe_errors_m[i] = err;
    sum += err;
    sum_sq += err * err;
    max_err = std::max(max_err, err);
  }

  result.mean_m = sum / static_cast<double>(n);
  result.rmse_m = std::sqrt(sum_sq / static_cast<double>(n));
  result.max_m = max_err;

  std::vector<double> sorted_errors = result.per_keyframe_errors_m;
  std::sort(sorted_errors.begin(), sorted_errors.end());
  result.median_m = sorted_errors[sorted_errors.size() / 2];

  return result;
}

}  // namespace vio
