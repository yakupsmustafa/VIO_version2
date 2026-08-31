// =====================================================================================
// CameraModel.cpp
// ---------------------------------------------------------------------------
// CameraModel.hpp içinde bildirilen PinholeCameraModel sınıfının implementasyonu.
// =====================================================================================

#include "CameraModel.hpp"

#include <opencv2/calib3d.hpp>

namespace vio {

PinholeCameraModel::PinholeCameraModel(const CameraCalibration& calib) : calib_(calib) {
  K_ = (cv::Mat_<double>(3, 3) << calib_.fx, 0.0, calib_.cx,
                                   0.0, calib_.fy, calib_.cy,
                                   0.0, 0.0, 1.0);

  dist_coeffs_ = cv::Mat(static_cast<int>(calib_.distortion.size()), 1, CV_64F);
  for (std::size_t i = 0; i < calib_.distortion.size(); ++i) {
    dist_coeffs_.at<double>(static_cast<int>(i)) = calib_.distortion[i];
  }
}

std::vector<cv::Point2f> PinholeCameraModel::undistortNormalized(
    const std::vector<cv::Point2f>& pixel_pts) const {
  std::vector<cv::Point2f> out;
  // P (projeksiyon) matrisi verilmezse cv::undistortPoints çıktıyı normalize
  // edilmiş kamera koordinatlarında (K'siz) döndürür - tam istediğimiz şey.
  cv::undistortPoints(pixel_pts, out, K_, dist_coeffs_);
  return out;
}

std::vector<cv::Point2f> PinholeCameraModel::undistortPixel(
    const std::vector<cv::Point2f>& pixel_pts) const {
  std::vector<cv::Point2f> out;
  // P=K_ vererek normalize edilmiş sonucu tekrar piksel uzayina tasiyoruz;
  // boylece "distorsiyonu giderilmis ama hala piksel biriminde" nokta elde ediyoruz.
  cv::undistortPoints(pixel_pts, out, K_, dist_coeffs_, cv::noArray(), K_);
  return out;
}

}  // namespace vio
