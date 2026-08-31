// =====================================================================================
// CameraModel.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GÖREVİ:
//   cam0'ın pinhole (iğne deliği) kamera modelini ve radial-tangential lens
//   distorsiyonunu (bozulmasını) matematiksel olarak temsil etmek.
//
//   Neden gerekli? Kameralar gerçek dünyadaki düz çizgileri hafifçe kavisli
//   çekerler (özellikle geniş açılı lensler). Essential Matrix / triangulation
//   gibi tüm geometrik hesaplamalar "ideal" (distorsiyonsuz) pinhole modeli
//   varsayar. Bu yüzden ham piksel koordinatlarını kullanmadan ÖNCE
//   distorsiyonu gidermemiz (undistort) gerekir - aksi halde epipolar
//   geometri hatalı çıkar ve doğruluk (ATE) ciddi şekilde bozulur.
//
//   Bu sınıf DataLoader'ın okuduğu CameraCalibration verisini alıp OpenCV'nin
//   anladığı K matrisi + distorsiyon katsayıları formatına çevirir ve
//   undistort işlemlerini dışarıya sade bir arayüzle sunar.
// =====================================================================================

#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "DataLoader.hpp"

namespace vio {

// -------------------------------------------------------------------------------------
// PinholeCameraModel
// ---------------------------------------------------------------------------
// cam0 için pinhole + radial-tangential distorsiyon modeli. Durumsuzdur
// (parametreler kurucuda bir kere ayarlanır, sonrasında sadece hesaplama
// yapılır).
// -------------------------------------------------------------------------------------
class PinholeCameraModel {
 public:
  explicit PinholeCameraModel(const CameraCalibration& calib);

  // Ham (distorsiyonlu) piksel noktalarını, distorsiyonu giderilmiş VE
  // normalize edilmiş kamera koordinatlarına çevirir (odak uzaklığı=1,
  // optik merkez=(0,0)). Bu format Essential Matrix / triangulation gibi
  // matematiksel işlemler için idealdir.
  std::vector<cv::Point2f> undistortNormalized(const std::vector<cv::Point2f>& pixel_pts) const;

  // Ham piksel noktalarını distorsiyonu giderilmiş piksel koordinatlarına
  // çevirir (K ile geri çarpılmış hali) - görselleştirme veya piksel
  // biriminde RANSAC eşiği kullanmak isteyen kodlar için kullanışlıdır.
  std::vector<cv::Point2f> undistortPixel(const std::vector<cv::Point2f>& pixel_pts) const;

  // 3x3 kamera iç parametre (intrinsics) matrisi:
  //   [ fx  0  cx ]
  //   [  0 fy  cy ]
  //   [  0  0   1 ]
  const cv::Mat& K() const { return K_; }

  // Distorsiyon katsayıları [k1, k2, p1, p2, (k3)] - OpenCV formatında.
  const cv::Mat& distCoeffs() const { return dist_coeffs_; }

  int width() const { return calib_.width; }
  int height() const { return calib_.height; }

  // Kamera -> govde(IMU) dis parametre (extrinsic) donusumu. VioInitializer
  // gibi kamera pozunu govde pozuna cevirmesi gereken modullerin ihtiyaci var.
  const Eigen::Matrix4d& T_BS() const { return calib_.T_BS; }

 private:
  CameraCalibration calib_;
  cv::Mat K_;            // 3x3, CV_64F
  cv::Mat dist_coeffs_;  // Nx1, CV_64F
};

}  // namespace vio
