// =====================================================================================
// Triangulation.cpp
// ---------------------------------------------------------------------------
// Triangulation.hpp icinde bildirilen Triangulator sinifinin implementasyonu.
// =====================================================================================

#include "Triangulation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>

namespace vio {

Triangulator::Triangulator(const PinholeCameraModel& camera, const TriangulationParams& params)
    : camera_(camera), params_(params) {}

TriangulatedPoint Triangulator::triangulate(const cv::Point2f& pixel1, const cv::Point2f& pixel2,
                                             const Eigen::Matrix4d& T_w_c1,
                                             const Eigen::Matrix4d& T_w_c2) const {
  const std::vector<cv::Point2f> p1{pixel1};
  const std::vector<cv::Point2f> p2{pixel2};
  return triangulateBatch(p1, p2, T_w_c1, T_w_c2).front();
}

std::vector<TriangulatedPoint> Triangulator::triangulateBatch(const std::vector<cv::Point2f>& pixels1,
                                                                const std::vector<cv::Point2f>& pixels2,
                                                                const Eigen::Matrix4d& T_w_c1,
                                                                const Eigen::Matrix4d& T_w_c2) const {
  const std::size_t n = pixels1.size();
  std::vector<TriangulatedPoint> results(n);
  if (n == 0) return results;

  // Ham piksel noktalarini distorsiyonu giderilmis VE normalize edilmis
  // (K'siz) koordinatlara cevir. Normalize koordinat kullanmamizin sebebi:
  // boylece projeksiyon matrisleri sadece [R|t] olur, K carpimina hic gerek
  // kalmaz - hem daha sade hem de K'dan bagimsiz calisir.
  const std::vector<cv::Point2f> norm1 = camera_.undistortNormalized(pixels1);
  const std::vector<cv::Point2f> norm2 = camera_.undistortNormalized(pixels2);

  // world -> camera donusumleri (T_w_c'nin tersi). cv::triangulatePoints'in
  // beklediği projeksiyon matrisi formati budur: x_normalized = [R|t] * X_world.
  const Eigen::Matrix4d T_c1_w = T_w_c1.inverse();
  const Eigen::Matrix4d T_c2_w = T_w_c2.inverse();

  cv::Mat P1(3, 4, CV_64F), P2(3, 4, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      P1.at<double>(r, c) = T_c1_w(r, c);
      P2.at<double>(r, c) = T_c2_w(r, c);
    }
  }

  cv::Mat pts1_mat(2, static_cast<int>(n), CV_64F);
  cv::Mat pts2_mat(2, static_cast<int>(n), CV_64F);
  for (std::size_t i = 0; i < n; ++i) {
    pts1_mat.at<double>(0, static_cast<int>(i)) = norm1[i].x;
    pts1_mat.at<double>(1, static_cast<int>(i)) = norm1[i].y;
    pts2_mat.at<double>(0, static_cast<int>(i)) = norm2[i].x;
    pts2_mat.at<double>(1, static_cast<int>(i)) = norm2[i].y;
  }

  cv::Mat points4d;
  cv::triangulatePoints(P1, P2, pts1_mat, pts2_mat, points4d);
  // cv::triangulatePoints CV_32F donduruyor, kendi hesaplarimizda CV_64F
  // kullanabilmek icin donusturuyoruz.
  points4d.convertTo(points4d, CV_64F);

  const double fx = camera_.K().at<double>(0, 0);
  const double fy = camera_.K().at<double>(1, 1);

  for (std::size_t i = 0; i < n; ++i) {
    const int idx = static_cast<int>(i);
    const double w = points4d.at<double>(3, idx);

    Eigen::Vector4d X_world_h(points4d.at<double>(0, idx), points4d.at<double>(1, idx),
                               points4d.at<double>(2, idx), w);
    if (std::abs(w) > 1e-12) {
      X_world_h /= w;
    }
    const Eigen::Vector3d X_world = X_world_h.head<3>();

    TriangulatedPoint& res = results[i];
    res.position_world = X_world;

    // --- Pozitif derinlik kontrolu (her iki kameranin da ONUNDE olmali) ---
    const Eigen::Vector3d X_c1 = (T_c1_w * X_world_h).head<3>();
    const Eigen::Vector3d X_c2 = (T_c2_w * X_world_h).head<3>();
    const bool positive_depth = (X_c1.z() > 0.0) && (X_c2.z() > 0.0);

    // --- Reprojeksiyon hatasi (normalize duzlemde hesaplanip piksel birime cevrilir) ---
    double err1_px = std::numeric_limits<double>::infinity();
    double err2_px = std::numeric_limits<double>::infinity();
    if (positive_depth) {
      const Eigen::Vector2d pred1(X_c1.x() / X_c1.z(), X_c1.y() / X_c1.z());
      const Eigen::Vector2d pred2(X_c2.x() / X_c2.z(), X_c2.y() / X_c2.z());
      const Eigen::Vector2d obs1(norm1[i].x, norm1[i].y);
      const Eigen::Vector2d obs2(norm2[i].x, norm2[i].y);
      // normalize duzlemdeki hata, fx/fy odak uzakligiyla carpilinca yaklasik
      // piksel biriminde hataya karsilik gelir (kucuk aci yaklasimi).
      err1_px = (pred1 - obs1).norm() * fx;
      err2_px = (pred2 - obs2).norm() * fy;
    }
    res.reprojection_error_px_1 = err1_px;
    res.reprojection_error_px_2 = err2_px;

    // --- Parallax acisi ---
    const Eigen::Vector3d C1 = T_w_c1.block<3, 1>(0, 3);
    const Eigen::Vector3d C2 = T_w_c2.block<3, 1>(0, 3);
    const Eigen::Vector3d ray1 = (X_world - C1).normalized();
    const Eigen::Vector3d ray2 = (X_world - C2).normalized();
    const double cos_angle = std::clamp(ray1.dot(ray2), -1.0, 1.0);
    const double parallax_deg = std::acos(cos_angle) * 180.0 / M_PI;
    res.parallax_deg = parallax_deg;

    res.valid = positive_depth && (err1_px <= params_.max_reprojection_error_px) &&
                (err2_px <= params_.max_reprojection_error_px) && (parallax_deg >= params_.min_parallax_deg);
  }

  return results;
}

}  // namespace vio
