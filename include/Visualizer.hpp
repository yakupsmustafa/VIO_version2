// =====================================================================================
// Visualizer.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   Calisma sirasinda IKI ayri OpenCV penceresi gostermek:
//     1) Kamera penceresi: o anki goruntu + takip edilen ORB ozellik
//        noktalari + FPS.
//     2) Trajectory penceresi: VIO'nun kestirdigi yorunge ile GT (ground
//        truth) yorungesinin USTTEN GORUNUMU (top-down, dunya cercevesinin
//        X-Y duzlemi - cunku dunya cercevemiz Z-up, bkz. VioInitializer'in
//        yercekimi-hizalama adimi). Iki yorunge FARKLI RENKTE cizilir, hangi
//        rengin hangisi oldugunu gosteren bir LEJANT ve o anki FPS metni
//        pencerenin uzerine yazilir (kullaniciyla kararlastirildi).
//
//   NEDEN SADECE OpenCV? Pangolin gibi 3D gorsellestirme kutuphaneleri daha
//   "etkileyici" olabilir ama ekstra derleme bagimliligi getirir. Zaten
//   kurulu olan OpenCV ile 2D ustten-gorunum bir trajectory grafigi cizmek
//   hem yeterince bilgilendirici hem de ek bagimliliksiz, hizli bir cozum
//   (kullaniciyla birlikte kararlastirildi).
// =====================================================================================

#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace vio {

// Trajectory penceresinin ustune yazilacak canli ATE/scale degerleri.
// valid=false ise (henuz yeterli GT-eslesmesi yoksa) hicbir sey yazilmaz.
struct LiveMetrics {
  bool valid = false;
  double ate_rmse_m = 0.0;   // hedef: 0.3-0.6 m
  double umeyama_scale = 0.0;  // hedef: ~1.0
};

struct VisualizerParams {
  std::string camera_window_name = "Kamera (ORB Ozellikleri)";
  std::string trajectory_window_name = "Trajectory (VIO vs GT)";
  int trajectory_canvas_size_px = 800;  // kare canvas, piksel
  int trajectory_margin_px = 50;

  // BGR format (OpenCV kurali).
  cv::Scalar vio_color = cv::Scalar(0, 0, 255);     // kirmizi
  cv::Scalar gt_color = cv::Scalar(0, 180, 0);      // yesil
  cv::Scalar keypoint_color = cv::Scalar(0, 255, 0);  // yesil (kamera penceresindeki noktalar icin)
  cv::Scalar text_color = cv::Scalar(255, 255, 255);  // beyaz
};

// -------------------------------------------------------------------------------------
// Visualizer
// -------------------------------------------------------------------------------------
class Visualizer {
 public:
  explicit Visualizer(const VisualizerParams& params = VisualizerParams());

  // Kamera penceresini gunceller: goruntu uzerine takip edilen ozellik
  // noktalarini ve FPS'i cizip gosterir (cv::imshow).
  void showCameraFrame(const cv::Mat& image_gray, const std::vector<cv::KeyPoint>& keypoints, double fps);

  // Trajectory penceresini gunceller: VIO ve GT yorungelerini (o ana kadarki
  // TUM gecmis noktalarla) ustten-gorunum (X-Y duzlemi) olarak cizer -
  // farkli renkte, lejantli, FPS'li (cv::imshow). metrics gecerliyse (bkz.
  // LiveMetrics), pencerenin USTUNE canli ATE ve Umeyama scale degerleri de
  // yazilir (kullanicinin talebi - calisirken anlik takip edebilmek icin).
  void showTrajectory(const std::vector<Eigen::Vector3d>& vio_positions,
                      const std::vector<Eigen::Vector3d>& gt_positions, double fps,
                      const LiveMetrics& metrics = LiveMetrics());

  // Son cizilen trajectory canvas'i (dosyaya kaydetmek/otomatik test etmek
  // icin - imshow'un goruntulenemedigi bir ortamda dogrulama amacli).
  const cv::Mat& lastTrajectoryCanvas() const { return trajectory_canvas_; }

  // cv::waitKey sarmalayici. 'q' veya ESC basilirsa true (cikis istendi) doner.
  bool pollKeyboard(int wait_ms = 1) const;

 private:
  VisualizerParams params_;
  cv::Mat trajectory_canvas_;
};

}  // namespace vio
