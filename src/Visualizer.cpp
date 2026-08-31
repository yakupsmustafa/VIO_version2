// =====================================================================================
// Visualizer.cpp
// ---------------------------------------------------------------------------
// Visualizer.hpp icinde bildirilen sinifin implementasyonu.
// =====================================================================================

#include "Visualizer.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace vio {

Visualizer::Visualizer(const VisualizerParams& params) : params_(params) {}

void Visualizer::showCameraFrame(const cv::Mat& image_gray, const std::vector<cv::KeyPoint>& keypoints, double fps) {
  cv::Mat vis;
  cv::cvtColor(image_gray, vis, cv::COLOR_GRAY2BGR);

  for (const auto& kp : keypoints) {
    cv::circle(vis, kp.pt, 3, params_.keypoint_color, 1, cv::LINE_AA);
  }

  const std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps + 0.5));
  // Once siyah "golge" (okunabilirlik icin), sonra asil metni ustune yaz.
  cv::putText(vis, fps_text, cv::Point(11, 26), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
  cv::putText(vis, fps_text, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7, params_.text_color, 2, cv::LINE_AA);

  cv::imshow(params_.camera_window_name, vis);
}

void Visualizer::showTrajectory(const std::vector<Eigen::Vector3d>& vio_positions,
                                 const std::vector<Eigen::Vector3d>& gt_positions, double fps,
                                 const LiveMetrics& metrics) {
  const int size = params_.trajectory_canvas_size_px;
  const int margin = params_.trajectory_margin_px;
  trajectory_canvas_ = cv::Mat(size, size, CV_8UC3, cv::Scalar(30, 30, 30));  // koyu gri arka plan

  if (vio_positions.empty() && gt_positions.empty()) {
    cv::imshow(params_.trajectory_window_name, trajectory_canvas_);
    return;
  }

  // --- TUM noktalarin (VIO+GT) X-Y sinirlarini bul (otomatik olcekleme icin) ---
  double min_x = std::numeric_limits<double>::max();
  double max_x = -min_x;
  double min_y = min_x;
  double max_y = -min_x;
  auto updateBounds = [&](const std::vector<Eigen::Vector3d>& pts) {
    for (const auto& p : pts) {
      min_x = std::min(min_x, p.x());
      max_x = std::max(max_x, p.x());
      min_y = std::min(min_y, p.y());
      max_y = std::max(max_y, p.y());
    }
  };
  updateBounds(vio_positions);
  updateBounds(gt_positions);

  const double range_x = std::max(max_x - min_x, 0.1);
  const double range_y = std::max(max_y - min_y, 0.1);
  const double usable_px = static_cast<double>(size - 2 * margin);
  // Ayni olcegi hem X hem Y'ye uygula (aspect ratio korunsun, sekil bozulmasin).
  const double scale = usable_px / std::max(range_x, range_y);

  const double cx = (min_x + max_x) / 2.0;
  const double cy = (min_y + max_y) / 2.0;
  const double half = size / 2.0;

  // Dunya (X,Y) koordinatini piksel koordinatina cevirir. Y ekseni TERS
  // CEVRILIR ki dunya +Y yonu goruntude "yukari" gorunsun (harita/kus bakisi
  // aliskanligina uygun - OpenCV'nin goruntu Y'si asagi dogru artar).
  auto worldToPixel = [&](double x, double y) {
    const double px = (x - cx) * scale + half;
    const double py = half - (y - cy) * scale;
    return cv::Point(static_cast<int>(px), static_cast<int>(py));
  };

  auto drawTrajectory = [&](const std::vector<Eigen::Vector3d>& pts, const cv::Scalar& color) {
    for (std::size_t i = 1; i < pts.size(); ++i) {
      cv::line(trajectory_canvas_, worldToPixel(pts[i - 1].x(), pts[i - 1].y()),
               worldToPixel(pts[i].x(), pts[i].y()), color, 2, cv::LINE_AA);
    }
    if (!pts.empty()) {
      // Su anki (en son) pozisyonu daha buyuk bir daireyle vurgula.
      cv::circle(trajectory_canvas_, worldToPixel(pts.back().x(), pts.back().y()), 6, color, cv::FILLED, cv::LINE_AA);
    }
  };

  drawTrajectory(gt_positions, params_.gt_color);
  drawTrajectory(vio_positions, params_.vio_color);

  // --- Lejant: hangi rengin VIO, hangisinin GT oldugunu yaz ---
  cv::line(trajectory_canvas_, cv::Point(15, 20), cv::Point(45, 20), params_.vio_color, 3, cv::LINE_AA);
  cv::putText(trajectory_canvas_, "VIO", cv::Point(52, 26), cv::FONT_HERSHEY_SIMPLEX, 0.6, params_.text_color, 2,
              cv::LINE_AA);
  cv::line(trajectory_canvas_, cv::Point(15, 45), cv::Point(45, 45), params_.gt_color, 3, cv::LINE_AA);
  cv::putText(trajectory_canvas_, "GT", cv::Point(52, 51), cv::FONT_HERSHEY_SIMPLEX, 0.6, params_.text_color, 2,
              cv::LINE_AA);

  // --- Canli ATE ve Umeyama scale (ustte, ortada) - kullanicinin talebi:
  // calisirken anlik olarak takip edebilmek icin. ---
  if (metrics.valid) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "ATE: %.3f m   Scale: %.3f", metrics.ate_rmse_m, metrics.umeyama_scale);
    const std::string metrics_text(buf);
    int baseline = 0;
    const cv::Size text_size = cv::getTextSize(metrics_text, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
    const int text_x = std::max(0, (size - text_size.width) / 2);
    const cv::Point text_org(text_x, 30);
    // Once siyah golge (okunabilirlik icin), sonra sari (dikkat cekici) asil metin.
    cv::putText(trajectory_canvas_, metrics_text, text_org + cv::Point(1, 1), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
    cv::putText(trajectory_canvas_, metrics_text, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255),
                2, cv::LINE_AA);
  }

  // --- FPS metni (sag ust kose) ---
  const std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps + 0.5));
  cv::putText(trajectory_canvas_, fps_text, cv::Point(size - 150, 26), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              params_.text_color, 2, cv::LINE_AA);

  cv::imshow(params_.trajectory_window_name, trajectory_canvas_);
}

bool Visualizer::pollKeyboard(int wait_ms) const {
  const int key = cv::waitKey(wait_ms) & 0xFF;
  return (key == 'q' || key == 27);  // 'q' veya ESC
}

}  // namespace vio
