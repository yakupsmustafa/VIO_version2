// =====================================================================================
// main.cpp
// ---------------------------------------------------------------------------
// TUM pipeline'i bir araya getiren ana program:
//   1) FeatureTracker ile ilk 10 keyframe'lik pencere toplanir
//   2) VioInitializer ile baslatilir (VI-SfM: olcek+yercekimi+hiz+bias)
//   3) FactorGraphBackend bu pencereyle kurulur (initializeWithWindow)
//   4) Devam eden karelerden gelen YENI keyframe'ler addKeyframe() ile
//      tek tek islenir (iSAM2/IncrementalFixedLagSmoother ile artimli
//      optimizasyon, sliding window marginalization)
//   5) HER kare icin: kamera penceresi (ORB noktalari + FPS) ve trajectory
//      penceresi (VIO vs GT, renkli+lejantli+FPS) guncellenir
//   6) Program sonunda (veya 'q' ile cikildiginda) GtAligner+AteEvaluator ile
//      NIHAI ATE (Absolute Trajectory Error) hesaplanip yazdirilir.
//
// GT (ground truth), tanimi geregi SADECE gorsellestirme ve ATE degerlendirme
// icin kullanilir - VIO kestirim algoritmasinin HICBIR yerinde girdi olarak
// KULLANILMAZ.
// =====================================================================================

#include <chrono>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

#include "AteEvaluator.hpp"
#include "CameraModel.hpp"
#include "DataLoader.hpp"
#include "FactorGraphBackend.hpp"
#include "FeatureTracker.hpp"
#include "FpsCounter.hpp"
#include "GtAligner.hpp"
#include "VioInitializer.hpp"
#include "Visualizer.hpp"

namespace {

// Su ana kadar biriken VIO/GT trajectory'lerini gunceller: yeni keyframe'in
// VIO pozunu ve (varsa) en yakin GT pozunu listelere ekler. GT SADECE
// gorsellestirme icindir, kestirime asla girdi olarak verilmez.
void appendTrajectoryPoint(const vio::FactorGraphBackend& backend, const vio::EurocDataLoader& loader, int keyframe_id,
                            vio::TimestampNs timestamp_ns, std::vector<vio::TimestampNs>& vio_timestamps,
                            std::vector<Eigen::Vector3d>& vio_positions, std::vector<Eigen::Vector3d>& gt_positions_vis) {
  const gtsam::Pose3 pose = backend.poseAt(keyframe_id);
  vio_timestamps.push_back(timestamp_ns);
  vio_positions.push_back(pose.translation());

  vio::GroundTruthPose gt;
  if (loader.nearestGroundTruth(timestamp_ns, gt)) {
    gt_positions_vis.push_back(gt.position);
  }
}

// Su ana kadarki VIO trajectory'sini GT ile hizalayip CANLI ATE/scale
// hesaplar (trajectory penceresinin ustune yazdirmak icin - kullanicinin
// talebi). Yeterli eslesen GT noktasi yoksa metrics.valid=false doner.
vio::LiveMetrics computeLiveMetrics(const std::vector<vio::TimestampNs>& vio_timestamps,
                                     const std::vector<Eigen::Vector3d>& vio_positions,
                                     const vio::EurocDataLoader& loader) {
  vio::LiveMetrics metrics;
  const vio::MatchedTrajectories matched = vio::GtAligner::match(vio_timestamps, vio_positions, loader);
  if (matched.vio_positions.size() < 3) return metrics;

  const vio::AlignmentResult rigid = vio::GtAligner::align(matched, /*with_scale=*/false);
  const vio::AlignmentResult sim3 = vio::GtAligner::align(matched, /*with_scale=*/true);
  const vio::AteResult ate = vio::AteEvaluator::evaluate(rigid.aligned_vio_positions, matched.gt_positions);

  metrics.valid = true;
  metrics.ate_rmse_m = ate.rmse_m;
  metrics.umeyama_scale = sim3.scale;
  return metrics;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dataset_root = "data/MH_01_easy";
  if (argc > 1) dataset_root = argv[1];

  vio::EurocDataLoader loader(dataset_root);
  const auto& cam = loader.cameraFrames();
  const auto& cc = loader.cameraCalibration();

  vio::PinholeCameraModel camera(cc);
  vio::FeatureTracker tracker(vio::OrbExtractionParams(), vio::MatcherParams(), vio::TrackerParams(), camera);
  vio::Visualizer visualizer;
  vio::FpsCounter fps_counter;

  // GT ~1.075s (~21.5 kare) sonra basliyor (bkz. proje hafizasi) - baslangic
  // penceresini bu noktadan sonraki keyframe'lerle dolduruyoruz.
  const vio::TimestampNs first_gt_ts = loader.groundTruth().front().timestamp_ns;

  std::vector<vio::KeyframeObservation> window;
  const int target_window_size = 10;

  std::size_t frame_idx = 0;
  for (; frame_idx < cam.size() && static_cast<int>(window.size()) < target_window_size; ++frame_idx) {
    cv::Mat img = cv::imread(cam[frame_idx].image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) continue;

    vio::KeyframeDecision d = tracker.processFrame(img, cam[frame_idx].timestamp_ns);
    fps_counter.tick();
    visualizer.showCameraFrame(img, tracker.currentKeypoints(), fps_counter.fps());
    visualizer.pollKeyboard();

    if (d.is_keyframe && cam[frame_idx].timestamp_ns > first_gt_ts) {
      vio::KeyframeObservation obs;
      obs.keyframe_id = static_cast<int>(window.size());
      obs.timestamp_ns = cam[frame_idx].timestamp_ns;
      for (const auto& kv : tracker.activeTracks()) obs.observations[kv.first] = kv.second.current_pixel;
      window.push_back(std::move(obs));
    }
  }

  std::cout << "Baslangic penceresi boyutu: " << window.size() << "\n";
  if (static_cast<int>(window.size()) < target_window_size) {
    std::cerr << "Yeterli keyframe toplanamadi.\n";
    return 1;
  }

  std::vector<std::vector<vio::ImuMeasurement>> imu_between_keyframes;
  for (std::size_t k = 0; k + 1 < window.size(); ++k) {
    imu_between_keyframes.push_back(loader.imuBetween(window[k].timestamp_ns, window[k + 1].timestamp_ns));
  }

  vio::VioInitializer initializer(camera, loader.imuCalibration());
  vio::VioInitializationResult init_result = initializer.initialize(window, imu_between_keyframes);
  if (!init_result.success) {
    std::cerr << "VioInitializer BASARISIZ: " << init_result.failure_reason << "\n";
    return 1;
  }
  std::cout << "VioInitializer basarili.\n";

  // --- TANI (DIAGNOSTIC): scale sorununun FactorGraphBackend/SmartFactor'dan
  // ONCE, VioInitializer'in HAM ciktisinda zaten var olup olmadigini kontrol
  // et. Baslangic penceresindeki (n keyframe) VioInitializer p_w_b
  // pozisyonlarinin toplam yol uzunlugunu, AYNI keyframe'lerin en yakin GT
  // pozisyonlarinin toplam yol uzunluguyla kiyaslayarak kaba bir "olcek"
  // orani hesapliyoruz. Bu, hatanin kok nedeninin initializer'in kendi
  // dogrusal hizalama sisteminde mi (ADIM C), yoksa daha sonraki
  // FactorGraphBackend/SmartFactor asamasinda mi oldugunu ayirt etmek icindir.
  {
    double vio_path_len = 0.0, gt_path_len = 0.0;
    Eigen::Vector3d prev_vio, prev_gt;
    bool have_prev = false;
    int matched_count = 0;
    for (const auto& kf : init_result.keyframes) {
      vio::GroundTruthPose gt;
      if (!loader.nearestGroundTruth(kf.timestamp_ns, gt)) continue;
      ++matched_count;
      if (have_prev) {
        vio_path_len += (kf.p_w_b - prev_vio).norm();
        gt_path_len += (gt.position - prev_gt).norm();
      }
      prev_vio = kf.p_w_b;
      prev_gt = gt.position;
      have_prev = true;
    }
    std::cout << "[TANI] Baslangic penceresi (" << matched_count << " eslesen keyframe): "
              << "VIO yol uzunlugu=" << vio_path_len << " m, GT yol uzunlugu=" << gt_path_len << " m, "
              << "kaba olcek orani (VIO/GT)=" << (gt_path_len > 1e-9 ? vio_path_len / gt_path_len : -1.0) << "\n";
  }

  vio::FactorGraphBackend backend(camera, loader.imuCalibration());
  backend.initializeWithWindow(window, imu_between_keyframes, init_result);

  // --- Trajectory gecmisi (gorsellestirme + nihai ATE icin) ---
  std::vector<vio::TimestampNs> vio_timestamps;
  std::vector<Eigen::Vector3d> vio_positions;
  std::vector<Eigen::Vector3d> gt_positions_vis;  // SADECE gorsellestirme icin, kestirime GIRMEZ
  for (const auto& kf : window) {
    appendTrajectoryPoint(backend, loader, kf.keyframe_id, kf.timestamp_ns, vio_timestamps, vio_positions,
                          gt_positions_vis);
  }

  int next_keyframe_id = static_cast<int>(window.size());
  vio::TimestampNs last_keyframe_ts = window.back().timestamp_ns;
  bool user_quit = false;
  vio::LiveMetrics live_metrics = computeLiveMetrics(vio_timestamps, vio_positions, loader);

  // --- Ana dongu: kalan tum kareler ---
  for (; frame_idx < cam.size() && !user_quit; ++frame_idx) {
    cv::Mat img = cv::imread(cam[frame_idx].image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) continue;

    vio::KeyframeDecision d = tracker.processFrame(img, cam[frame_idx].timestamp_ns);
    fps_counter.tick();

    if (d.is_keyframe) {
      vio::KeyframeObservation obs;
      obs.keyframe_id = next_keyframe_id;
      obs.timestamp_ns = cam[frame_idx].timestamp_ns;
      for (const auto& kv : tracker.activeTracks()) obs.observations[kv.first] = kv.second.current_pixel;

      std::vector<vio::ImuMeasurement> imu_since_prev = loader.imuBetween(last_keyframe_ts, obs.timestamp_ns);

      try {
        backend.addKeyframe(obs, imu_since_prev);
      } catch (const std::exception& e) {
        std::cerr << "\nBackend istisnasi (keyframe " << obs.keyframe_id << "): " << e.what()
                  << "\nIslenen trajectory ile devam ediliyor.\n";
        break;
      }

      appendTrajectoryPoint(backend, loader, obs.keyframe_id, obs.timestamp_ns, vio_timestamps, vio_positions,
                            gt_positions_vis);
      // Canli ATE/scale'i HER keyframe'de tazele (Umeyama SVD'si ucuz - bu,
      // FPS'i hissedilir sekilde etkilemez) - trajectory penceresinin
      // ustunde anlik takip icin (kullanicinin talebi).
      live_metrics = computeLiveMetrics(vio_timestamps, vio_positions, loader);

      last_keyframe_ts = obs.timestamp_ns;
      next_keyframe_id++;

      if (obs.keyframe_id % 20 == 0) {
        const vio::FactorGraphBackend::SmartFactorHealth health = backend.smartFactorHealth();
        std::cout << "Ilerleme: frame=" << frame_idx << " keyframe=" << obs.keyframe_id
                  << " FPS=" << fps_counter.fps() << " ATE=" << live_metrics.ate_rmse_m
                  << " scale=" << live_metrics.umeyama_scale
                  << " [TANI] gecerli_track=" << health.valid << "/" << health.total
                  << " (degenerate=" << health.degenerate << " arkada=" << health.behind_camera
                  << " uzak=" << health.far_point << " aykiri=" << health.outlier << ")" << std::endl;
      }
    }

    visualizer.showCameraFrame(img, tracker.currentKeypoints(), fps_counter.fps());
    visualizer.showTrajectory(vio_positions, gt_positions_vis, fps_counter.fps(), live_metrics);
    user_quit = visualizer.pollKeyboard();
  }

  std::cout << "\nToplam islenen keyframe sayisi: " << vio_positions.size() << "\n";
  std::cout << "Ortalama FPS: " << fps_counter.fps() << "\n";

  // --- Nihai ATE hesaplamasi (GtAligner + AteEvaluator, SADECE dogrulama amacli) ---
  const vio::MatchedTrajectories matched = vio::GtAligner::match(vio_timestamps, vio_positions, loader);
  if (matched.vio_positions.size() >= 3) {
    const vio::AlignmentResult rigid = vio::GtAligner::align(matched, /*with_scale=*/false);
    const vio::AlignmentResult sim3 = vio::GtAligner::align(matched, /*with_scale=*/true);
    const vio::AteResult ate = vio::AteEvaluator::evaluate(rigid.aligned_vio_positions, matched.gt_positions);

    std::cout << "\n--- ATE Degerlendirmesi (GtAligner + AteEvaluator) ---\n";
    std::cout << "ATE RMSE (rijit hizalama, hedef: 0.3-0.6 m): " << ate.rmse_m << " m\n";
    std::cout << "ATE ortalama: " << ate.mean_m << " m, medyan: " << ate.median_m << " m, max: " << ate.max_m
              << " m\n";
    std::cout << "Umeyama (Sim3) scale (hedef: ~1.0): " << sim3.scale << "\n";
  } else {
    std::cerr << "ATE hesaplamak icin yeterli eslesen (GT araligindaki) keyframe yok.\n";
  }

  return 0;
}
