// =====================================================================================
// LoopClosureDetector.cpp
// ---------------------------------------------------------------------------
// LoopClosureDetector.hpp icinde bildirilen sinifin implementasyonu.
// =====================================================================================

#include "LoopClosureDetector.hpp"

#include <map>
#include <utility>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace vio {

LoopClosureDetector::LoopClosureDetector(const PinholeCameraModel& camera, const LoopClosureParams& params)
    : camera_(camera), params_(params), matcher_(MatcherParams()) {}

LoopClosureResult LoopClosureDetector::detect(const KeyframeRecord& current, const KeyframeDatabase& db) {
  LoopClosureResult result;
  result.new_keyframe_id = current.keyframe_id;

  std::vector<int> candidates = db.candidateIds(current.keyframe_id, params_.min_keyframe_gap);

  // Maliyeti sinirlamak icin: cok fazla aday varsa, TUM gecmis araligini
  // kapsayan esit-aralikli (stride) bir alt-kume secilir (bkz. header'daki
  // "ONEMLI" notu - deneyerek kesfedildi, tum adaylari taramak FPS'i
  // hedefin cok altina dusuruyordu).
  if (static_cast<int>(candidates.size()) > params_.max_candidates_per_check) {
    std::vector<int> sampled;
    sampled.reserve(params_.max_candidates_per_check);
    const double stride = static_cast<double>(candidates.size()) / params_.max_candidates_per_check;
    for (int i = 0; i < params_.max_candidates_per_check; ++i) {
      sampled.push_back(candidates[static_cast<std::size_t>(i * stride)]);
    }
    candidates = std::move(sampled);
  }

  for (int old_id : candidates) {
    const KeyframeRecord& old_kf = db.at(old_id);
    if (old_kf.landmarks.empty()) continue;  // bu adayin hic 3D bilgisi yoksa PnP imkansiz

    // --- 1) UCUZ on-filtre: HAM (oran testi + simetrik kontrol) eslesme sayisi ---
    const std::vector<cv::DMatch> raw_matches =
        matcher_.matchRaw(old_kf.keypoints, old_kf.descriptors, current.keypoints, current.descriptors);
    if (static_cast<int>(raw_matches.size()) < params_.min_raw_matches) continue;

    // --- 2) Adayin piksel -> 3D nokta haritasini kur (bkz. header'daki "piksel-deger esitligi" notu) ---
    std::map<std::pair<float, float>, gtsam::Point3> pixel_to_point3;
    for (const auto& lm : old_kf.landmarks) {
      pixel_to_point3[{lm.pixel.x, lm.pixel.y}] = lm.point_w;
    }

    // --- 3) 2D(yeni keyframe)-3D(adayin dunyasi) eslesme kumesini olustur ---
    std::vector<cv::Point3f> object_points;
    std::vector<cv::Point2f> image_points;
    object_points.reserve(raw_matches.size());
    image_points.reserve(raw_matches.size());
    for (const auto& m : raw_matches) {
      const cv::Point2f& old_pixel = old_kf.keypoints[m.queryIdx].pt;
      const auto it = pixel_to_point3.find({old_pixel.x, old_pixel.y});
      if (it == pixel_to_point3.end()) continue;  // bu eslesen nokta 3D bilgisi olmayan bir keypoint
      const gtsam::Point3& p = it->second;
      object_points.emplace_back(static_cast<float>(p.x()), static_cast<float>(p.y()), static_cast<float>(p.z()));
      image_points.push_back(current.keypoints[m.trainIdx].pt);
    }
    if (static_cast<int>(object_points.size()) < params_.min_pnp_inliers) continue;

    // --- 4) Geometrik dogrulama: PnP + RANSAC ---
    cv::Mat rvec, tvec, inliers;
    // NOT (tutarlilik): FactorGraphBackend, SmartProjectionPoseFactor'e HAM
    // (distorsiyonu giderilmemis) piksel + distorsiyonsuz bir Cal3_S2 besliyor
    // (bkz. header'daki "Tutarlilik notu") - triangulate edilmis 3D noktalarla
    // TUTARLI kalmak icin burada da AYNI konvansiyonu (distorsiyon YOK) kullaniyoruz.
    const bool pnp_ok = cv::solvePnPRansac(object_points, image_points, camera_.K(), cv::Mat(), rvec, tvec,
                                            /*useExtrinsicGuess=*/false, /*iterationsCount=*/200,
                                            static_cast<float>(params_.pnp_reprojection_error_px),
                                            /*confidence=*/0.99, inliers);
    if (!pnp_ok || inliers.rows < params_.min_pnp_inliers) continue;

    // --- 5) Basarili: rvec/tvec'i gtsam::Pose3'e cevir ---
    // OpenCV konvansiyonu: X_kamera = R * X_dunya + t (yani (R,t) = T_kamera_dunya).
    // Kamera-govde (body) donusumu FactorGraphBackend'deki body_P_sensor_ ile
    // AYNI kurulumla hesaplaniyor (T_body_kamera).
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Matrix3d R_cw;
    cv::cv2eigen(R_cv, R_cw);
    Eigen::Vector3d t_cw(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    const gtsam::Pose3 T_camera_world(gtsam::Rot3(R_cw), t_cw);
    const gtsam::Pose3 T_world_camera = T_camera_world.inverse();

    const gtsam::Pose3 T_body_camera(gtsam::Rot3(camera_.T_BS().block<3, 3>(0, 0)), camera_.T_BS().block<3, 1>(0, 3));
    const gtsam::Pose3 T_world_body_new = T_world_camera.compose(T_body_camera.inverse());

    result.found = true;
    result.old_keyframe_id = old_id;
    result.T_world_new = T_world_body_new;
    result.num_inliers = inliers.rows;
    return result;
  }

  return result;
}

}  // namespace vio
