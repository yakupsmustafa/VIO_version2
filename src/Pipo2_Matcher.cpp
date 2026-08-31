// =====================================================================================
// Pipo2_Matcher.cpp
// ---------------------------------------------------------------------------
// Pipo2_Matcher.hpp icinde bildirilen OrbMatcher sinifinin implementasyonu.
// =====================================================================================

#include "Pipo2_Matcher.hpp"

#include <unordered_map>

#include <opencv2/calib3d.hpp>

namespace vio {

OrbMatcher::OrbMatcher(const MatcherParams& params) : params_(params) {
  // crossCheck=false: kendi simetrik eslestirmemizi (oran testi + karsilikli
  // kontrol) elle yapacagimiz icin OpenCV'nin hazir cross-check ozelligini
  // kullanmiyoruz - boylece once oran testi sonra simetri kontrolu gibi iki
  // asamayi ayri ayri uygulayabiliyoruz.
  matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, /*crossCheck=*/false);
}

std::vector<cv::DMatch> OrbMatcher::ratioTestMatch(const cv::Mat& query_desc,
                                                    const cv::Mat& train_desc) const {
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher_->knnMatch(query_desc, train_desc, knn_matches, /*k=*/2);

  std::vector<cv::DMatch> good_matches;
  good_matches.reserve(knn_matches.size());
  for (const auto& pair : knn_matches) {
    // Bazi noktalarin (train_desc çok kucukse) 2. en yakin komsusu
    // olmayabilir; bu durumda oran testi uygulanamaz, atlanir.
    if (pair.size() < 2) continue;
    if (pair[0].distance < params_.ratio_test_threshold * pair[1].distance) {
      good_matches.push_back(pair[0]);
    }
  }
  return good_matches;
}

std::vector<cv::DMatch> OrbMatcher::match(const std::vector<cv::KeyPoint>& kpts1, const cv::Mat& desc1,
                                           const std::vector<cv::KeyPoint>& kpts2, const cv::Mat& desc2,
                                           const PinholeCameraModel& camera) const {
  // --- 1) Oran testi, her iki yonde ---
  const std::vector<cv::DMatch> forward = ratioTestMatch(desc1, desc2);   // 1 -> 2
  const std::vector<cv::DMatch> backward = ratioTestMatch(desc2, desc1);  // 2 -> 1

  // --- 2) Simetrik (karsilikli) kontrol ---
  // backward_map[j] = i demek: 2 numarali karedeki j noktasi, oran testine
  // gore en iyi 1 numarali karedeki i noktasiyla eslesiyor. forward yondeki
  // bir eslesme (i -> j) sadece backward_map[j] == i ise "karsilikli
  // dogrulanmis" sayilir.
  std::unordered_map<int, int> backward_map;
  backward_map.reserve(backward.size());
  for (const auto& bm : backward) {
    backward_map[bm.queryIdx] = bm.trainIdx;  // queryIdx: kpts2 indeksi, trainIdx: kpts1 indeksi
  }

  std::vector<cv::DMatch> symmetric_matches;
  symmetric_matches.reserve(forward.size());
  for (const auto& fm : forward) {
    auto it = backward_map.find(fm.trainIdx);  // fm.trainIdx: kpts2 indeksi
    if (it != backward_map.end() && it->second == fm.queryIdx) {
      symmetric_matches.push_back(fm);
    }
  }

  // --- 3) Geometrik dogrulama (RANSAC + Essential Matrix) ---
  if (static_cast<int>(symmetric_matches.size()) < params_.min_points_for_ransac) {
    // Guvenilir bir Essential Matrix kurmak icin yeterli nokta yok; elimizdeki
    // (zaten simetrik dogrulamadan gecmis) eslesmeleri oldugu gibi donduruyoruz.
    return symmetric_matches;
  }

  std::vector<cv::Point2f> pts1, pts2;
  pts1.reserve(symmetric_matches.size());
  pts2.reserve(symmetric_matches.size());
  for (const auto& m : symmetric_matches) {
    pts1.push_back(kpts1[m.queryIdx].pt);
    pts2.push_back(kpts2[m.trainIdx].pt);
  }

  // RANSAC oncesi lens distorsiyonunu gideriyoruz (piksel biriminde kalarak,
  // K matrisiyle tutarli threshold kullanabilmek icin).
  const std::vector<cv::Point2f> pts1_undist = camera.undistortPixel(pts1);
  const std::vector<cv::Point2f> pts2_undist = camera.undistortPixel(pts2);

  cv::Mat inlier_mask;
  cv::findEssentialMat(pts1_undist, pts2_undist, camera.K(), cv::RANSAC,
                        params_.ransac_confidence, params_.ransac_threshold_px, inlier_mask);

  std::vector<cv::DMatch> inlier_matches;
  inlier_matches.reserve(symmetric_matches.size());
  for (int i = 0; i < inlier_mask.rows; ++i) {
    if (inlier_mask.at<uchar>(i)) {
      inlier_matches.push_back(symmetric_matches[i]);
    }
  }
  return inlier_matches;
}

}  // namespace vio
