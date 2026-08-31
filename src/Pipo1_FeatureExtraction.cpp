// =====================================================================================
// Pipo1_FeatureExtraction.cpp
// ---------------------------------------------------------------------------
// Pipo1_FeatureExtraction.hpp içinde bildirilen OrbFeatureExtractor sınıfının
// gerçek implementasyonu. Grid-tabanlı + piramitli + adaptif eşikli FAST
// algılama, ardından OpenCV'nin kendi ORB tanımlayıcı hesaplayıcısı.
// =====================================================================================

#include "Pipo1_FeatureExtraction.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace vio {

OrbFeatureExtractor::OrbFeatureExtractor(const OrbExtractionParams& params)
    : params_(params) {
  // orb_ nesnesini SADECE tanımlayıcı (descriptor) hesaplamak için kullanacağız
  // (orb_->compute), kendi keypoint'lerimizi kendimiz bulacağız (orb_->detect
  // kullanılmayacak). Yine de scaleFactor/nLevels/edgeThreshold/patchSize
  // parametrelerinin bizim piramidimizle BİREBİR aynı olması şart: çünkü
  // orb_->compute() çağrıldığında OpenCV görüntüden kendi iç piramidini
  // yeniden kurar ve her keypoint'in .octave alanına bakarak "bu nokta hangi
  // piramit seviyesinden geliyor" diye karar verir. Piramit ayarları
  // tutmazsa yanlış seviyeden yanlış ölçekte bir yama (patch) okunur.
  orb_ = cv::ORB::create(params_.target_features, params_.scale_factor, params_.n_levels,
                          params_.edge_threshold, /*firstLevel=*/0, /*WTA_K=*/2,
                          cv::ORB::HARRIS_SCORE, params_.patch_size, params_.fast_threshold);

  scale_per_level_.resize(params_.n_levels);
  scale_per_level_[0] = 1.0f;
  for (int lvl = 1; lvl < params_.n_levels; ++lvl) {
    scale_per_level_[lvl] = scale_per_level_[lvl - 1] * params_.scale_factor;
  }
}

std::vector<cv::Mat> OrbFeatureExtractor::buildPyramid(const cv::Mat& image_gray) const {
  std::vector<cv::Mat> pyramid(params_.n_levels);
  pyramid[0] = image_gray;
  for (int lvl = 1; lvl < params_.n_levels; ++lvl) {
    cv::Size sz(cvRound(pyramid[lvl - 1].cols / params_.scale_factor),
                cvRound(pyramid[lvl - 1].rows / params_.scale_factor));
    cv::resize(pyramid[lvl - 1], pyramid[lvl], sz, 0, 0, cv::INTER_LINEAR);
  }
  return pyramid;
}

std::vector<int> OrbFeatureExtractor::computeFeaturesPerLevel() const {
  // Geometrik seri paylaşımı: en alttaki (en büyük, en detaylı) seviye en
  // fazla, en üstteki (en küçük) seviye en az özellik alır. Bu, ORB-SLAM2'nin
  // ORBextractor'ında kullanılan formülle aynı mantığa dayanır: her seviyenin
  // "payı", bir öncekinin (1/scale_factor) katı kadardır.
  std::vector<int> features_per_level(params_.n_levels);
  float factor = 1.0f / params_.scale_factor;
  float desired_per_scale =
      params_.target_features * (1.0f - factor) / (1.0f - std::pow(factor, params_.n_levels));

  int sum = 0;
  for (int lvl = 0; lvl < params_.n_levels - 1; ++lvl) {
    features_per_level[lvl] = cvRound(desired_per_scale);
    sum += features_per_level[lvl];
    desired_per_scale *= factor;
  }
  // Yuvarlama hatalarını telafi etmek için kalan tüm pay son seviyeye verilir.
  features_per_level[params_.n_levels - 1] = std::max(params_.target_features - sum, 0);
  return features_per_level;
}

std::vector<cv::KeyPoint> OrbFeatureExtractor::detectGrid(const std::vector<cv::Mat>& pyramid) const {
  std::vector<cv::KeyPoint> all_keypoints;
  const std::vector<int> features_per_level = computeFeaturesPerLevel();

  for (int lvl = 0; lvl < params_.n_levels; ++lvl) {
    const cv::Mat& level_img = pyramid[lvl];
    const int border = params_.edge_threshold;

    // Kenar payını çıkardıktan sonra kalan kullanılabilir alan.
    const int usable_w = level_img.cols - 2 * border;
    const int usable_h = level_img.rows - 2 * border;
    if (usable_w <= 0 || usable_h <= 0) continue;

    // Hücre sayısını piksel cinsinden hedef hücre boyutuna göre belirle.
    int n_cols = std::max(1, usable_w / params_.grid_cell_size);
    int n_rows = std::max(1, usable_h / params_.grid_cell_size);
    const float cell_w = static_cast<float>(usable_w) / n_cols;
    const float cell_h = static_cast<float>(usable_h) / n_rows;

    const int total_cells = n_cols * n_rows;
    // Bu seviyeye ayrılan payı hücrelere eşit dağıt; en az 1 nokta/hücre.
    const int quota_per_cell = std::max(1, cvRound(static_cast<float>(features_per_level[lvl]) / total_cells));

    std::vector<cv::KeyPoint> level_keypoints;
    level_keypoints.reserve(features_per_level[lvl] * 2);

    for (int r = 0; r < n_rows; ++r) {
      for (int c = 0; c < n_cols; ++c) {
        const int x0 = border + static_cast<int>(std::round(c * cell_w));
        const int y0 = border + static_cast<int>(std::round(r * cell_h));
        const int x1 = (c == n_cols - 1) ? (level_img.cols - border)
                                          : border + static_cast<int>(std::round((c + 1) * cell_w));
        const int y1 = (r == n_rows - 1) ? (level_img.rows - border)
                                          : border + static_cast<int>(std::round((r + 1) * cell_h));

        cv::Rect cell_rect(x0, y0, x1 - x0, y1 - y0);
        if (cell_rect.width <= 0 || cell_rect.height <= 0) continue;

        cv::Mat cell_img = level_img(cell_rect);

        std::vector<cv::KeyPoint> cell_kps;
        cv::FAST(cell_img, cell_kps, params_.fast_threshold, /*nonmaxSuppression=*/true);

        // Doku az olan bölgelerde (gökyüzü, düz zemin gibi) normal eşikle hiç
        // köşe bulunamayabilir. Bu hücreleri tamamen boş bırakmak yerine daha
        // toleranslı bir eşikle tekrar deniyoruz (ORB-SLAM'in de kullandığı
        // "adaptif eşik" fikri) - böylece dokusuz alanlarda bile birkaç zayıf
        // da olsa özellik noktası elde ederek görüntünün her yerinde takip
        // imkanı sağlıyoruz.
        if (cell_kps.empty()) {
          cv::FAST(cell_img, cell_kps, params_.min_fast_threshold, /*nonmaxSuppression=*/true);
        }
        if (cell_kps.empty()) continue;

        // Hücre içindeki en iyi (en yüksek response'lu) quota_per_cell kadar
        // noktayı tut, gerisini at.
        if (static_cast<int>(cell_kps.size()) > quota_per_cell) {
          std::partial_sort(cell_kps.begin(), cell_kps.begin() + quota_per_cell, cell_kps.end(),
                             [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                               return a.response > b.response;
                             });
          cell_kps.resize(quota_per_cell);
        }

        // Koordinatları hücre-yerelinden seviye-yerel tam görüntü koordinatına taşı.
        for (auto& kp : cell_kps) {
          kp.pt.x += x0;
          kp.pt.y += y0;
          kp.octave = lvl;
          kp.size = params_.patch_size * scale_per_level_[lvl];
        }
        level_keypoints.insert(level_keypoints.end(), cell_kps.begin(), cell_kps.end());
      }
    }

    // Seviye-yerel koordinatları seviye-0 (orijinal görüntü) koordinatına
    // çevir: üst seviyeler daha küçük görüntü olduğu için ölçek çarpanıyla
    // büyütmemiz gerekiyor.
    const float scale = scale_per_level_[lvl];
    if (scale != 1.0f) {
      for (auto& kp : level_keypoints) {
        kp.pt.x *= scale;
        kp.pt.y *= scale;
      }
    }

    all_keypoints.insert(all_keypoints.end(), level_keypoints.begin(), level_keypoints.end());
  }

  return all_keypoints;
}

void OrbFeatureExtractor::extract(const cv::Mat& image_gray,
                                   std::vector<cv::KeyPoint>& keypoints,
                                   cv::Mat& descriptors) const {
  const std::vector<cv::Mat> pyramid = buildPyramid(image_gray);
  keypoints = detectGrid(pyramid);

  // Seviye-0 koordinatına çevrilmiş noktalardan, orijinal görüntü sınırına
  // (edge_threshold kadar içeride) çok yakın olanları temizle. Piramidin üst
  // seviyelerinde bir kenar hücresi orijinal görüntüde farklı bir kenara denk
  // gelebiliyor, bu yüzden bu son kontrolü seviye-0 üzerinde tekrar yapıyoruz.
  cv::KeyPointsFilter::runByImageBorder(keypoints, image_gray.size(), params_.edge_threshold);

  // Tanımlayıcıları (descriptor) OpenCV'nin kendi ORB implementasyonuyla
  // hesapla. Bu adım aynı zamanda her noktanın yönünü (intensity centroid
  // açısı) de hesaplayıp descriptor'ı o yöne göre döndürür (rotation
  // invariance / dönme-değişmezliği).
  orb_->compute(image_gray, keypoints, descriptors);
}

}  // namespace vio
