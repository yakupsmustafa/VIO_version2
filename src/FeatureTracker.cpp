// =====================================================================================
// FeatureTracker.cpp
// ---------------------------------------------------------------------------
// FeatureTracker.hpp icinde bildirilen FeatureTracker sinifinin implementasyonu.
// =====================================================================================

#include "FeatureTracker.hpp"

#include <cmath>
#include <unordered_set>

namespace vio {

FeatureTracker::FeatureTracker(const OrbExtractionParams& extraction_params,
                                const MatcherParams& matcher_params,
                                const TrackerParams& tracker_params,
                                const PinholeCameraModel& camera)
    : extractor_(extraction_params),
      matcher_(matcher_params),
      camera_(camera),
      params_(tracker_params) {}

KeyframeDecision FeatureTracker::processFrame(const cv::Mat& image_gray, TimestampNs timestamp_ns) {
  (void)timestamp_ns;  // su an icin kullanilmiyor, ileride zaman-tabanli kriterler icin saklanabilir

  // Su anki karenin ORB ozelliklerini cikar.
  extractor_.extract(image_gray, cur_keypoints_, cur_descriptors_);

  KeyframeDecision decision;

  // --- Ilk kare (bootstrap) durumu ---
  // Henuz onceki kare yoksa eslestirme yapilamaz. Ilk kare her zaman keyframe
  // kabul edilir (sistemin baslangic referansi budur), ama henuz hicbir track
  // olusturulmaz - tracklar ilk eslesme (2. kare islendiginde) ortaya cikar.
  if (!has_prev_frame_) {
    has_prev_frame_ = true;
    keyframe_counter_ = 1;
    frames_since_last_keyframe_ = 0;
    tracked_count_at_last_keyframe_ = 0;

    decision.is_keyframe = true;
    decision.is_bootstrap = true;
    decision.reason = "ilk kare (baslangic keyframe'i)";

    prev_keypoints_ = cur_keypoints_;
    prev_descriptors_ = cur_descriptors_.clone();
    prev_keypoint_to_track_.clear();
    return decision;
  }

  // --- Onceki kareyle eslestirme ---
  // queryIdx -> prev_keypoints_ indeksi, trainIdx -> cur_keypoints_ indeksi.
  const std::vector<cv::DMatch> matches =
      matcher_.match(prev_keypoints_, prev_descriptors_, cur_keypoints_, cur_descriptors_, camera_);

  std::unordered_map<int, int> cur_keypoint_to_track;
  cur_keypoint_to_track.reserve(matches.size());

  // Bu turda DOGRUDAN eslesen prev_idx'leri ayri tutuyoruz - asagidaki yerel
  // kurtarma adiminda "bu track zaten eslesti mi" kontrolu icin gerekli.
  std::unordered_set<int> matched_prev_indices;
  matched_prev_indices.reserve(matches.size());

  for (const auto& m : matches) {
    const int prev_idx = m.queryIdx;
    const int cur_idx = m.trainIdx;
    const cv::Point2f& cur_pt = cur_keypoints_[cur_idx].pt;
    matched_prev_indices.insert(prev_idx);

    int track_id;
    auto it = prev_keypoint_to_track_.find(prev_idx);
    if (it != prev_keypoint_to_track_.end()) {
      // Bu nokta zaten bilinen bir track'e ait - devam ettiriyoruz.
      track_id = it->second;
      FeatureTrack& track = tracks_.at(track_id);
      track.current_pixel = cur_pt;
      track.age += 1;
    } else {
      // Bu, bu track'in ilk kez eslesmesi (bir onceki karede vardi ama henuz
      // bir track'e baglanmamisti) - yeni track olusturuyoruz.
      track_id = next_track_id_++;
      FeatureTrack track;
      track.id = track_id;
      track.current_pixel = cur_pt;
      track.age = 2;  // onceki karede + bu karede goruldu
      tracks_.emplace(track_id, track);
    }

    cur_keypoint_to_track[cur_idx] = track_id;
  }

  // --- YEREL KURTARMA (recovery) ---
  // Onceki karede aktif olan ama bu turda DOGRUDAN (matcher_.match uzerinden)
  // eslesmeyen track'ler icin son sans: son bilinen piksel konumu etrafinda
  // (recovery_radius_px) kalan (henuz baska bir track'e atanmamis) keypoint'
  // ler arasinda, track'in ONCEKI karedeki ORB tanimlayicisina (Hamming
  // mesafesiyle) EN YAKIN adayi ara. Yeterince yakinsa (recovery_max_
  // hamming_distance altinda) track'i CANLANDIR. Bu, RANSAC/Essential-Matrix
  // dogrulamasindan GECMEDIGI icin esikler bilinclu olarak KATI tutuluyor
  // (bkz. header aciklamasi).
  std::vector<bool> cur_idx_claimed(cur_keypoints_.size(), false);
  for (const auto& kv : cur_keypoint_to_track) cur_idx_claimed[kv.first] = true;

  for (const auto& kv : prev_keypoint_to_track_) {
    const int prev_idx = kv.first;
    const int track_id = kv.second;
    if (matched_prev_indices.count(prev_idx) > 0) continue;  // zaten dogrudan eslesti

    const auto track_it = tracks_.find(track_id);
    if (track_it == tracks_.end()) continue;  // guvenlik: teorik olarak olmamali

    const cv::Point2f last_pos = track_it->second.current_pixel;
    const cv::Mat track_desc = prev_descriptors_.row(prev_idx);
    const double radius_sq = params_.recovery_radius_px * params_.recovery_radius_px;

    int best_cur_idx = -1;
    double best_hamming = static_cast<double>(params_.recovery_max_hamming_distance) + 1.0;
    for (int cur_idx = 0; cur_idx < static_cast<int>(cur_keypoints_.size()); ++cur_idx) {
      if (cur_idx_claimed[cur_idx]) continue;
      const cv::Point2f diff = cur_keypoints_[cur_idx].pt - last_pos;
      const double dist_sq = static_cast<double>(diff.x) * diff.x + static_cast<double>(diff.y) * diff.y;
      if (dist_sq > radius_sq) continue;

      const double hamming = cv::norm(track_desc, cur_descriptors_.row(cur_idx), cv::NORM_HAMMING);
      if (hamming < best_hamming) {
        best_hamming = hamming;
        best_cur_idx = cur_idx;
      }
    }

    if (best_cur_idx >= 0) {
      cur_idx_claimed[best_cur_idx] = true;
      cur_keypoint_to_track[best_cur_idx] = track_id;
      FeatureTrack& track = tracks_.at(track_id);
      track.current_pixel = cur_keypoints_[best_cur_idx].pt;
      track.age += 1;
    }
  }

  // --- Parallax hesabi ---
  // SADECE bu track'in "son keyframe referansi" gecerliyse (yani bu track o
  // keyframe aninda da izleniyorduysa) anlamlidir. Yeni dogan tracklarin
  // henuz bir keyframe referansi yok, onlari parallax ortalamasina
  // katmiyoruz. Bu hesap, dogrudan VE yerel-kurtarma ile canlandirilmis TUM
  // track'leri kapsayacak sekilde final cur_keypoint_to_track uzerinden
  // yapiliyor.
  double parallax_sum = 0.0;
  int parallax_count = 0;
  for (const auto& kv : cur_keypoint_to_track) {
    const FeatureTrack& track_ref = tracks_.at(kv.second);
    if (track_ref.keyframe_ref_id == keyframe_counter_) {
      const double dx = track_ref.current_pixel.x - track_ref.keyframe_pixel.x;
      const double dy = track_ref.current_pixel.y - track_ref.keyframe_pixel.y;
      parallax_sum += std::sqrt(dx * dx + dy * dy);
      parallax_count += 1;
    }
  }

  // Bu karede (dogrudan VEYA yerel kurtarma ile) eslesmeyen tracklari
  // temizle - bellek buyumesini sinirlamak icin.
  std::unordered_map<int, int> track_id_still_alive;
  track_id_still_alive.reserve(cur_keypoint_to_track.size());
  for (const auto& kv : cur_keypoint_to_track) track_id_still_alive[kv.second] = 1;
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    if (track_id_still_alive.find(it->first) == track_id_still_alive.end()) {
      it = tracks_.erase(it);
    } else {
      ++it;
    }
  }

  const double avg_parallax = (parallax_count > 0) ? (parallax_sum / parallax_count) : 0.0;
  // NOT: matches.size() DEGIL cur_keypoint_to_track.size() kullaniliyor -
  // yerel kurtarma ile canlandirilmis track'ler de "takip edilen" sayilmali.
  const int tracked_count = static_cast<int>(cur_keypoint_to_track.size());
  frames_since_last_keyframe_ += 1;

  // --- Uc kriterin birlesimi ---
  bool trigger_parallax = (parallax_count > 0) && (avg_parallax >= params_.parallax_threshold_px);
  bool trigger_track_ratio =
      (tracked_count_at_last_keyframe_ > 0) &&
      (tracked_count < params_.min_tracked_ratio * tracked_count_at_last_keyframe_);
  bool trigger_track_absolute = tracked_count < params_.min_tracked_absolute;
  bool trigger_max_gap = frames_since_last_keyframe_ >= params_.max_frame_gap;

  const bool is_keyframe = trigger_parallax || trigger_track_ratio || trigger_track_absolute || trigger_max_gap;

  decision.is_keyframe = is_keyframe;
  decision.is_bootstrap = false;
  decision.avg_parallax_px = avg_parallax;
  decision.tracked_count = tracked_count;
  decision.frames_since_last_keyframe = frames_since_last_keyframe_;

  if (trigger_parallax) decision.reason += "[parallax esigi] ";
  if (trigger_track_ratio) decision.reason += "[takip orani dustu] ";
  if (trigger_track_absolute) decision.reason += "[takip sayisi mutlak esigin altinda] ";
  if (trigger_max_gap) decision.reason += "[maksimum kare araligi asildi] ";
  if (decision.reason.empty()) decision.reason = "keyframe degil";

  if (is_keyframe) {
    keyframe_counter_ += 1;
    frames_since_last_keyframe_ = 0;
    tracked_count_at_last_keyframe_ = tracked_count;

    // Su an aktif olan tum tracklarin keyframe referansini guncelle: bir
    // sonraki parallax hesabi bu ana gore yapilacak.
    for (auto& kv : cur_keypoint_to_track) {
      FeatureTrack& track = tracks_.at(kv.second);
      track.keyframe_pixel = track.current_pixel;
      track.keyframe_ref_id = keyframe_counter_;
    }
  }

  // Su anki kareyi "onceki kare" yap, bir sonraki cagriya hazirla.
  prev_keypoints_ = cur_keypoints_;
  prev_descriptors_ = cur_descriptors_.clone();
  prev_keypoint_to_track_ = std::move(cur_keypoint_to_track);

  return decision;
}

}  // namespace vio
