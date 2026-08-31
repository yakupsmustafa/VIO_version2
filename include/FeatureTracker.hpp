// =====================================================================================
// FeatureTracker.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   Pipo1 (ORB cikarimi) ve Pipo2 (eslestirme) modullerini bir araya getirip
//   kareler arasinda SUREKLI ozellik takibi yapmak ve KEYFRAME (anahtar kare)
//   secim kararini vermek.
//
//   "Takip" ne demek? ORB her karede sifirdan yeniden cikariliyor, yani ayni
//   fiziksel nokta iki farkli karede FARKLI keypoint indekslerine sahip olur.
//   Bu sinif, ardisik karelerdeki eslesmeleri kullanarak her fiziksel noktaya
//   sabit bir "track_id" atar ve bu id'yi kare kare tasir - boylece "bu nokta
//   kac karedir takip ediliyor" ve "bu nokta son keyframe'den bu yana ne kadar
//   kaydi (parallax)" gibi sorulara cevap verebiliriz.
//
//   KEYFRAME NEDEN GEREKLI? Her kareyi optimizasyon penceresine (factor graph)
//   eklemek hem gereksiz hesaplama hem de neredeyse ayni bakis acisindan
//   (kucuk baseline) triangulation yapmaya calismak demektir - bu da gurultulu
//   ve yanlis derinlik kestirimine yol acar. Bunun yerine sadece "yeterince
//   farkli" karaleri (keyframe) secip onlari optimize ediyoruz.
//
//   KEYFRAME KARARI (kullaniciyla birlikte netlestirildi) UC KRITERIN
//   HERHANGI BIRI TETIKLENIRSE verilir:
//     1) Parallax esigi: son keyframe'e gore ortalama piksel kaymasi belli
//        bir esigi asarsa (triangulation icin yeterli baseline var demektir)
//     2) Takip sayisi dususu: takip edilen ozellik sayisi son keyframe'dekinin
//        belli bir oraninin veya mutlak bir sayinin altina duserse (takip
//        sürekliliği tehlikeye giriyor demektir, hemen yeni referans lazim)
//     3) Maksimum kare araligi: cok uzun sure keyframe eklenmezse (ornegin
//        drone tamamen durgun kaldiginda), IMU tek basina surukleyebilir;
//        bu bir guvenlik agi (safety net) olarak calisir.
//
//   YEREL KURTARMA (recovery, 2026-08-30 eklendi): dogrudan frame-to-frame
//   eslesmesi basarisiz olan bir track, HEMEN kalici olarak olmek yerine,
//   son bilinen konumu etrafinda kucuk bir pencerede ORB tanimlayicisiyla
//   yeniden eslesme sansi kazanir (bkz. TrackerParams::recovery_*). Amac,
//   scale-sorunu arastirmasinda tespit edilen "track'lerin cogu cok kisa
//   omurlu, bu yuzden SmartFactor'a yeterli olcek bilgisi tasimiyor"
//   bulgusuna cozum aramak - bkz. proje hafizasi.
// =====================================================================================

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "CameraModel.hpp"
#include "DataLoader.hpp"
#include "Pipo1_FeatureExtraction.hpp"
#include "Pipo2_Matcher.hpp"

namespace vio {

// Tek bir fiziksel noktanin kareler arasi takip bilgisi.
struct FeatureTrack {
  int id = -1;
  cv::Point2f current_pixel;     // en son gorulduku karedeki piksel konumu
  cv::Point2f keyframe_pixel;    // son keyframe'deki piksel konumu (SADECE keyframe_ref_id gecerliyse anlamli)
  int keyframe_ref_id = -1;      // keyframe_pixel'in hangi keyframe sayacina ait oldugu (gecerlilik kontrolu icin)
  int age = 1;                   // ilk goruldugunden beri kesintisiz kac karede takip edildi
};

// Bir processFrame() cagrisinin sonucu: bu kare keyframe mi oldu, neden.
struct KeyframeDecision {
  bool is_keyframe = false;
  bool is_bootstrap = false;      // true ise: bu, islenen ilk karedir (baslangic keyframe'i)
  double avg_parallax_px = 0.0;
  int tracked_count = 0;
  int frames_since_last_keyframe = 0;
  std::string reason;              // hangi kriter(ler) tetikledi, Turkce aciklama (loglama/debug icin)
};

struct TrackerParams {
  double parallax_threshold_px = 20.0;  // bu kadar piksel ortalama kayma -> keyframe
  double min_tracked_ratio = 0.6;       // son keyframe'deki takip sayisinin bu orani altina dusme -> keyframe
  int min_tracked_absolute = 80;        // VEYA mutlak takip sayisi bunun altina dusme -> keyframe
  int max_frame_gap = 10;                // bu kadar kare gecti ve hala keyframe olmadiysa -> zorla keyframe

  // --- YEREL KURTARMA (recovery) parametreleri (2026-08-30 eklendi) ---
  // Amac: bir track'in DOGRUDAN frame-to-frame eslesmesi basarisiz olsa bile
  // (orn. ORB tanimlayicisi o karede hafifce degisti, ratio testini
  // gecemedi), onu HEMEN kalici olarak olduruvermek yerine, son bilinen
  // piksel konumu etrafinda KUCUK bir pencerede ORB tanimlayicisiyla
  // yeniden eslesme SANSI vermek - boylece daha uzun omurlu (daha yuksek
  // parallakli) track'ler elde edip SmartFactor'a daha guvenilir olcek
  // bilgisi tasinmasi hedefleniyor. Bu adim RANSAC/Essential-Matrix
  // dogrulamasindan GECMIYOR (tek basina, konum+tanimlayici benzerligiyle
  // karar veriliyor) - bu yuzden esikler BILINCLI OLARAK KATI tutuluyor
  // (dar yaricap + dusuk Hamming mesafesi) ki yanlis-pozitif riski dusuk
  // kalsin (bkz. Pipo2_Matcher esik gevsetme denemesinin BASARISIZ olmasi -
  // proje hafizasi: gevsek esikler bu sistemde net zarar veriyor).
  double recovery_radius_px = 15.0;        // son bilinen konum etrafinda aranacak yaricap (piksel)
  int recovery_max_hamming_distance = 40;  // ORB (256-bit) tanimlayicilar arasi izin verilen max Hamming mesafesi
};

// -------------------------------------------------------------------------------------
// FeatureTracker
// ---------------------------------------------------------------------------
// Kare kare cagrilir (processFrame). Icerde onceki karenin ozelliklerini
// saklar, yeni kareyle eslestirir, track ID'lerini gunceller ve keyframe
// karari verir.
// -------------------------------------------------------------------------------------
class FeatureTracker {
 public:
  FeatureTracker(const OrbExtractionParams& extraction_params,
                 const MatcherParams& matcher_params,
                 const TrackerParams& tracker_params,
                 const PinholeCameraModel& camera);

  // Yeni bir kareyi isler. image_gray: gri tonlamali goruntu.
  KeyframeDecision processFrame(const cv::Mat& image_gray, TimestampNs timestamp_ns);

  // Su anki karede aktif (basariyla eslesmis) tum tracklar. Triangulation ve
  // FactorGraphBackend gibi sonraki modullerin girdisi olacak.
  const std::unordered_map<int, FeatureTrack>& activeTracks() const { return tracks_; }

  // Su anki karenin ham ORB verisi (ileride ihtiyac olursa - orn. yeni
  // triangulation icin descriptor erisimi).
  const std::vector<cv::KeyPoint>& currentKeypoints() const { return cur_keypoints_; }
  const cv::Mat& currentDescriptors() const { return cur_descriptors_; }

 private:
  OrbFeatureExtractor extractor_;
  OrbMatcher matcher_;
  const PinholeCameraModel& camera_;
  TrackerParams params_;

  // Onceki karenin ozellikleri (bir sonraki processFrame cagrisinda kullanilir).
  std::vector<cv::KeyPoint> prev_keypoints_;
  cv::Mat prev_descriptors_;

  // Su anki karenin ozellikleri (disariya activeTracks/currentKeypoints ile acilir).
  std::vector<cv::KeyPoint> cur_keypoints_;
  cv::Mat cur_descriptors_;

  // keypoint_index -> track_id eslemesi. Her processFrame sonunda "su anki
  // kare" haritasi "onceki kare" haritasina donusur.
  std::unordered_map<int, int> prev_keypoint_to_track_;

  // Tum aktif tracklar, track_id ile erisilir.
  std::unordered_map<int, FeatureTrack> tracks_;

  int next_track_id_ = 0;
  int keyframe_counter_ = 0;
  int frames_since_last_keyframe_ = 0;
  int tracked_count_at_last_keyframe_ = 0;
  bool has_prev_frame_ = false;
};

}  // namespace vio
