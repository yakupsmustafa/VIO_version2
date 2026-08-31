// =====================================================================================
// KeyframeDatabase.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   Loop closure icin gerekli olan, HER keyframe'in TAM ORB keypoint/descriptor
//   kumesini VE o anki gecerli (triangulate edilebilmis) 3D landmark anlik
//   goruntusunu (FactorGraphBackend::LandmarkSnapshot) KALICI olarak saklamak.
//
//   NEDEN GEREKLI? FactorGraphBackend, sliding window (kayan pencere) ile
//   SADECE SON N keyframe'i aktif tutuyor - daha eskiler marginalize edilip
//   "donduruluyor" (bkz. FactorGraphBackend.hpp). Bir keyframe loop-closure
//   ADAYI olacak kadar eski oldugunda (yani cok once marginalize olmustur),
//   onun ORB verisi ve 3D landmark bilgisi ARTIK canli backend'den sorulamaz.
//   Bu yuzden HER keyframe eklendigi ANDA (hala tazeyken) bu bilgiyi buraya
//   SNAPSHOT olarak kaydediyoruz - loop closure denemesi cok daha sonra,
//   keyframe'in kendisi uzun zaman once marginalize olmus olsa bile, hala
//   bu kayittan calisabilir.
//
//   NOT: Bu modul SADECE bir veri deposu - eslestirme/geometrik dogrulama
//   mantigi burada YOK (bkz. LoopClosureDetector.hpp).
// =====================================================================================

#pragma once

#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "DataLoader.hpp"
#include "FactorGraphBackend.hpp"

namespace vio {

// Tek bir keyframe'in loop-closure icin gerekli TUM verisi.
struct KeyframeRecord {
  int keyframe_id = -1;
  TimestampNs timestamp_ns = 0;

  // TAM ORB kumesi (SADECE aktif track'ler DEGIL - FeatureTracker'in o anki
  // TUM cikarilan ozellikleri, yer-tanima eslestirmesi icin maksimum kapsama
  // sahip olmak amaciyla).
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;

  // Alt kume: bu keyframe'in gozlemledigi track'lerden, o anda GECERLI 3D
  // konuma sahip olanlar (bkz. FactorGraphBackend::snapshotValidLandmarks).
  std::vector<FactorGraphBackend::LandmarkSnapshot> landmarks;
};

// -------------------------------------------------------------------------------------
// KeyframeDatabase
// ---------------------------------------------------------------------------
// Basit bir kayit deposu: ekle, id ile eris, "yeterince eski" adaylari listele.
// -------------------------------------------------------------------------------------
class KeyframeDatabase {
 public:
  void add(KeyframeRecord record);

  // Loop-closure ADAYI olabilecek keyframe id'lerini dondurur: su anki
  // keyframe'den EN AZ min_keyframe_gap kadar eski olanlar (cok yakin zamanli
  // keyframe'ler zaten normal takip ile baglantili - bunlar "loop" degil,
  // sadece surekliligin kendisidir).
  std::vector<int> candidateIds(int current_keyframe_id, int min_keyframe_gap) const;

  const KeyframeRecord& at(int keyframe_id) const;
  bool contains(int keyframe_id) const { return records_.find(keyframe_id) != records_.end(); }
  std::size_t size() const { return records_.size(); }

 private:
  std::unordered_map<int, KeyframeRecord> records_;
  // Ekleniş sirasi (keyframe_id'ler zaten artan sirada eklendigi icin bu,
  // ayni zamanda id'ye gore sirali bir listedir) - candidateIds icin.
  std::vector<int> insertion_order_;
};

}  // namespace vio
