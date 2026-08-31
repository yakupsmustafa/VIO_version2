// =====================================================================================
// LoopClosureDetector.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   Yeni bir keyframe icin, KeyframeDatabase'deki YETERINCE ESKI adaylar
//   arasinda gercek bir "buradan daha once gectim" (loop closure) eslesmesi
//   arayip, bulunursa iki keyframe arasindaki METRIK (dogru olcekli) goreli
//   pozu dondurmek.
//
//   NEDEN Essential Matrix DEGIL, PnP? Essential Matrix + recoverPose SADECE
//   YON verir (t birim-normdur, gercek metrik olcek YOKTUR). Loop closure
//   kenarinin METRIK olmasi SART - aksi halde pose-graph'a YANLIS OLCEKLI bir
//   kisit eklemis oluruz (bu projede aylarca ugrastigimiz "yanlis olcek"
//   sorununu MANUEL olarak tekrar sokmus oluruz). Bunun yerine: ADAY (eski)
//   keyframe'in KENDI zamaninda triangulate edilmis GERCEK 3D noktalarini
//   (KeyframeDatabase'de saklanan LandmarkSnapshot'lar) kullanip, YENI
//   keyframe'in 2D gozlemleriyle solvePnPRansac calistiriyoruz - bu DOGRUDAN
//   metrik bir goreli poz verir (ORB-SLAM'in loop closure yaklasiminin temeli).
//
//   ALGORITMA (detect() icinde):
//     1) KeyframeDatabase'den YETERINCE ESKI (min_keyframe_gap) adaylari alir.
//     2) Her aday icin: OrbMatcher::matchRaw() (SADECE oran testi + simetrik
//        kontrol, Essential-Matrix RANSAC YOK - bkz. Pipo2_Matcher.hpp) ile
//        HAM eslesme sayisini bulur; azsa (min_raw_matches altinda) o adayi
//        UCUZ sekilde eler.
//     3) Eslesen noktalardan, adayin LandmarkSnapshot listesinde (piksel-deger
//        esitligiyle) 3D karsiligi olanlari toplar (2D-3D eslesme kumesi).
//     4) cv::solvePnPRansac ile geometrik dogrulama yapar; yeterli inlier
//        varsa (min_pnp_inliers) bunu GERCEK bir loop closure sayar.
//
//   MALIYET KONTROLU: bu sinif KENDISI bir "her N keyframe'de bir calis"
//   mekanizmasi TUTMAZ - bunun nedeni, en pahali islemin aslinda burasi degil
//   FactorGraphBackend::snapshotValidLandmarks() (her keyframe'in KENDI
//   landmark'larini triangulate etme) olmasi (deneyerek kesfedildi - bu
//   "her zaman acik" maliyet, throttle'lanan arama maliyetinden bile daha
//   fazla FPS dusuruyordu). Bu yuzden hem KeyframeDatabase'e KAYIT hem de
//   detect() cagrisi, main.cpp'de TEK bir ortak "her N keyframe'de bir"
//   araligiyla birlikte kontrol ediliyor (bkz. main.cpp'deki
//   loop_closure_interval_keyframes).
//
//   ONEMLI: PnP'nin urettigi poz, adayin 3D noktalarinin ZATEN icinde oldugu
//   PAYLASILAN (tum sistemin ortak) dunya cercevesinde MUTLAK bir govde
//   pozudur (T_world_new) - "eskiye GORELI" bir donusum DEGILDIR. Bunu
//   PoseGraphOptimizer'in kendi X(eski) tahminine gore GORELIYE cevirmesi
//   gerekir (bkz. PoseGraphOptimizer::addLoopClosure).
// =====================================================================================

#pragma once

#include "CameraModel.hpp"
#include "KeyframeDatabase.hpp"
#include "Pipo2_Matcher.hpp"

#include <gtsam/geometry/Pose3.h>

namespace vio {

struct LoopClosureParams {
  int min_keyframe_gap = 50;  // bu kadar keyframe'den daha YENI adaylar loop sayilmaz (atlanir)
  int min_raw_matches = 30;   // PnP denemeden ONCE ucuz bir on-filtre (bkz. adim 3)
  int min_pnp_inliers = 20;         // PnP-RANSAC sonrasi kabul esigi
  double pnp_reprojection_error_px = 6.0;

  // ONEMLI (deneyerek kesfedildi): aday sayisi calisma ilerledikce (yuzlerce
  // keyframe biriktikce) DOGRUSAL olarak buyuyor - TUM adaylari her seferinde
  // taramak (her biri icin tam ORB eslestirme) FPS'i hedefin (30-60) COK
  // altina (tek hanelere) dusurdu. Bu yuzden bir detect() cagrisinda EN FAZLA
  // bu kadar aday kontrol edilir - adaylar bu sayidan fazlaysa, TUM gecmis
  // araligini kapsayacak sekilde ESIT ARALIKLI (stride) bir alt-kume secilir
  // (en yakin/en olasi adaylari tahmin etmeye calismak yerine - cunku "en
  // olasi" konumu bilmek zaten drift'ten etkilenmis bir tahmine guvenmek
  // demektir, bu da amaca aykiridir).
  int max_candidates_per_check = 25;
};

struct LoopClosureResult {
  bool found = false;
  int old_keyframe_id = -1;
  int new_keyframe_id = -1;
  // Paylasilan dunya cercevesinde MUTLAK govde pozu (T_world_new) - eskiye
  // GORELI bir donusum DEGIL (bkz. yukaridaki "ONEMLI" notu).
  gtsam::Pose3 T_world_new;
  int num_inliers = 0;
};

// -------------------------------------------------------------------------------------
// LoopClosureDetector
// -------------------------------------------------------------------------------------
class LoopClosureDetector {
 public:
  explicit LoopClosureDetector(const PinholeCameraModel& camera,
                                const LoopClosureParams& params = LoopClosureParams());

  // current: KeyframeDatabase'e YENI eklenmis olan (henuz kontrol edilmemis)
  // keyframe kaydi. db: TUM gecmis kayitlarin deposu (current DAHIL olabilir,
  // fonksiyon kendi kendiyle eslesmeyi otomatik atlar).
  LoopClosureResult detect(const KeyframeRecord& current, const KeyframeDatabase& db);

 private:
  const PinholeCameraModel& camera_;
  LoopClosureParams params_;
  OrbMatcher matcher_;
};

}  // namespace vio
