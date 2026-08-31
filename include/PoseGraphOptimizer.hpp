// =====================================================================================
// PoseGraphOptimizer.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   TUM keyframe pozlarini (FactorGraphBackend'in sliding window ile
//   MARGINALIZE EDIP DONDURDUGU eskiler DAHIL) ayri, basit bir Pose3-only
//   GTSAM grafinda tutmak; bir loop closure bulundugunda bu grafa yeni bir
//   kisit (BetweenFactor) ekleyip TUM trajectory'yi (gecmis dahil) yeniden
//   hizalamak.
//
//   NEDEN AYRI BIR GRAF? FactorGraphBackend, gercek-zamanli calisabilmek icin
//   SADECE SON N keyframe'i aktif tutuyor (sliding window) - daha eskiler
//   marginalize edilip KALICI OLARAK DONDURULUYOR (bkz. FactorGraphBackend.hpp
//   - bu, bilincli bir tasarim: donmus pozlari "canlandirmaya" calismak
//   gecmiste sayisal kararsizliga yol acmisti). Loop closure'in TUM faydasi
//   ("eski, donmus pozlari da duzelt") ancak bu donmus pozlarin HALA
//   degistirilebilir oldugu AYRI bir yapida mumkun - bu yuzden bu modul ana
//   backend'e HIC DOKUNMAZ, ondan tamamen BAGIMSIZ/PARALEL calisir.
//
//   YAPI: her keyframe icin bir Pose3 degiskeni (X(i)) + bir onceki keyframe'e
//   goreli "odometry" BetweenFactor (FactorGraphBackend'in o anki tahmininden
//   turetilir) + (varsa) loop-closure BetweenFactor'leri. Ilk keyframe icin
//   bir PriorFactor (gauge freedom'i sabitlemek icin, FactorGraphBackend'deki
//   ile ayni mantik). Graf, SADECE bir loop closure eklendiginde (ucuz
//   degil ama NADIR bir olay oldugu icin sorun degil) Levenberg-Marquardt
//   ile TAMAMEN yeniden cozulur - bu olcekte (yuzlerce-binlerce Pose3
//   degiskeni) bu, incremental (iSAM2 gibi) bir cozucuye gerek duymayacak
//   kadar UCUZDUR.
// =====================================================================================

#pragma once

#include <unordered_map>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

namespace vio {

struct PoseGraphParams {
  // Ardisik keyframe'ler arasi "odometry" kisitinin guven duzeyi
  // (FactorGraphBackend'in kendi tahminine ne kadar guvendigimiz).
  double odometry_rot_sigma_rad = 0.05;
  double odometry_trans_sigma_m = 0.05;

  // Loop-closure kisitlarinin guven duzeyi - PnP tabanli oldugu ve tek bir
  // olcumden geldigi icin odometriden biraz daha GEVSEK tutuluyor.
  double loop_rot_sigma_rad = 0.10;
  double loop_trans_sigma_m = 0.15;

  // Ilk keyframe icin "capa" (anchor) prior'unun guven duzeyi (gauge freedom'i sabitler).
  double prior_rot_sigma_rad = 0.01;
  double prior_trans_sigma_m = 0.01;
};

// -------------------------------------------------------------------------------------
// PoseGraphOptimizer
// -------------------------------------------------------------------------------------
class PoseGraphOptimizer {
 public:
  explicit PoseGraphOptimizer(const PoseGraphParams& params = PoseGraphParams());

  // Yeni bir keyframe'i grafa ekler. raw_pose: FactorGraphBackend'in O ANKI
  // (henuz loop-closure'suz) tahmini - hem baslangic degeri hem de bir onceki
  // keyframe'e gore "odometry" kisitini turetmek icin kullanilir.
  void addKeyframe(int keyframe_id, const gtsam::Pose3& raw_pose);

  // Bir loop-closure kisitini ekler ve TUM grafi yeniden optimize eder.
  // T_world_new: LoopClosureDetector'in PnP ile bulduğu, PAYLASILAN dunya
  // cercevesindeki MUTLAK govde pozu (bkz. LoopClosureDetector.hpp) - bu
  // fonksiyon bunu kendi ic olarak X(old_keyframe_id)'in SU ANKI tahminine
  // gore GORELI bir kisita cevirir.
  void addLoopClosure(int old_keyframe_id, int new_keyframe_id, const gtsam::Pose3& T_world_new);

  // Bir keyframe'in GUNCEL (loop closure sonrasi DUZELTILMIS, hic loop
  // closure olmadiysa raw_pose ile AYNI) pozunu dondurur.
  gtsam::Pose3 correctedPoseAt(int keyframe_id) const;

  bool hasLoopClosure() const { return has_loop_closure_; }
  int loopClosureCount() const { return loop_closure_count_; }

 private:
  void reoptimize();

  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values values_;
  PoseGraphParams params_;
  int prev_keyframe_id_ = -1;
  bool has_loop_closure_ = false;
  int loop_closure_count_ = 0;
};

}  // namespace vio
