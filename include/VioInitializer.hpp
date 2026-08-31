// =====================================================================================
// VioInitializer.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   VIO sistemini "sifirdan" baslatmak: iSAM2 factor graph'ina ilk kez tutarli
//   bir pozlar+hizlar+bias tahmini vermeden once, bir pencere (window) dolusu
//   keyframe kullanarak METRIK OLCEK, YERCEKIMI YONU, HER KEYFRAME'IN HIZI ve
//   BASLANGIC GYRO BIAS'INI kestirmek.
//
//   NEDEN GEREKLI? Mono kamera TEK BASINA olcek bilgisi veremez. IMU ise KISA
//   surede cok hassas ama UZUN surede yerçekimi ve bias hatalari yuzunden
//   hizla sürüklenir (drift). Bu ikisini dogru sekilde birlestirmek icin,
//   optimizasyona baslamadan once "kaba ama tutarli" bir baslangic noktasi
//   (initialization) bulmamiz sart.
//
//   ONEMLI TASARIM NOTU (deneyerek ogrenildi): Ilk tasarimimiz, pencerenin
//   EN GENIS taban mesafeli iki keyframe'i arasinda tek bir Essential
//   Matrix + triangulation ile "olceksiz bir 3D harita" kurup, diger tum
//   keyframe'leri buna karsi PnP ile cozmekti (klasik VI-SfM/VINS-Mono
//   yaklasimi). Ancak FeatureTracker'imiz BILEREK kati (bir track ara
//   karelerin herhangi birinde eslesmezse kalici olarak olur, yeniden-tespit
//   yok), bu yuzden 2+ keyframe arayla ortak track sayisi hizla SIFIRA
//   dusuyor (olcum: (k,k+1) ciftlerinde 16-45 ortak track, (k,k+2)'de 2-14,
//   (k,k+3)+ pratikte ~0). Yani "genis taban" secimi is basitce COKMEZDI.
//
//   Bunun yerine SADECE ARDISIK keyframe ciftlerinin (guclu ortusmeye sahip)
//   gorsel bilgisini kullanan, daha basit VE daha saglam bir formulasyon
//   kullaniyoruz:
//
//   ADIM A - Ardisik Goreli Pozlar (Essential Matrix):
//     Her ardisik (k,k+1) cifti icin Essential Matrix ile goreli ROTASYON
//     (tam dogru, olcekten bagimsiz) ve goreli OTELEME YONU (birim norm,
//     olceksiz) cikarilir. Rotasyonlar zincirlenerek TUM keyframe'lerin
//     mutlak rotasyonu (R_c0_ck) elde edilir - bu islem olcekten tamamen
//     bagimsiz oldugu icin sorunsuz zincirlenebilir.
//
//   ADIM B - Gyro Bias Kestirimi:
//     Ardisik keyframe'ler arasindaki rotasyon, hem gorselden (Adim A) hem
//     de IMU'dan (preintegration) bilinir. Bu ikisi arasindaki fark, gyro
//     bias tahminimizin ne kadar yanlis oldugunu gosterir - GTSAM'in
//     preintegration Jacobian'i kullanilarak dogrusal bir en kucuk kareler
//     cozumu uygulanir.
//
//   ADIM C - Olcek + Yercekimi + Hiz Hizalamasi (Dogrusal):
//     Her ardisik cift icin, IMU'nun onerdigi goreli konum/hiz degisimi ile
//     gorselin onerdigi YON (ve o cifte ozel bilinmeyen bir s_k olcek
//     katsayisi) arasinda bir denklem kurulur. Bilinmeyenler: her keyframe'in
//     hizi + yercekimi vektoru + HER CIFT icin ayri bir olcek katsayisi. Bu
//     sistem tek seferde (Eigen ile en kucuk kareler) cozulur. Per-cift
//     olcek kullanmak, "tum pencere boyunca tek bir tutarli olceksiz 3D
//     harita" gereksinimini ORTADAN KALDIRIR - sadece ardisik ciftlerin
//     gorsel bilgisine ihtiyac duyulur.
//
//   SON ADIM - Mutlak Pozisyonlar (IMU ileri-entegrasyon) + Yercekimi Hizalamasi:
//     v_k ve g_c0 cozuldukten sonra, TUM keyframe'lerin mutlak pozisyonu
//     dogrudan IMU preintegration denklemiyle (p_{k+1}=p_k+v_k*dt+0.5*g*dt^2+R_k*alpha_k)
//     ILERI DOGRU entegre edilerek bulunur - gorsel oteleme zincirlemeye HIC
//     gerek kalmaz. Son olarak TUM pozlar/hizlar, yercekiminin tam -Z
//     ekseninde oldugu bir "dunya" cercevesine donusturulur (yaw keyfi
//     olarak ilk keyframe'in gorsel yonune sabitlenir).
// =====================================================================================

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <gtsam/navigation/ImuBias.h>
#include <opencv2/core.hpp>

#include "CameraModel.hpp"
#include "DataLoader.hpp"
#include "ImuPreintegration.hpp"

namespace vio {

// Tek bir keyframe'in gozlemleri: o an aktif olan TUM tracklarin piksel
// konumlari, kalici track_id ile anahtarli (FeatureTracker::activeTracks()'tan
// turetilir). Cagiran kod, her keyframe karari verildiginde bu yapiyi
// doldurup bir pencere (window) biriktirmelidir.
struct KeyframeObservation {
  int keyframe_id = -1;
  TimestampNs timestamp_ns = 0;
  std::unordered_map<int, cv::Point2f> observations;  // track_id -> piksel (distorsiyonlu, ham)
};

// Baslatma basarili olduysa, her keyframe icin hesaplanan METRIK (olcekli,
// yercekimi-hizali dunya cercevesinde) durum.
struct InitializedKeyframe {
  int keyframe_id = -1;
  TimestampNs timestamp_ns = 0;
  Eigen::Matrix3d R_w_b = Eigen::Matrix3d::Identity();  // govde (IMU) rotasyonu, dunya cercevesinde
  Eigen::Vector3d p_w_b = Eigen::Vector3d::Zero();      // govde pozisyonu, metre, dunya cercevesinde
  Eigen::Vector3d v_w = Eigen::Vector3d::Zero();        // govde hizi, m/s, dunya cercevesinde
};

struct VioInitializationResult {
  bool success = false;
  std::string failure_reason;  // basarisizsa NEDEN oldugu (Turkce, loglama icin)
  gtsam::imuBias::ConstantBias initial_bias;  // kestirilen baslangic bias'i (sadece gyro; accel=0)
  std::vector<InitializedKeyframe> keyframes;  // window ile AYNI boyutta ve sirada
};

struct VioInitializerParams {
  int min_common_tracks_for_essential = 15;  // ardisik cift icin Essential Matrix'te gereken min ortak track
  int min_essential_inliers = 5;               // recoverPose sonrasi gereken min inlier (5-point algoritmasinin teorik minimumu) (mutlak sayi, orana gore degil -
                                               // kisa-tabanli ardisik ciftlerde dusuk inlier orani dogal olabilir,
                                               // rotasyon+yon bilgisi yine de kullanilabilir kalitede olur)
  double essential_ransac_threshold_px = 3.0;  // kisa-tabanli ardisik ciftlerde daha toleransli olmak icin gevsetildi
  double essential_ransac_confidence = 0.999;
};

// -------------------------------------------------------------------------------------
// VioInitializer
// -------------------------------------------------------------------------------------
class VioInitializer {
 public:
  VioInitializer(const PinholeCameraModel& camera, const ImuCalibration& imu_calib,
                 const VioInitializerParams& params = VioInitializerParams());

  // window: zaman sirali keyframe gozlemleri (en az 5, onerilen ~10).
  // imu_between_keyframes[k]: window[k] ile window[k+1] arasindaki IMU
  // olcumleri (DataLoader::imuBetween(window[k].timestamp_ns, window[k+1].timestamp_ns)
  // ile alinmis olmali). Boyutu window.size()-1 olmalidir.
  VioInitializationResult initialize(const std::vector<KeyframeObservation>& window,
                                      const std::vector<std::vector<ImuMeasurement>>& imu_between_keyframes) const;

 private:
  // Iki ardisik keyframe arasindaki Essential-Matrix tabanli goreli poz.
  // R_step, t_step: recoverPose'un DOGRUDAN ciktisi, yani X_b = R_step*X_a+t_step
  // (a: onceki keyframe'in kamera cercevesi, b: sonrakinin). t_step birim normdur
  // (OLCEKSIZ) - gercek fiziksel buyuklugu ADIM C'de per-cift s_k ile cozulur.
  struct PairwiseVision {
    bool success = false;
    std::string failure_reason;
    Eigen::Matrix3d R_step = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_step = Eigen::Vector3d::Zero();
  };
  PairwiseVision estimateRelativePose(const KeyframeObservation& a, const KeyframeObservation& b) const;

  // ADIM B: R_c0_ck (Adim A'da zincirlenmis, TUM keyframe'lerin mutlak
  // rotasyonu) ile IMU (sifir-bias preintegration) rotasyonlari arasindaki
  // ardisik farktan gyro bias duzeltmesini dogrusal en kucuk kareler ile cozer.
  Eigen::Vector3d solveGyroBias(const std::vector<Eigen::Matrix3d>& R_c0_ck,
                                const std::vector<std::vector<ImuMeasurement>>& imu_between_keyframes,
                                const std::vector<KeyframeObservation>& window) const;

  // ADIM C sonucu: yercekimi (c0 cercevesinde) ve her keyframe'in hizi (c0 cercevesinde).
  struct LinearAlignmentResult {
    bool success = false;
    std::string failure_reason;
    Eigen::Vector3d gravity_c0 = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> velocities_c0;
  };
  // position_reliable[k]: pairwise[k]'nin gorsel ESTIMASYONU basarili miydi
  // (Essential Matrix yeterli inlier ile cozulebildi mi). false ise (orn.
  // saf-donus/pure-rotation gibi dejenere bir durum), o ciftin pozisyon
  // denklemi ATLANIR (gorsel oteleme yonu guvenilmez) ama hiz denklemi
  // (SADECE IMU'ya dayandigi icin) yine de kullanilir.
  // fixed_gravity: nullptr ise yercekimi de bilinmeyen olarak cozulur (ilk
  // gecis). Bir deger verilirse (buyuklugu 9.81'e duzeltilmis, ikinci gecis),
  // yercekimi SABIT kabul edilir (bilinmeyenlerden cikarilir, denklemin bilinen
  // tarafina tasinir) ve SADECE hizlar/olcekler bu duzeltilmis yercekimine gore
  // YENIDEN cozulur - boylece hizlar, ilk cozumdeki (9.81'den sapmis) yercekimi
  // buyuklugu yuzunden olusan kucuk tutarsizliktan arindirilir (deneyerek
  // kesfedilen bir dogruluk sorunuydu - bkz. initialize()'daki yorum).
  LinearAlignmentResult linearAlignment(const std::vector<PairwiseVision>& pairwise,
                                        const std::vector<bool>& position_reliable,
                                        const std::vector<Eigen::Matrix3d>& R_c0_ck,
                                        const std::vector<std::vector<ImuMeasurement>>& imu_between_keyframes,
                                        const std::vector<KeyframeObservation>& window,
                                        const gtsam::imuBias::ConstantBias& bias,
                                        const Eigen::Vector3d* fixed_gravity = nullptr) const;

  // g_c0 (yaklasik 9.81 buyuklukte, yonu c0 cercevesinde bilinmeyen) verilip,
  // yercekiminin tam -Z ekseninde oldugu bir "dunya" cercevesine gecis
  // rotasyonunu (R_w_c0) hesaplar. Yaw keyfi oldugu icin c0'in orijinal
  // yonune gore sifirlanir (sadece gorsellestirme/yorumlanabilirlik icin,
  // matematiksel olarak herhangi bir yaw secimi es-degerdir).
  Eigen::Matrix3d computeGravityAlignmentRotation(const Eigen::Vector3d& g_c0) const;

  const PinholeCameraModel& camera_;
  ImuCalibration imu_calib_;
  VioInitializerParams params_;
};

}  // namespace vio
