// =====================================================================================
// ImuPreintegration.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   Iki keyframe arasinda biriken (potansiyel olarak yuzlerce, 200Hz IMU'da
//   iki kare arasi ~10 ornek) IMU olcumunu TEK bir "goreli hareket ozeti"ne
//   (preintegrated measurement) sikistirmak.
//
//   NEDEN GEREKLI? Factor graph'a HER IMU ornegini ayri ayri eklemek hem
//   asiri yavas olur (yuzlerce ekstra degisken) hem de optimizasyon sirasinda
//   bias tahmini her degistiginde TUM IMU orneklerinin yeniden entegre
//   edilmesini gerektirir. "On-manifold preintegration" teorisi (Forster ve
//   ark., 2015) bu sorunu cozer: IMU olcumleri bias'tan BAGIMSIZ sekilde bir
//   kere entegre edilir, bias degisince de birinci-mertebe bir Jacobian
//   duzeltmesiyle (koca bir yeniden-hesaplama yapmadan) guncellenir.
//
//   GTSAM bu teoriyi hazir sunuyor: gtsam::PreintegratedCombinedMeasurements
//   IMU orneklerini biriktirir, gtsam::CombinedImuFactor de bu ozeti iki
//   ardisik durum (pose+hiz+bias) arasindaki kisit olarak factor graph'a ekler.
//   "Combined" olan versiyon (Imu +Bias birlikte), bias'in kendisinin de
//   zamanla nasil degisebilecegini (random walk) faktor icinde modelleyip
//   ayrica bir bias-bias factor'u eklemeyi gereksiz kilar.
// =====================================================================================

#pragma once

#include <memory>
#include <vector>

#include <gtsam/inference/Key.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>

#include "DataLoader.hpp"

namespace vio {

// NOT: bu GTSAM derlemesi std::shared_ptr degil boost::shared_ptr kullaniyor
// (MakeSharedU/MakeSharedD boost::shared_ptr donduruyor) - bu yuzden burada
// da boost::shared_ptr kullaniyoruz.
//
// ImuCalibration'daki (sensor.yaml'dan okunan) gurultu yogunluklarini,
// GTSAM'in PreintegrationCombinedParams'inin bekledigi kovaryans matrislerine
// cevirir. Yercekimi Z-up (yukari pozitif Z) dunya cercevesi varsayimiyla
// 9.81 m/s^2 olarak ayarlanir (EuRoC'un Vicon/Leica referans cercevesi Z-up'tir).
boost::shared_ptr<gtsam::PreintegrationCombinedParams> makePreintegrationParams(const ImuCalibration& imu_calib);

// -------------------------------------------------------------------------------------
// ImuPreintegrator
// ---------------------------------------------------------------------------
// Kullanim sekli: her yeni keyframe araligi baslarken reset() cagrilir
// (guncel bias tahminiyle), sonra o araliktaki tum IMU ornekleri
// integrateInterval() ile eklenir, en sonunda makeFactor() ile factor
// graph'a eklenecek CombinedImuFactor elde edilir.
// -------------------------------------------------------------------------------------
class ImuPreintegrator {
 public:
  explicit ImuPreintegrator(const ImuCalibration& imu_calib);

  // Yeni bir keyframe araligina baslarken cagrilir: birikmis entegrasyonu
  // sifirlar ve verilen (o anki en guncel tahmini) bias'i baslangic noktasi
  // olarak kullanir - preintegration bias'a gore linearize edildigi icin
  // baslangic bias'inin dogru olmasi onemlidir.
  void reset(const gtsam::imuBias::ConstantBias& current_bias);

  // t_start (onceki keyframe zamani) ile t_end (yeni keyframe zamani)
  // arasindaki TUM IMU orneklerini sirayla entegre eder. measurements,
  // DataLoader::imuBetween(t_start, t_end) ile alinmis olmali (yani
  // t_start < zaman_damgasi <= t_end kosulunu saglamali).
  void integrateInterval(const std::vector<ImuMeasurement>& measurements, TimestampNs t_start, TimestampNs t_end);

  // Su ana kadar biriken preintegration sonucunu, verilen pose/hiz/bias
  // anahtarlariyla (key) factor graph'a eklenecek CombinedImuFactor'e cevirir.
  // i: onceki keyframe'in durum anahtarlari, j: yeni keyframe'inkiler.
  gtsam::CombinedImuFactor makeFactor(gtsam::Key pose_i, gtsam::Key vel_i, gtsam::Key bias_i, gtsam::Key pose_j,
                                       gtsam::Key vel_j, gtsam::Key bias_j) const;

  // reset()'ten bu yana biriken toplam sure (saniye) - VioInitializer'da
  // yercekimi/olcek hizalamasi icin kullanilacak.
  double deltaTimeSeconds() const;

  const gtsam::PreintegratedCombinedMeasurements& preintegrated() const { return *pim_; }

 private:
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> params_;
  std::unique_ptr<gtsam::PreintegratedCombinedMeasurements> pim_;
};

}  // namespace vio
