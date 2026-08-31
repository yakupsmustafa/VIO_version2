// =====================================================================================
// ImuPreintegration.cpp
// ---------------------------------------------------------------------------
// ImuPreintegration.hpp icinde bildirilen fonksiyon ve sinifin implementasyonu.
// =====================================================================================

#include "ImuPreintegration.hpp"

namespace vio {

boost::shared_ptr<gtsam::PreintegrationCombinedParams> makePreintegrationParams(const ImuCalibration& imu_calib) {
  // MakeSharedU(g): "Up" (Z-up) dunya cercevesi konvansiyonu, yercekimi
  // vektorunu otomatik olarak (0,0,-g) yapar.
  auto params = gtsam::PreintegrationCombinedParams::MakeSharedU(9.81);

  const double accel_sigma = imu_calib.accel_noise_density;
  const double gyro_sigma = imu_calib.gyro_noise_density;
  const double accel_bias_rw = imu_calib.accel_random_walk;
  const double gyro_bias_rw = imu_calib.gyro_random_walk;

  // sensor.yaml'daki degerler "noise density" (surekli-zaman gurultu yogunlugu,
  // birim: .../sqrt(Hz)) - GTSAM kovaryans (varyans) bekledigi icin karesini aliyoruz.
  params->accelerometerCovariance = gtsam::Matrix3::Identity() * (accel_sigma * accel_sigma);
  params->gyroscopeCovariance = gtsam::Matrix3::Identity() * (gyro_sigma * gyro_sigma);
  params->biasAccCovariance = gtsam::Matrix3::Identity() * (accel_bias_rw * accel_bias_rw);
  params->biasOmegaCovariance = gtsam::Matrix3::Identity() * (gyro_bias_rw * gyro_bias_rw);

  // integrationCovariance: sayisal entegrasyon belirsizligi icin kucuk bir
  // sabit (kalibrasyon dosyasinda verilmez, literaturde standart bir deger).
  params->integrationCovariance = gtsam::Matrix3::Identity() * 1e-8;

  // biasAccOmegaInt: "bias'in baslangic tahmininin kovaryansi" (preintegration
  // sirasinda bias belirsizliginin pozisyon/hiz/rotasyon kovaryansina
  // ikinci-mertebe katkisini hesaplamak icin kullanilir). GTSAM'in kendi
  // varsayilani (I_6x6, yani 1.0) DENENDI ama scale sorununu (ayri, hala
  // arastirilan bir konu - bkz. proje hafizasi) DAHA DA KOTULESTIRDI (IMU
  // factor'u ASIRI gevsetip sistemi scale acisindan daha da az kisitli hale
  // getirdi, crash daha ERKEN olustu). 1e-5 (siki) burada TUTULUYOR - iki
  // asiri uc da (1e-5 ve 1.0) scale sorununu tam cozmuyor, bu parametre
  // SCALE SORUNUNUN KOK NEDENI DEGIL, sadece semptomu hafifce etkileyen bir
  // ayar oldugu sonucuna varildi.
  params->biasAccOmegaInt = gtsam::Matrix6::Identity() * 1e-5;

  return params;
}

ImuPreintegrator::ImuPreintegrator(const ImuCalibration& imu_calib) {
  params_ = makePreintegrationParams(imu_calib);
  pim_ = std::make_unique<gtsam::PreintegratedCombinedMeasurements>(params_, gtsam::imuBias::ConstantBias());
}

void ImuPreintegrator::reset(const gtsam::imuBias::ConstantBias& current_bias) {
  pim_->resetIntegrationAndSetBias(current_bias);
}

void ImuPreintegrator::integrateInterval(const std::vector<ImuMeasurement>& measurements, TimestampNs t_start,
                                          TimestampNs t_end) {
  TimestampNs prev_t = t_start;
  for (const auto& m : measurements) {
    const double dt = nsToSeconds(m.timestamp_ns - prev_t);
    // dt<=0 teorik olarak olmamali (imuBetween sirali ve t_start'tan sonraki
    // ornekleri dondurur) ama savunmaci bir kontrol: GTSAM sifir/negatif dt
    // ile cagrilirsa hata firlatir.
    if (dt > 0.0) {
      pim_->integrateMeasurement(m.accel, m.gyro, dt);
    }
    prev_t = m.timestamp_ns;
  }
  // NOT: kamera zaman damgasi (t_end) ile ona en yakin IMU orneginin zaman
  // damgasi tam ortusmeyebilir; bu durumda aradaki (tipik olarak <5ms) fark
  // entegre edilmeden birakilir. EuRoC gibi senkron tetiklenen verisetlerinde
  // bu fark ihmal edilebilir duzeydedir; bu, alanda yaygin kabul goren bir
  // basitlestirmedir.
  (void)t_end;
}

gtsam::CombinedImuFactor ImuPreintegrator::makeFactor(gtsam::Key pose_i, gtsam::Key vel_i, gtsam::Key bias_i,
                                                       gtsam::Key pose_j, gtsam::Key vel_j, gtsam::Key bias_j) const {
  return gtsam::CombinedImuFactor(pose_i, vel_i, pose_j, vel_j, bias_i, bias_j, *pim_);
}

double ImuPreintegrator::deltaTimeSeconds() const { return pim_->deltaTij(); }

}  // namespace vio
