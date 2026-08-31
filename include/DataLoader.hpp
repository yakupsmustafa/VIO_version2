// =====================================================================================
// BU HEADER'IN GÖREVİ:
//   EuRoC MAV formatındaki veriseti klasörünü diskten okuyup bellekte kullanılabilir hale getirmek.
//   Yani bu dosya SADECE veri OKUMA (I/O) işini yapar; hiçbir tahmin/optimizasyon/matematik içermez.
//
//   Okunan veriler:
//     - Kamera kareleri  (cam0/data.csv + data/*.png dosya yolları)
//     - IMU ölçümleri    (imu0/data.csv: jiroskop + ivmeölçer, 200 Hz)
//     - Ground truth     (state_groundtruth_estimate0/data.csv: gerçek pozisyon/
//                          hız/oryantasyon, Vicon/Leica ile ölçülmüş)
//     - Kalibrasyon      (cam0/sensor.yaml ve imu0/sensor.yaml: iç/dış parametreler)
//
//   Diğer modüller (CameraModel, FeatureExtraction, ImuPreintegration vb.) bu
//   sınıfın ürettiği veriyi girdi olarak kullanacak.


#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace vio {

// EuRoC zaman damgaları epoch'tan bu yana nanosaniye cinsindendir (~1.4e18 gibi
// büyük bir sayı). double tipi yalnızca ~15-16 anlamlı basamak taşıyabildiği için
// bu değeri doğrudan double'a çevirirsek sessizce ~100ns veya daha kötü bir
// yuvarlama hatası oluşur. Bu yüzden ham zaman damgasını int64 olarak saklıyoruz
// ve sadece küçük *farkları* (dt gibi) saniyeye çeviriyoruz.
using TimestampNs = int64_t;

// İki zaman damgası arasındaki (küçük) farkı saniyeye çevirir. Mutlak zaman
// damgasını değil, yalnızca dt gibi farkları çevirmek için kullanılmalı.
inline double nsToSeconds(TimestampNs ns) {
  return static_cast<double>(ns) * 1e-9;
}

// Tek bir IMU örneği: açısal hız (jiroskop) ve ivme (ivmeölçer).
struct ImuMeasurement {
  TimestampNs timestamp_ns;
  Eigen::Vector3d gyro;   // rad/s  (açısal hız)
  Eigen::Vector3d accel;  // m/s^2  (ivme, yerçekimi dahil - ham sensör verisi)
};

// Tek bir kamera karesi: zaman damgası + görüntü dosyasının diskteki tam yolu.
// Görüntü burada henüz yüklenmiyor (imread), sadece yolu tutuluyor; asıl
// yükleme main.cpp / ileride yazılacak görüntü işleme adımında yapılacak.
struct CameraFrame {
  TimestampNs timestamp_ns;
  std::string image_path;  // .png dosyasının tam (mutlak) yolu
};

// Ground truth (gerçek referans) poz bilgisi. Bu veri ASLA VIO tahmin
// algoritmasına girdi olarak verilmeyecek; sadece ATE hesaplama ve
// görselleştirmede karşılaştırma amaçlı kullanılacak.
struct GroundTruthPose {
  TimestampNs timestamp_ns;
  Eigen::Vector3d position;        // metre, dünya (world) çerçevesinde
  Eigen::Quaterniond orientation;  // gövde (body) -> dünya (world) rotasyonu
  Eigen::Vector3d velocity;        // m/s, dünya çerçevesinde
  Eigen::Vector3d gyro_bias;       // rad/s  (referans jiroskop bias'ı)
  Eigen::Vector3d accel_bias;      // m/s^2  (referans ivmeölçer bias'ı)
};

// Kameranın iç parametreleri (intrinsics) + lens bozulma (distortion) katsayıları
// + gövdeye (IMU'ya) göre dış parametreleri (extrinsics). cam0/sensor.yaml'dan
// okunur.
struct CameraCalibration {
  int width = 0;
  int height = 0;
  double fx = 0, fy = 0, cx = 0, cy = 0;  // odak uzaklığı (fx,fy) ve optik merkez (cx,cy), piksel cinsinden
  std::string distortion_model;      // EuRoC'ta: "radial-tangential"
  std::vector<double> distortion;    // [k1, k2, p1, p2, (k3)] - radyal + teğetsel bozulma katsayıları
  Eigen::Matrix4d T_BS = Eigen::Matrix4d::Identity();  // kamera -> gövde(IMU) dönüşüm matrisi (4x4, homojen)
  double rate_hz = 0;                // kameranın çalışma frekansı (EuRoC'ta 20 Hz)
};

// IMU'nun gürültü modeli parametreleri + gövdeye göre dış parametreleri.
// Bu parametreler GTSAM'ın IMU preintegration'ında gürültü kovaryansı olarak
// kullanılacak (ImuPreintegration.hpp'de).
struct ImuCalibration {
  double gyro_noise_density = 0;    // rad/s/sqrt(Hz)   - jiroskop "beyaz gürültü" yoğunluğu
  double gyro_random_walk = 0;      // rad/s^2/sqrt(Hz) - jiroskop bias'ının zamanla kayması (rastgele yürüyüş)
  double accel_noise_density = 0;   // m/s^2/sqrt(Hz)   - ivmeölçer "beyaz gürültü" yoğunluğu
  double accel_random_walk = 0;     // m/s^3/sqrt(Hz)   - ivmeölçer bias'ının zamanla kayması
  double rate_hz = 0;               // IMU'nun çalışma frekansı (EuRoC'ta 200 Hz)
  Eigen::Matrix4d T_BS = Eigen::Matrix4d::Identity();  // IMU zaten gövde çerçevesini tanımlar -> birim matris
};

// -------------------------------------------------------------------------------------
// EurocDataLoader
// ---------------------------------------------------------------------------
// Tek bir EuRoC verisetini (örn. "data/MH_01_easy") okuyup kamera/IMU/ground-truth
// akışlarını ve kalibrasyonu dışarıya açan sınıf. Salt okunur (read-only): bu
// sınıf hiçbir tahmin/kestirim yapmaz, sadece dosyadan okuduğunu bellekte tutar.
// -------------------------------------------------------------------------------------
class EurocDataLoader {
 public:
  // dataset_root, içinde "mav0" alt klasörü barındıran veriseti kök dizinidir
  // (EuRoC klasör yapısı). Örn: "data/MH_01_easy"
  explicit EurocDataLoader(const std::string& dataset_root);

  const std::vector<CameraFrame>& cameraFrames() const { return cam_frames_; }
  const std::vector<ImuMeasurement>& imuMeasurements() const { return imu_data_; }
  const std::vector<GroundTruthPose>& groundTruth() const { return gt_data_; }

  const CameraCalibration& cameraCalibration() const { return cam_calib_; }
  const ImuCalibration& imuCalibration() const { return imu_calib_; }

  // t0 < zaman damgası <= t1 aralığındaki IMU örneklerini zaman sırasıyla döndürür.
  // Bu fonksiyon, iki ardışık kamera karesi arasındaki IMU hareketini
  // "preintegrate" etmek (ön-entegrasyon) için kullanılacak.
  std::vector<ImuMeasurement> imuBetween(TimestampNs t0, TimestampNs t1) const;

  // Verilen zaman damgasına en yakın ground truth pozu bulur (en yakın komşu
  // arama - nearest neighbor). SADECE değerlendirme/görselleştirme için
  // kullanılmalı, kestirim algoritmasına asla girdi olarak verilmemeli.
  // Ground truth verisi boşsa false döner.
  bool nearestGroundTruth(TimestampNs t, GroundTruthPose& out) const;

 private:
  // Her biri tek bir dosyayı okuyup ilgili üye değişkeni doldurur.
  void loadCamCsv(const std::string& csv_path, const std::string& data_dir);
  void loadImuCsv(const std::string& csv_path);
  void loadGtCsv(const std::string& csv_path);
  void loadCamYaml(const std::string& yaml_path);
  void loadImuYaml(const std::string& yaml_path);

  std::vector<CameraFrame> cam_frames_;
  std::vector<ImuMeasurement> imu_data_;
  std::vector<GroundTruthPose> gt_data_;
  CameraCalibration cam_calib_;
  ImuCalibration imu_calib_;
};

}  // namespace vio
