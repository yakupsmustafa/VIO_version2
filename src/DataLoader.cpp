// =====================================================================================
// DataLoader.cpp
// ---------------------------------------------------------------------------
// DataLoader.hpp içinde bildirilen EurocDataLoader sınıfının gerçek implementasyonu.
// Burada CSV dosyalarını satır satır okuyup parse ediyoruz ve YAML kalibrasyon
// dosyalarını yaml-cpp kütüphanesiyle okuyoruz.
// =====================================================================================

#include "DataLoader.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace vio {

namespace {

// Bir CSV satırını virgülle böler, her bir sütunu string olarak döndürür.
std::vector<std::string> splitCsvLine(const std::string& line) {
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, ',')) {
    tokens.push_back(token);
  }
  return tokens;
}

// EuRoC CSV dosyalarında '#' ile başlayan başlık (header) satırlarını ve boş
// satırları atlamak için kullanılır.
bool isCommentOrEmpty(const std::string& line) {
  return line.empty() || line[0] == '#' || line[0] == '\r';
}

// sensor.yaml içindeki "T_BS" alanını (4x4'lük, satır-öncelikli düzende
// düzleştirilmiş 16 sayılık liste) Eigen 4x4 matrisine çevirir.
Eigen::Matrix4d parseT_BS(const YAML::Node& node) {
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  const YAML::Node& data = node["data"];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      T(r, c) = data[r * 4 + c].as<double>();
    }
  }
  return T;
}

}  // namespace

// Kurucu: önce kalibrasyon (yaml) dosyalarını, sonra veri (csv) dosyalarını okur.
// Kalibrasyonun önce okunmasının bir zorunluluğu yok, sadece okuma sırasını
// mantıksal olarak "önce sensörü tanı, sonra verisini oku" şeklinde tutuyoruz.
EurocDataLoader::EurocDataLoader(const std::string& dataset_root) {
  const std::string mav0 = dataset_root + "/mav0";

  loadCamYaml(mav0 + "/cam0/sensor.yaml");
  loadCamCsv(mav0 + "/cam0/data.csv", mav0 + "/cam0/data");

  loadImuYaml(mav0 + "/imu0/sensor.yaml");
  loadImuCsv(mav0 + "/imu0/data.csv");

  loadGtCsv(mav0 + "/state_groundtruth_estimate0/data.csv");
}

// cam0/sensor.yaml dosyasını okuyup kamera kalibrasyonunu doldurur:
// dış parametre (T_BS), çözünürlük, iç parametreler (fx,fy,cx,cy) ve
// distortion (lens bozulma) katsayıları.
void EurocDataLoader::loadCamYaml(const std::string& yaml_path) {
  YAML::Node node = YAML::LoadFile(yaml_path);

  cam_calib_.T_BS = parseT_BS(node["T_BS"]);
  cam_calib_.rate_hz = node["rate_hz"].as<double>();

  const YAML::Node& res = node["resolution"];
  cam_calib_.width = res[0].as<int>();
  cam_calib_.height = res[1].as<int>();

  // intrinsics sırası EuRoC'ta [fu, fv, cu, cv] şeklindedir.
  const YAML::Node& intr = node["intrinsics"];
  cam_calib_.fx = intr[0].as<double>();
  cam_calib_.fy = intr[1].as<double>();
  cam_calib_.cx = intr[2].as<double>();
  cam_calib_.cy = intr[3].as<double>();

  cam_calib_.distortion_model = node["distortion_model"].as<std::string>();
  const YAML::Node& dist = node["distortion_coefficients"];
  for (std::size_t i = 0; i < dist.size(); ++i) {
    cam_calib_.distortion.push_back(dist[i].as<double>());
  }
}

// imu0/sensor.yaml dosyasını okuyup IMU gürültü modeli parametrelerini doldurur.
void EurocDataLoader::loadImuYaml(const std::string& yaml_path) {
  YAML::Node node = YAML::LoadFile(yaml_path);

  imu_calib_.T_BS = parseT_BS(node["T_BS"]);
  imu_calib_.rate_hz = node["rate_hz"].as<double>();
  imu_calib_.gyro_noise_density = node["gyroscope_noise_density"].as<double>();
  imu_calib_.gyro_random_walk = node["gyroscope_random_walk"].as<double>();
  imu_calib_.accel_noise_density = node["accelerometer_noise_density"].as<double>();
  imu_calib_.accel_random_walk = node["accelerometer_random_walk"].as<double>();
}

// cam0/data.csv dosyasını okur: her satır "zaman_damgası,dosya_adı.png" şeklindedir.
// Dosya adını, görüntünün gerçek klasördeki (data/) tam yoluyla birleştirip saklarız.
void EurocDataLoader::loadCamCsv(const std::string& csv_path, const std::string& data_dir) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("DataLoader: dosya acilamadi: " + csv_path);
  }

  std::string line;
  while (std::getline(file, line)) {
    if (isCommentOrEmpty(line)) continue;
    auto tokens = splitCsvLine(line);
    if (tokens.size() < 2) continue;

    CameraFrame frame;
    frame.timestamp_ns = std::stoll(tokens[0]);
    // Bazı editör/aktarımlarda satır sonunda '\r' kalabiliyor, temizliyoruz.
    std::string filename = tokens[1];
    if (!filename.empty() && filename.back() == '\r') filename.pop_back();
    frame.image_path = data_dir + "/" + filename;
    cam_frames_.push_back(std::move(frame));
  }

  // Emin olmak için zaman damgasına göre sıralıyoruz (EuRoC zaten sıralı gelir
  // ama garanti olsun diye).
  std::sort(cam_frames_.begin(), cam_frames_.end(),
            [](const CameraFrame& a, const CameraFrame& b) {
              return a.timestamp_ns < b.timestamp_ns;
            });
}

// imu0/data.csv dosyasını okur: her satır
// "zaman_damgası, wx,wy,wz (jiroskop), ax,ay,az (ivmeölçer)" şeklindedir.
void EurocDataLoader::loadImuCsv(const std::string& csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("DataLoader: dosya acilamadi: " + csv_path);
  }

  std::string line;
  while (std::getline(file, line)) {
    if (isCommentOrEmpty(line)) continue;
    auto tokens = splitCsvLine(line);
    if (tokens.size() < 7) continue;

    ImuMeasurement m;
    m.timestamp_ns = std::stoll(tokens[0]);
    m.gyro = Eigen::Vector3d(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
    m.accel = Eigen::Vector3d(std::stod(tokens[4]), std::stod(tokens[5]), std::stod(tokens[6]));
    imu_data_.push_back(m);
  }

  std::sort(imu_data_.begin(), imu_data_.end(),
            [](const ImuMeasurement& a, const ImuMeasurement& b) {
              return a.timestamp_ns < b.timestamp_ns;
            });
}

// state_groundtruth_estimate0/data.csv dosyasını okur: her satır
// "zaman_damgası, px,py,pz, qw,qx,qy,qz, vx,vy,vz, bwx,bwy,bwz, bax,bay,baz"
// şeklindedir (pozisyon, kuaterniyon, hız, jiroskop bias, ivmeölçer bias).
void EurocDataLoader::loadGtCsv(const std::string& csv_path) {
  std::ifstream file(csv_path);
  if (!file.is_open()) {
    throw std::runtime_error("DataLoader: dosya acilamadi: " + csv_path);
  }

  std::string line;
  while (std::getline(file, line)) {
    if (isCommentOrEmpty(line)) continue;
    auto tokens = splitCsvLine(line);
    if (tokens.size() < 17) continue;

    GroundTruthPose p;
    p.timestamp_ns = std::stoll(tokens[0]);
    p.position = Eigen::Vector3d(std::stod(tokens[1]), std::stod(tokens[2]), std::stod(tokens[3]));
    // EuRoC'ta kuaterniyon sırası: q_w, q_x, q_y, q_z (önce reel kısım).
    p.orientation = Eigen::Quaterniond(std::stod(tokens[4]), std::stod(tokens[5]),
                                        std::stod(tokens[6]), std::stod(tokens[7]));
    p.velocity = Eigen::Vector3d(std::stod(tokens[8]), std::stod(tokens[9]), std::stod(tokens[10]));
    p.gyro_bias = Eigen::Vector3d(std::stod(tokens[11]), std::stod(tokens[12]), std::stod(tokens[13]));
    p.accel_bias = Eigen::Vector3d(std::stod(tokens[14]), std::stod(tokens[15]), std::stod(tokens[16]));
    gt_data_.push_back(p);
  }

  std::sort(gt_data_.begin(), gt_data_.end(),
            [](const GroundTruthPose& a, const GroundTruthPose& b) {
              return a.timestamp_ns < b.timestamp_ns;
            });
}

// [t0, t1] aralığında kalan IMU örneklerini döndürür (t0 haric, t1 dahil).
// std::upper_bound ile ikili arama (binary search) yapıyoruz; imu_data_ zaten
// zaman damgasına göre sıralı olduğu için bu O(log n) çalışır.
std::vector<ImuMeasurement> EurocDataLoader::imuBetween(TimestampNs t0, TimestampNs t1) const {
  auto lo = std::upper_bound(imu_data_.begin(), imu_data_.end(), t0,
                              [](TimestampNs t, const ImuMeasurement& m) { return t < m.timestamp_ns; });
  auto hi = std::upper_bound(imu_data_.begin(), imu_data_.end(), t1,
                              [](TimestampNs t, const ImuMeasurement& m) { return t < m.timestamp_ns; });
  return std::vector<ImuMeasurement>(lo, hi);
}

// Verilen zaman damgasına en yakın ground truth kaydını bulur. std::lower_bound
// ile "t'den büyük veya eşit ilk eleman"ı buluyoruz, sonra bir önceki eleman
// ile karşılaştırıp hangisinin daha yakın olduğuna bakıyoruz.
bool EurocDataLoader::nearestGroundTruth(TimestampNs t, GroundTruthPose& out) const {
  if (gt_data_.empty()) return false;

  auto it = std::lower_bound(gt_data_.begin(), gt_data_.end(), t,
                              [](const GroundTruthPose& p, TimestampNs val) { return p.timestamp_ns < val; });

  if (it == gt_data_.begin()) {
    out = *it;
    return true;
  }
  if (it == gt_data_.end()) {
    // t, en son ground truth zamanindan bile sonra ise elimizdeki son kaydi dondur.
    out = *(it - 1);
    return true;
  }

  auto prev = it - 1;
  TimestampNs d_next = it->timestamp_ns - t;
  TimestampNs d_prev = t - prev->timestamp_ns;
  out = (d_next < d_prev) ? *it : *prev;
  return true;
}

}  // namespace vio
