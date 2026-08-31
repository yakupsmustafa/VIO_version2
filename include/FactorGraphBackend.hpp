// =====================================================================================
// FactorGraphBackend.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   VioInitializer'in ciktisiyla baslayip, her yeni keyframe geldikce factor
//   graph'i BUYUTEREK ve GTSAM'in iSAM2 (incremental smoothing and mapping)
//   cozucusuyle GERCEK ZAMANLI olarak pozlari/hizlari/bias'i optimize eden
//   ana "beyin" modulu.
//
//   DEGISKENLER (her keyframe i icin, symbol_shorthand ile):
//     X(i): gtsam::Pose3   - govde (IMU) pozu, dunya cercevesinde
//     V(i): gtsam::Vector3 - govde hizi, dunya cercevesinde
//     B(i): gtsam::imuBias::ConstantBias - o andaki IMU bias tahmini
//
//   FACTOR'LER:
//     - PriorFactor: SADECE ilk keyframe (X(0),V(0),B(0)) icin - butun grafin
//       "capasi" (anchor). Bu olmadan tum sistem keyfi bir dunya cercevesinde
//       "kayar" (gauge freedom).
//     - CombinedImuFactor: her ardisik keyframe cifti arasinda, IMU
//       preintegration'a dayali kisit (bkz. ImuPreintegration.hpp).
//     - SmartProjectionPoseFactor: her takip edilen ozellik (track) icin BIR
//       tane - o track'in TUM gozlemlerini (birden fazla keyframe'den) icinde
//       tutar. 3D nokta konumu ayri bir degisken DEGILDIR; factor kendi
//       icinde matematiksel olarak yok eder (Schur complement) - bu yuzden
//       triangulation/degenerate-nokta yonetimiyle elle ugrasmamiza gerek
//       kalmaz (GTSAM'in modern, Kimera-VIO gibi sistemlerin de kullandigi
//       yontemi - kullaniciyla birlikte kararlastirildi).
//
//   ISAM2 VE SMART FACTOR GUNCELLEMESI: Bir track daha once gorulmusse (yani
//   onun icin zaten bir SmartProjectionPoseFactor varsa), o factor'e YENI
//   gozlem eklenir (->add(...)) ve iSAM2'ye "eski versiyonunu sil, yeni
//   (guncellenmis) versiyonunu ekle" seklinde bildirilir - bu, GTSAM'in
//   smart factor'lerle incremental SLAM yapmanin standart, belgelenmis
//   yontemidir.
//
//   SLIDING WINDOW / MARGINALIZATION: Graf sinirsizca buyumesin (hem
//   hesaplama zamanla yavaslar hem de - deneyerek kesfedildi - cok
//   uzatilinca sayisal olarak kararsizlasip cokebiliyor) diye SADECE SON
//   N keyframe aktif tutulur. Ilk denemede ISAM2::marginalizeLeaves'i
//   DOGRUDAN cagirmak "Requested to eliminate a key that is not in the
//   factors" hatasina yol acti - cunku ISAM2'nin Bayes agacinin
//   marginalize edilecek degiskenleri "yaprak" (leaf) konumunda tutmasi
//   icin ONCEDEN bir eliminasyon SIRASI KISITLAMASI (ordering constraint)
//   verilmesi gerekiyor (GTSAM'in kendi IncrementalFixedLagSmoother
//   implementasyonu buna ihtiyac duyuyor - bkz. createOrderingConstraints).
//   Bunu elle dogru yapmaya calismak yerine, GTSAM'in ZATEN BU ISI YAPAN
//   hazir sinifini kullaniyoruz: gtsam_unstable::IncrementalFixedLagSmoother.
//   Her degiskene (X/V/B) bir "zaman damgasi" olarak KEYFRAME INDEKSINI
//   veriyoruz; smootherLag = sliding_window_size ayarlanarak, en yeni
//   keyframe'den sliding_window_size'dan daha eski olan TUM degiskenler
//   otomatik ve GUVENLI sekilde marginalize ediliyor.
// =====================================================================================

#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/SmartProjectionPoseFactor.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

#include "CameraModel.hpp"
#include "DataLoader.hpp"
#include "VioInitializer.hpp"

namespace vio {

struct BackendParams {
  // Baslangic (ilk keyframe) icin prior gurultu sigmalari.
  double prior_rot_sigma_rad = 0.05;    // rotasyon (roll/pitch/yaw bileseni basina)
  double prior_trans_sigma_m = 0.05;    // pozisyon (x/y/z bileseni basina)
  double prior_vel_sigma_mps = 0.10;    // hiz
  // bias (accel+gyro icin ortak basit izotropik sigma). NOT (deneyerek
  // kesfedildi): 0.05 gibi siki bir deger, VioInitializer'in accel bias'i
  // KASITLI OLARAK SIFIR birakmasiyla (bkz. VioInitializer.hpp) birlesince
  // optimizasyonu "yanlis (kucuk) scale + sifira yakin bias" gibi hatali
  // ama yerel-tutarli bir minimuma kilitleyip, sonunda
  // IndeterminantLinearSystemException ile cokmesine yol aciyordu. 0.3 gibi
  // daha gevsek bir deger, optimizasyona gercek bias'i bulmak icin daha
  // fazla ozgurluk taniyip crash'i ORTADAN KALDIRDI (scale sorununun
  // KENDISINI cozmedi, o ayri ve hala arastiriliyor - bkz. proje hafizasi).
  double prior_bias_sigma = 0.3;

  // SmartProjectionPoseFactor icin piksel-uzayinda gozlem gurultusu (izotropik).
  double reprojection_noise_sigma_px = 1.5;

  // iSAM2'nin kendi ayarlari - varsayilanlar cogu senaryoda iyi calisir.
  double isam2_relinearize_threshold = 0.1;
  int isam2_relinearize_skip = 1;

  // Sliding window (kayan pencere) boyutu: aktif tutulan keyframe sayisi.
  // Bu sayidan daha eski keyframe'ler IncrementalFixedLagSmoother tarafindan
  // otomatik olarak marginalize edilir (bkz. header'in basindaki aciklama).
  // Gercek-zamanli surekli calisma icin bu ZORUNLU bir mekanizma.
  //
  // DENENDI ama GERI ALINDI (2026-08-30, "gec-donem kayma/drift"
  // arastirmasi): hipotez, gec-donem kaymanin (bkz. proje hafizasi -
  // keyframe~620+'ta ATE/scale yeniden kotulesiyor) sliding window'un (12)
  // fast-motion doneminden gelen olcek bilgisini "unutmasi" oldugu yonundeydi.
  // 25.0 denendi: keyframe 26'da (TAM ilk marginalizasyon aninda) HEMEN
  // COKTU - ciddi regresyon. 15.0 denendi: cokme olmadi AMA gec-donem kayma
  // AYNI YERDE (keyframe~640-660) VE BENZER SIDDETTE olustu, final sonuc
  // hatta biraz daha KOTU cikti (ATE 2.64m vs 12.0 ile 2.21m). Pencere
  // boyutundan BAGIMSIZ olarak kaymanin AYNI ZAMANDA baslamasi, "pencere
  // unutuyor" hipotezini CURUTUYOR - sorun genel bir hafiza/unutma sorunu
  // degil, veri setinin o bolgesine OZGU bir sey (bkz. asagidaki GT hiz
  // profili incelemesi). 12.0 (dogrulanmis en iyi deger) GERI YUKLENDI.
  double sliding_window_size = 12.0;
};

// Tek bir keyframe'in optimize edilmis durumu (disariya acilan sonuc).
struct BackendState {
  gtsam::Pose3 pose;                       // R_w_b, p_w_b
  gtsam::Vector3 velocity = gtsam::Vector3::Zero();
  gtsam::imuBias::ConstantBias bias;
};

// -------------------------------------------------------------------------------------
// FactorGraphBackend
// -------------------------------------------------------------------------------------
class FactorGraphBackend {
 public:
  FactorGraphBackend(const PinholeCameraModel& camera, const ImuCalibration& imu_calib,
                     const BackendParams& params = BackendParams());

  // VioInitializer'in basariyla urettigi baslangic penceresini grafa ekler.
  // Bu, addKeyframe()'den ONCE, SADECE BIR KEZ cagrilmalidir.
  // imu_between_keyframes[k]: window[k] ile window[k+1] arasindaki IMU olcumleri
  // (VioInitializer::initialize()'a verilenle AYNI dizi).
  void initializeWithWindow(const std::vector<KeyframeObservation>& window,
                             const std::vector<std::vector<ImuMeasurement>>& imu_between_keyframes,
                             const VioInitializationResult& init_result);

  // Baslangic penceresinden SONRA her yeni keyframe icin cagrilir.
  // imu_since_prev: bir onceki keyframe'den bu yana biriken IMU olcumleri.
  void addKeyframe(const KeyframeObservation& obs, const std::vector<ImuMeasurement>& imu_since_prev);

  // En son eklenen keyframe'in GUNCEL (iSAM2 optimizasyonundan sonraki) durumu.
  BackendState latestState() const;

  // Belirli bir keyframe_id'nin kestirilen pozu (ATE/gorsellestirme icin).
  // Keyframe HALA aktif pencerede ise smoother'dan CANLI (guncel) deger doner;
  // sliding window ile MARGINALIZE EDILMIS bir keyframe ise, marginalize
  // edilmeden HEMEN ONCE DONDURULMUS (bir daha degismeyecek) son tahmini
  // doner (bkz. pose_history_).
  gtsam::Pose3 poseAt(int keyframe_id) const;

  int latestKeyframeId() const { return latest_keyframe_id_; }

  // --- TANI (DIAGNOSTIC): scale sorununun kok nedenini bulmak icin eklendi.
  // Su anda aktif olan SmartFactor'larin ne kadarinin GERCEKTEN triangulate
  // edilebildigini (valid) sayar - geri kalani (degenerate/behindCamera/
  // farPoint/outlier) o an icin VIZYONDAN HICBIR OLCEK BILGISI VERMIYOR
  // (ZERO_ON_DEGENERACY nedeniyle). Eger dusuk-paralakslı (yavas/durgun)
  // donemlerde valid orani cok dusukse, bu donemlerde sistemin FIILEN
  // SADECE IMU'ya dayandigini (ve bu yuzden scale'in coktugunu) dogrudan
  // kanitlar.
  struct SmartFactorHealth {
    int total = 0;
    int valid = 0;
    int degenerate = 0;
    int behind_camera = 0;
    int far_point = 0;
    int outlier = 0;
  };
  SmartFactorHealth smartFactorHealth() const;

 private:
  using SmartFactor = gtsam::SmartProjectionPoseFactor<gtsam::Cal3_S2>;

  // Bir keyframe'in track gozlemlerini smoother'a eklenecek yeni-factor/silinecek-
  // factor listelerine isler (yeni track ise yeni SmartFactor olusturur, eski
  // track ise mevcut SmartFactor'u guncelleyip "sil+yeniden ekle" siraya koyar).
  // new_factor_positions: (new_factors icindeki POZISYON, track_id) ciftleri -
  // new_factors'a prior/IMU factor'leri de karisik eklendigi icin sadece
  // track_id sirasi YETERLI DEGIL, tam pozisyon gerekiyor (bkz. commitFactorIndices).
  // touched_this_round: track_id -> new_factors icindeki POZISYON, SADECE bu
  // "round" (bir update() cagrisina kadar) icin gecerli. initializeWithWindow
  // gibi BIRDEN FAZLA keyframe'i TEK bir update() cagrisinda toplu isleyen
  // durumlarda, ayni track birden fazla keyframe'de gorulebilir - boyle bir
  // durumda new_factors'a YENI bir eleman eklemek yerine (ki bu GTSAM'in ic
  // indeksini bozar - deneyerek kesfedildi) AYNI pozisyondaki elemani
  // GUNCELLENMIS (tum bu rounddaki gozlemleri iceren) factor ile DEGISTIRIYORUZ
  // (new_factors[position] = factor;), boylece hicbir gozlem kaybolmaz.
  void stageObservations(int keyframe_id, const std::unordered_map<int, cv::Point2f>& observations,
                          gtsam::NonlinearFactorGraph& new_factors, gtsam::FactorIndices& remove_indices,
                          std::vector<std::pair<int, int>>& new_factor_positions,
                          std::unordered_map<int, int>& touched_this_round);

  // smoother_'un ISAM2Result::newFactorsIndices'i, new_factors'a EKLENEN
  // SIRAYLA birebir eslesir - new_factor_positions'daki pozisyonlari
  // kullanarak track_to_factor_index_'i gunceller.
  void commitFactorIndices(const gtsam::ISAM2Result& isam_result,
                            const std::vector<std::pair<int, int>>& new_factor_positions);

  const PinholeCameraModel& camera_;
  ImuCalibration imu_calib_;
  BackendParams params_;

  gtsam::Cal3_S2::shared_ptr K_;
  gtsam::SharedNoiseModel reprojection_noise_;
  gtsam::SmartProjectionParams smart_params_;
  // Kamera pozu, govde (X(i)) pozundan FARKLI (cam0'da hem oteleme hem de
  // BUYUK bir rotasyon farki var - T_BS neredeyse 90 derecelik eksen
  // degisimi iceriyor). SmartProjectionPoseFactor'e bu ofseti (body_P_sensor)
  // vermezsek, X(i)'yi SANKI kamera pozuymus gibi kullanir - bu da devasa
  // reprojeksiyon hatasina ve optimizasyonun cilginca sapmasina yol acar
  // (deneyerek kesfedildi).
  gtsam::Pose3 body_P_sensor_;

  gtsam::IncrementalFixedLagSmoother smoother_;

  // ONEMLI (deneyerek kesfedildi): bir SmartFactor'u smoother'a ekledikten
  // SONRA AYNI nesneyi mutasyona ugratip (->add ile yeni gozlem ekleyip)
  // SONRA "eski index'i sil + bu (mutasyona ugramis) nesneyi yeniden ekle"
  // yapmak CRASH'e yol acar: GTSAM'in VariableIndex::remove'u, factor'un
  // eklendigi andaki anahtar (key) kumesini bekler, ama mutasyon factor'un
  // anahtar kumesini DEGISTIRIR (yeni pose key eklenir), bu da "indices/factors
  // tutarsiz" hatasina (veya optimize derlemede sessiz segfault'a) yol acar.
  // COZUM: HER guncellemede bu track'in TUM GECMIS gozlemleriyle YEPYENI bir
  // SmartFactor nesnesi olustur - eski nesneye ASLA dokunma. track_observations_
  // bu yuzden gerekli (gecmisi saklamak icin).
  std::unordered_map<int, std::vector<std::pair<gtsam::Key, gtsam::Point2>>> track_observations_;
  std::unordered_map<int, SmartFactor::shared_ptr> smart_factors_;  // track_id -> EN GUNCEL factor nesnesi
  std::unordered_map<int, int> track_to_factor_index_;              // track_id -> ISAM2 icindeki guncel index

  int latest_keyframe_id_ = -1;
  gtsam::Pose3 latest_pose_;
  gtsam::Vector3 latest_velocity_ = gtsam::Vector3::Zero();
  gtsam::imuBias::ConstantBias latest_bias_;
  TimestampNs latest_timestamp_ns_ = 0;

  // Sliding window: su an aktif olan keyframe id'leri, ekleniş sirasiyla (en
  // eski onde). Smoother, "zaman damgasi" (keyframe indeksi) sliding_window_size'dan
  // daha eski olan degiskenleri OTOMATIK marginalize eder - bu deque, hangi
  // keyframe'in bir sonraki update() cagrisinda marginalize EDILECEGINI
  // ONCEDEN bilip pose_history_'yi zamaninda doldurabilmemiz icin tutulur.
  std::deque<int> active_keyframe_ids_;

  // Marginalize EDILMIS (artik smoother'da olmayan) TUM keyframe id'leri -
  // bir kere marginalize edilen bir id BIR DAHA ASLA geri gelmez, bu yuzden
  // kalici bir kayit. stageObservations, bir track'in gecmisini bu ID'lere
  // gore FILTRELER (bkz. .cpp'deki aciklama - aksi halde bir track'in
  // gecmisinde ARTIK VAR OLMAYAN bir keyframe'e referans kalabilir, bu da
  // factor yeniden kurulurken GTSAM'in "olmayan key" hatasina yol acar -
  // deneyerek kesfedildi).
  std::unordered_map<int, bool> marginalized_ids_;

  // Marginalize edilecek keyframe'lerin, marginalize edilmeden HEMEN ONCE
  // kaydedilen DONDURULMUS son pozu - poseAt() bunlari smoother'dan bir daha
  // sorgulayamayacagimiz icin (key artik sistemde yok) buradan doner.
  std::unordered_map<int, gtsam::Pose3> pose_history_;
};

}  // namespace vio
