// =====================================================================================
// FactorGraphBackend.cpp
// ---------------------------------------------------------------------------
// FactorGraphBackend.hpp icinde bildirilen sinifin implementasyonu.
// =====================================================================================

#include "FactorGraphBackend.hpp"

#include <algorithm>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include "ImuPreintegration.hpp"

namespace vio {

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

namespace {

gtsam::Pose3 toGtsamPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
  return gtsam::Pose3(gtsam::Rot3(R), p);
}

// NOT: bu deger ImuPreintegration.cpp'deki makePreintegrationParams()'in
// MakeSharedU(9.81) cagrisiyla AYNI olmali (Z-up dunya cercevesi, g=(0,0,-9.81)).
// VioInitializer'in yercekimi-hizalama adimindan sonra TUM sistem bu
// varsayima gore calisir.
const Eigen::Vector3d kGravityWorld(0.0, 0.0, -9.81);

}  // namespace

FactorGraphBackend::FactorGraphBackend(const PinholeCameraModel& camera, const ImuCalibration& imu_calib,
                                        const BackendParams& params)
    : camera_(camera),
      imu_calib_(imu_calib),
      params_(params),
      smoother_(params.sliding_window_size, [&params] {
        gtsam::ISAM2Params isam_params;
        isam_params.relinearizeThreshold = params.isam2_relinearize_threshold;
        isam_params.relinearizeSkip = params.isam2_relinearize_skip;
        // IncrementalFixedLagSmoother'in kendi varsayilani (true) bosalan
        // factor "slot"larini YENIDEN KULLANIR - bu, bizim kendi
        // track_to_factor_index_ takibimizi (marginalization sonrasi eski
        // bir index'in BASKA bir factor'e ait hale gelmesiyle) gecersiz
        // kilip crash'e yol aciyordu (deneyerek kesfedildi). false birakarak
        // index'lerin KALICI (bir daha asla baska bir factor'e verilmeyecek)
        // olmasini sagliyoruz - hafif bir bellek israfi pahasina saglamlik.
        isam_params.findUnusedFactorSlots = false;
        return isam_params;
      }()) {
  const cv::Mat& K = camera_.K();
  K_ = boost::make_shared<gtsam::Cal3_S2>(K.at<double>(0, 0), K.at<double>(1, 1), /*skew=*/0.0, K.at<double>(0, 2),
                                           K.at<double>(1, 2));

  reprojection_noise_ = gtsam::noiseModel::Isotropic::Sigma(2, params_.reprojection_noise_sigma_px);

  body_P_sensor_ = gtsam::Pose3(gtsam::Rot3(camera_.T_BS().block<3, 3>(0, 0)), camera_.T_BS().block<3, 1>(0, 3));

  // Bir landmark'in triangulate edilen 3D konumu, optimizasyon sirasinda bir
  // kameranin ARKASINA duserse (cheirality ihlali), varsayilan davranis
  // yakalanmamis bir istisna (exception) firlatip programi cokertebilir.
  // ZERO_ON_DEGENERACY: boyle bir durumda o factor'un katkisini sifirlar
  // (crash yerine o karede o landmark yok sayilir). DynamicOutlierRejection,
  // reprojeksiyon hatasi asiri buyuk (gurultulu/yanlis) tracklari otomatik
  // reddederek boyle durumlarin olusma ihtimalini de azaltir.
  smart_params_.setDegeneracyMode(gtsam::DegeneracyMode::ZERO_ON_DEGENERACY);
  // DENENDI ama GERI ALINDI (2026-08-30, "gec-donem kayma" arastirmasi):
  // carpan 4.0->6.0 (6px->9px) denendi - hipotez, ani manevra sirasinda
  // GECERLI noktalarin bile yanlislikla "aykiri" sayilip elenmesini
  // onlemekti. SONUC DAHA KOTU oldu: bu sefer bozulma DAHA ERKEN (bilinen
  // ILK hizli-hareket segmenti, keyframe~180) baslad ve scale sadece 0.64'e
  // kadar toparlanabildi (final ATE 2.21m->3.60m). Bu, ARTIK UCUNCU kez
  // dogrulanan ayni genel deseni teyit ediyor: bu sistemde herhangi bir
  // kabul/red esigini GEVSETMEK (matcher esikleri, landmarkDistanceThreshold,
  // simdi de outlier esigi) SISTEMATIK OLARAK ZARAR VERIYOR - sistem zaten
  // az-kisitli (weakly-observable scale) bir rejimde calisiyor, "daha fazla
  // veriye izin ver" YERINE "sadece en guvenilir veriyi kullan" stratejisi
  // bu sistemde dogru yon. Orijinal deger (4.0 -> 6px) GERI YUKLENDI.
  smart_params_.setDynamicOutlierRejectionThreshold(params_.reprojection_noise_sigma_px * 4.0);

  // ONEMLI (teshis ile kesfedildi - bkz. scale sorunu arastirmasi): GTSAM'in
  // TriangulationParameters::rankTolerance VARSAYILANI (1.0) triangulation'in
  // NORMALIZE EDILMIS (kalibre edilmis, odak-uzakligi=1 olceginde) DLT
  // sistemi uzerinde calistigi goz onune alindiginda ASIRI KATI - bu olcekte
  // gercek/gecerli noktalarin DLT matrisinin en kucuk tekil degeri neredeyse
  // HER ZAMAN 1.0'in cok altinda kaliyor, bu da GECERLI noktalarin bile
  // SUREKLI "degenerate" (sozde-triangulate-edilemez) isaretlenmesine yol
  // aciyordu (olcum: tum calisma boyunca track'lerin ~%85-90'i - SADECE
  // yavas/durgun donemlerde degil, hizli hareket sirasinda bile - degenerate
  // cikiyordu, bu da "dusuk paralaks" degil "yanlis esik" oldugunu kanitliyor).
  // Sonuc: gorselden gelen olcek-duzeltici bilgi neredeyse tamamen atiliyordu,
  // sistem fiilen (cogunlukla) SADECE IMU'ya dayaniyor ve olcek coküyordu.
  // Cok daha kucuk bir deger (normalize koordinat olcegine uygun) kullaniyoruz.
  smart_params_.setRankTolerance(1e-9);

  // DENENDI ama GERI ALINDI: setEnableEPI(true) (DLT sonrasi nonlinear/LM
  // triangulation iyilestirmesi) tam veriseti testinde yakalanmamis bir
  // gtsam::CheiralityException ile CÖKTÜ (ZERO_ON_DEGENERACY SADECE ilk DLT
  // asamasindaki TriangulationCheiralityException'i yakaliyor - EPI'nin LM
  // iterasyonlari sirasinda olusan bu farkli/yakalanmamis istisna GTSAM'in bu
  // surumunde korumasiz birakiyor). enableEPI KAPALI birakiliyor.

  // DENENDI ama GERI ALINDI: setLandmarkDistanceThreshold(50.0). Beklenenin
  // TAM TERSI oldu - tam veriseti testinde keyframe~180'den itibaren gecerli
  // track sayisi SIFIRA dustu ve ATE binlerce metreye patlayip cokme
  // yasandi. Neden: sistemin ölçegi zaten yanlisken (scale << 1), kamera
  // pozlarinin KENDISI yanlis olcekli oldugu icin triangulate edilen
  // noktalarin (yanlis olcekli) world-frame mesafesi metrik esigi (50m)
  // sik sik asiyor - bu da GERCEKTE YAKIN olan noktalari "cok uzak" diye
  // eleyip TUM gorsel kisiti yok ediyor (SADECE IMU kalan sistem, daha once
  // gozlemlenen felaketle ayni sekilde cokuyor). Sabit metrik bir mesafe
  // esigi, henuz DOGRU olcege oturmamis bir sistemde GUVENILMEZ - sinirsiz
  // (-1, varsayilan) birakiliyor.
}

void FactorGraphBackend::stageObservations(int keyframe_id, const std::unordered_map<int, cv::Point2f>& observations,
                                            gtsam::NonlinearFactorGraph& new_factors,
                                            gtsam::FactorIndices& remove_indices,
                                            std::vector<std::pair<int, int>>& new_factor_positions,
                                            std::unordered_map<int, int>& touched_this_round) {
  for (const auto& kv : observations) {
    const int track_id = kv.first;
    const gtsam::Point2 measurement(kv.second.x, kv.second.y);

    const bool is_new_track = (smart_factors_.find(track_id) == smart_factors_.end());

    // ONEMLI (deneyerek kesfedildi): eger bu track'in gecmisinde ARTIK
    // SMOOTHER'DA VAR OLMAYAN (marginalize edilmis) bir keyframe'e referans
    // VARSA, o factor'u YENIDEN KURMUYORUZ. Ilk denemede gecmisi bu tur
    // referanslardan "arindirip" factor'u yeniden kurmayi denedik, ama bu,
    // GTSAM'in o keyframe marginalize edilirken zaten DOGRU sekilde grafiga
    // katladigi bilgiyi ATIP factor'u daha ZAYIF (daha az gozlemli) hale
    // getiriyordu - bu da sistemin sayisal kararliligini KOTULESTIRDI
    // (IndeterminantLinearSystemException daha da erken olusmaya basladi).
    // Bunun yerine: boyle bir track'i sadece DONDURUYORUZ (ne yeni gozlem
    // ekliyoruz ne de eski factor'u degistiriyoruz) - marginalize edilirken
    // katlanmis bilgi OLDUGU GIBI korunur, sadece bu track'in ARTIK DAHA
    // FAZLA iyilesmemesi (yeni gozlemle guncellenmemesi) gibi kucuk bir
    // bedeli var (pratikte nadir - bkz. asagidaki not).
    if (!is_new_track && !marginalized_ids_.empty()) {
      const std::vector<std::pair<gtsam::Key, gtsam::Point2>>& existing_history = track_observations_[track_id];
      const bool touches_marginalized =
          std::any_of(existing_history.begin(), existing_history.end(),
                      [this](const std::pair<gtsam::Key, gtsam::Point2>& o) {
                        const int obs_keyframe_id = static_cast<int>(gtsam::Symbol(o.first).index());
                        return marginalized_ids_.count(obs_keyframe_id) > 0;
                      });
      if (touches_marginalized) continue;
    }

    // Bu track'in gecmis gozlem listesine yeni gozlemi ekle, sonra TUM
    // gecmisle (mutasyon DEGIL) YEPYENI bir SmartFactor nesnesi olustur -
    // bkz. header'daki aciklama (mutasyon+remove kombinasyonu crash'e yol aciyor).
    std::vector<std::pair<gtsam::Key, gtsam::Point2>>& history = track_observations_[track_id];
    history.emplace_back(X(keyframe_id), measurement);

    SmartFactor::shared_ptr factor =
        boost::make_shared<SmartFactor>(reprojection_noise_, K_, body_P_sensor_, smart_params_);
    for (const auto& obs : history) {
      factor->add(obs.second, obs.first);
    }
    smart_factors_[track_id] = factor;

    auto round_it = touched_this_round.find(track_id);
    if (round_it != touched_this_round.end()) {
      // Bu track bu "round" icinde (bir onceki keyframe'de) zaten new_factors'a
      // eklendi - new_factors'a YENI bir eleman EKLEMEK yerine (GTSAM'in ic
      // indeksini bozar - deneyerek kesfedildi) AYNI pozisyondaki elemani bu
      // GUNCELLENMIS (tum bu rounddaki gozlemleri iceren) factor ile degistiriyoruz.
      new_factors[round_it->second] = factor;
      continue;
    }

    if (!is_new_track) {
      // Bilinen (onceki bir round'da commit edilmis) track: ISAM2'ye "eski
      // versiyonunu sil, bu (guncellenmis) versiyonu yeniden ekle" bildir -
      // GTSAM'in smart factor'leri incremental guncellemenin standart yontemi.
      auto idx_it = track_to_factor_index_.find(track_id);
      if (idx_it != track_to_factor_index_.end()) {
        remove_indices.push_back(idx_it->second);
      }
    }

    const int position = static_cast<int>(new_factors.size());
    new_factors.add(factor);
    new_factor_positions.emplace_back(position, track_id);
    touched_this_round[track_id] = position;
  }
}

void FactorGraphBackend::commitFactorIndices(const gtsam::ISAM2Result& isam_result,
                                              const std::vector<std::pair<int, int>>& new_factor_positions) {
  const gtsam::FactorIndices& new_indices = isam_result.newFactorsIndices;
  for (const auto& pr : new_factor_positions) {
    const int position = pr.first;
    const int track_id = pr.second;
    track_to_factor_index_[track_id] = static_cast<int>(new_indices[position]);
  }
}

void FactorGraphBackend::initializeWithWindow(
    const std::vector<KeyframeObservation>& window,
    const std::vector<std::vector<ImuMeasurement>>& imu_between_keyframes,
    const VioInitializationResult& init_result) {
  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;
  gtsam::FactorIndices remove_indices;  // ilk kurulumda silinecek bir sey yok, bos kalir
  gtsam::FixedLagSmoother::KeyTimestampMap timestamps;
  std::vector<std::pair<int, int>> new_factor_positions;
  std::unordered_map<int, int> touched_this_round;  // bkz. stageObservations aciklamasi

  const int n = static_cast<int>(window.size());

  for (int k = 0; k < n; ++k) {
    const InitializedKeyframe& kf = init_result.keyframes[k];
    const gtsam::Pose3 pose_k = toGtsamPose(kf.R_w_b, kf.p_w_b);
    const gtsam::Vector3 vel_k(kf.v_w);

    new_values.insert(X(k), pose_k);
    new_values.insert(V(k), vel_k);
    new_values.insert(B(k), init_result.initial_bias);

    // Sliding window'un "zaman damgasi" olarak keyframe indeksini kullaniyoruz
    // (bkz. header'daki aciklama - IncrementalFixedLagSmoother bunu kullanarak
    // en eski keyframe'leri otomatik marginalize eder).
    timestamps[X(k)] = static_cast<double>(k);
    timestamps[V(k)] = static_cast<double>(k);
    timestamps[B(k)] = static_cast<double>(k);

    if (k == 0) {
      // Butun grafin "capasi" (anchor): bu olmadan tum sistem keyfi bir
      // dunya cercevesinde kayar (gauge freedom).
      auto prior_pose_noise = gtsam::noiseModel::Diagonal::Sigmas(
          (gtsam::Vector(6) << params_.prior_rot_sigma_rad, params_.prior_rot_sigma_rad, params_.prior_rot_sigma_rad,
           params_.prior_trans_sigma_m, params_.prior_trans_sigma_m, params_.prior_trans_sigma_m)
              .finished());
      auto prior_vel_noise = gtsam::noiseModel::Isotropic::Sigma(3, params_.prior_vel_sigma_mps);
      auto prior_bias_noise = gtsam::noiseModel::Isotropic::Sigma(6, params_.prior_bias_sigma);

      new_factors.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), pose_k, prior_pose_noise));
      new_factors.add(gtsam::PriorFactor<gtsam::Vector3>(V(0), vel_k, prior_vel_noise));
      new_factors.add(
          gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), init_result.initial_bias, prior_bias_noise));
    } else {
      ImuPreintegrator preint(imu_calib_);
      preint.reset(init_result.initial_bias);
      preint.integrateInterval(imu_between_keyframes[k - 1], window[k - 1].timestamp_ns, window[k].timestamp_ns);
      new_factors.add(preint.makeFactor(X(k - 1), V(k - 1), B(k - 1), X(k), V(k), B(k)));
    }

    stageObservations(window[k].keyframe_id, window[k].observations, new_factors, remove_indices,
                       new_factor_positions, touched_this_round);
  }

  smoother_.update(new_factors, new_values, timestamps, remove_indices);
  commitFactorIndices(smoother_.getISAM2Result(), new_factor_positions);
  // Yeni degiskenlerin daha iyi relinearize edilmesi icin bos guncellemeler -
  // GTSAM'in kendi ornek kodlarinda da kullanilan yaygin bir pratik.
  smoother_.update();
  smoother_.update();

  latest_keyframe_id_ = window.back().keyframe_id;
  latest_timestamp_ns_ = window.back().timestamp_ns;
  latest_pose_ = smoother_.calculateEstimate<gtsam::Pose3>(X(n - 1));
  latest_velocity_ = smoother_.calculateEstimate<gtsam::Vector3>(V(n - 1));
  latest_bias_ = smoother_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(n - 1));

  // NOT: sliding_window_size, baslangic penceresi boyutundan (n) BUYUK
  // olmalidir (varsayilan: 12 > 10) - aksi halde bu ilk toplu ekleme
  // sirasinda bazi keyframe'ler hemen marginalize olabilir ve onlarin
  // pose_history_'sini kaydetme sansimiz olmaz. Bu, VioInitializer'in
  // urettigi pencere boyutuyla dogal olarak uyumludur.
  for (int k = 0; k < n; ++k) {
    active_keyframe_ids_.push_back(window[k].keyframe_id);
  }
}

void FactorGraphBackend::addKeyframe(const KeyframeObservation& obs, const std::vector<ImuMeasurement>& imu_since_prev) {
  const int new_id = obs.keyframe_id;
  const int prev_id = latest_keyframe_id_;

  ImuPreintegrator preint(imu_calib_);
  preint.reset(latest_bias_);
  preint.integrateInterval(imu_since_prev, latest_timestamp_ns_, obs.timestamp_ns);

  // IMU'dan ILERI-YURUTULEREK tahmin edilen baslangic degeri (smoother'a
  // "initial guess" olarak verilecek - optimizasyon buradan baslayip
  // duzeltecek).
  const double dt = preint.deltaTimeSeconds();
  const Eigen::Vector3d alpha = preint.preintegrated().deltaPij();
  const Eigen::Vector3d beta = preint.preintegrated().deltaVij();
  const gtsam::Rot3 delta_R = preint.preintegrated().deltaRij();

  const gtsam::Rot3 R_new = latest_pose_.rotation() * delta_R;
  const Eigen::Vector3d p_new = latest_pose_.translation() + latest_velocity_ * dt +
                                 0.5 * kGravityWorld * dt * dt + latest_pose_.rotation().matrix() * alpha;
  const Eigen::Vector3d v_new = latest_velocity_ + kGravityWorld * dt + latest_pose_.rotation().matrix() * beta;

  // --- Sliding window: HANGI keyframe'in bu round'da marginalize edilecegini
  // GTSAM'in ic zamanlama mantigini birebir taklit ederek tahmin etmeye
  // calismak kirilgan cikti (deneyerek kesfedildi - bir-birimlik fark
  // zamanla birikip crash'e yol acti). Bunun yerine daha SAGLAM bir yontem:
  // update() cagirmadan ONCE aktif TUM keyframe'lerin pozunu tazeleyip
  // kaydediyoruz (pose_history_), update() SONRASI ise hangilerinin
  // smoother'dan GERCEKTEN dustugunu dogrudan sorup (exists()) takip
  // listesinden cikariyoruz - boylece GTSAM'in tam ne zaman marginalize
  // ettigini bilmemize gerek kalmiyor.
  for (int id : active_keyframe_ids_) {
    pose_history_[id] = smoother_.calculateEstimate<gtsam::Pose3>(X(id));
  }

  gtsam::NonlinearFactorGraph new_factors;
  gtsam::Values new_values;
  gtsam::FactorIndices remove_indices;
  gtsam::FixedLagSmoother::KeyTimestampMap timestamps;
  std::vector<std::pair<int, int>> new_factor_positions;
  std::unordered_map<int, int> touched_this_round;

  new_values.insert(X(new_id), gtsam::Pose3(R_new, p_new));
  new_values.insert(V(new_id), gtsam::Vector3(v_new));
  new_values.insert(B(new_id), latest_bias_);

  timestamps[X(new_id)] = static_cast<double>(new_id);
  timestamps[V(new_id)] = static_cast<double>(new_id);
  timestamps[B(new_id)] = static_cast<double>(new_id);

  new_factors.add(preint.makeFactor(X(prev_id), V(prev_id), B(prev_id), X(new_id), V(new_id), B(new_id)));

  stageObservations(new_id, obs.observations, new_factors, remove_indices, new_factor_positions, touched_this_round);

  smoother_.update(new_factors, new_values, timestamps, remove_indices);
  commitFactorIndices(smoother_.getISAM2Result(), new_factor_positions);

  latest_keyframe_id_ = new_id;
  latest_timestamp_ns_ = obs.timestamp_ns;
  latest_pose_ = smoother_.calculateEstimate<gtsam::Pose3>(X(new_id));
  latest_velocity_ = smoother_.calculateEstimate<gtsam::Vector3>(V(new_id));
  latest_bias_ = smoother_.calculateEstimate<gtsam::imuBias::ConstantBias>(B(new_id));

  active_keyframe_ids_.push_back(new_id);

  // Bu round'da smoother tarafindan GERCEKTEN marginalize edilmis (artik
  // linearization point'te olmayan) keyframe'leri takip listesinden cikar.
  // pose_history_'deki (bir onceki adimda tazelenen) degerleri artik
  // DONDURULMUS son tahmin olarak kalir. NOT: sadece deque'nin ONUNU
  // kontrol etmek YETERLI DEGIL - marginalization her zaman kati bir FIFO
  // sirasinda olmayabilir (deneyerek kesfedildi) - bu yuzden TUM listeyi
  // taniyoruz.
  const gtsam::Values& linearization_point = smoother_.getLinearizationPoint();
  for (auto it = active_keyframe_ids_.begin(); it != active_keyframe_ids_.end();) {
    if (!linearization_point.exists(X(*it))) {
      marginalized_ids_[*it] = true;  // bkz. stageObservations - gecmis arindirma icin gerekli
      it = active_keyframe_ids_.erase(it);
    } else {
      ++it;
    }
  }
}

BackendState FactorGraphBackend::latestState() const {
  BackendState state;
  state.pose = latest_pose_;
  state.velocity = latest_velocity_;
  state.bias = latest_bias_;
  return state;
}

FactorGraphBackend::SmartFactorHealth FactorGraphBackend::smartFactorHealth() const {
  SmartFactorHealth health;
  const gtsam::Values estimate = smoother_.calculateEstimate();
  for (const auto& kv : smart_factors_) {
    const SmartFactor::shared_ptr& factor = kv.second;
    // Bu factor, marginalize edilip smoother'dan dusmus (artik estimate icinde
    // olmayan) bir keyframe'e hala referans veriyor olabilir ("donmus" track -
    // bkz. stageObservations). point() TUM keylerin estimate'te olmasini
    // gerektirir, aksi halde ValuesKeyDoesNotExist firlatir - bu yuzden once
    // kontrol ediyoruz.
    bool all_keys_present = true;
    for (const gtsam::Key& key : factor->keys()) {
      if (!estimate.exists(key)) {
        all_keys_present = false;
        break;
      }
    }
    if (!all_keys_present) continue;

    ++health.total;
    const gtsam::TriangulationResult result = factor->point(estimate);
    if (result.valid()) {
      ++health.valid;
    } else if (result.degenerate()) {
      ++health.degenerate;
    } else if (result.behindCamera()) {
      ++health.behind_camera;
    } else if (result.farPoint()) {
      ++health.far_point;
    } else if (result.outlier()) {
      ++health.outlier;
    }
  }
  return health;
}

std::vector<FactorGraphBackend::LandmarkSnapshot> FactorGraphBackend::snapshotValidLandmarks(
    int keyframe_id, const std::unordered_map<int, cv::Point2f>& observations) const {
  (void)keyframe_id;  // su an sadece log/gelecekteki kullanim icin saklaniyor
  std::vector<LandmarkSnapshot> result;
  const gtsam::Values estimate = smoother_.calculateEstimate();

  for (const auto& kv : observations) {
    const int track_id = kv.first;
    const auto factor_it = smart_factors_.find(track_id);
    if (factor_it == smart_factors_.end()) continue;

    const SmartFactor::shared_ptr& factor = factor_it->second;
    bool all_keys_present = true;
    for (const gtsam::Key& key : factor->keys()) {
      if (!estimate.exists(key)) {
        all_keys_present = false;
        break;
      }
    }
    if (!all_keys_present) continue;

    const gtsam::TriangulationResult tri = factor->point(estimate);
    if (!tri.valid()) continue;

    LandmarkSnapshot snap;
    snap.track_id = track_id;
    snap.pixel = kv.second;
    snap.point_w = *tri;
    result.push_back(snap);
  }
  return result;
}

gtsam::Pose3 FactorGraphBackend::poseAt(int keyframe_id) const {
  auto it = pose_history_.find(keyframe_id);
  if (it != pose_history_.end()) {
    // Bu keyframe marginalize edilmis (smoother icinde artik yok) - marginalize
    // edilmeden HEMEN ONCE kaydedilen dondurulmus son tahmini donuyoruz.
    return it->second;
  }
  return smoother_.calculateEstimate<gtsam::Pose3>(X(keyframe_id));
}

}  // namespace vio
