// =====================================================================================
// VioInitializer.cpp
// ---------------------------------------------------------------------------
// VioInitializer.hpp icinde bildirilen sinifin implementasyonu. Turetilen tum
// formuller header'daki buyuk aciklama blogunda anlatilan mantigi izler.
// =====================================================================================

#include "VioInitializer.hpp"

#include <cmath>

#include <Eigen/Dense>
#include <gtsam/geometry/Rot3.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace vio {

VioInitializer::VioInitializer(const PinholeCameraModel& camera, const ImuCalibration& imu_calib,
                                const VioInitializerParams& params)
    : camera_(camera), imu_calib_(imu_calib), params_(params) {}

VioInitializer::PairwiseVision VioInitializer::estimateRelativePose(const KeyframeObservation& a,
                                                                     const KeyframeObservation& b) const {
  PairwiseVision result;

  std::vector<int> common_ids;
  for (const auto& kv : a.observations) {
    if (b.observations.count(kv.first)) common_ids.push_back(kv.first);
  }

  if (static_cast<int>(common_ids.size()) < params_.min_common_tracks_for_essential) {
    result.failure_reason =
        "yeterli ortak track yok (" + std::to_string(common_ids.size()) + "/" +
        std::to_string(params_.min_common_tracks_for_essential) + ")";
    return result;
  }

  std::vector<cv::Point2f> pts_a, pts_b;
  pts_a.reserve(common_ids.size());
  pts_b.reserve(common_ids.size());
  for (int id : common_ids) {
    pts_a.push_back(a.observations.at(id));
    pts_b.push_back(b.observations.at(id));
  }

  const std::vector<cv::Point2f> pts_a_u = camera_.undistortPixel(pts_a);
  const std::vector<cv::Point2f> pts_b_u = camera_.undistortPixel(pts_b);

  cv::Mat mask;
  cv::Mat E = cv::findEssentialMat(pts_a_u, pts_b_u, camera_.K(), cv::RANSAC, params_.essential_ransac_confidence,
                                    params_.essential_ransac_threshold_px, mask);
  if (E.empty()) {
    result.failure_reason = "Essential Matrix hesaplanamadi";
    return result;
  }

  cv::Mat R_cv, t_cv;
  const int inliers = cv::recoverPose(E, pts_a_u, pts_b_u, camera_.K(), R_cv, t_cv, mask);
  if (inliers < params_.min_essential_inliers) {
    result.failure_reason = "recoverPose sonrasi yeterli inlier yok (" + std::to_string(inliers) + ")";
    return result;
  }

  // recoverPose kurali: X_b = R*X_a + t (yani R,t: "a -> b" yonunde koordinat
  // donusumu, t BIRIM NORM - olceksiz).
  cv::cv2eigen(R_cv, result.R_step);
  cv::cv2eigen(t_cv, result.t_step);
  result.success = true;
  return result;
}

Eigen::Vector3d VioInitializer::solveGyroBias(
    const std::vector<Eigen::Matrix3d>& R_c0_ck, const std::vector<std::vector<ImuMeasurement>>& imu_between_keyframes,
    const std::vector<KeyframeObservation>& window) const {
  const int num_pairs = static_cast<int>(R_c0_ck.size()) - 1;
  const Eigen::Matrix3d R_bc = camera_.T_BS().block<3, 3>(0, 0);
  const Eigen::Matrix3d R_cb = R_bc.transpose();

  std::vector<Eigen::Matrix3d> J_blocks;
  std::vector<Eigen::Vector3d> residuals;
  J_blocks.reserve(num_pairs);
  residuals.reserve(num_pairs);

  for (int k = 0; k < num_pairs; ++k) {
    // Gorselden (Adim A, zincirlenmis): ardisik keyframe'ler arasindaki govde
    // rotasyonu. Rotasyon olcekten bagimsiz oldugu icin bu deger TAM dogrudur.
    const Eigen::Matrix3d R_c0_bk = R_c0_ck[k] * R_cb;
    const Eigen::Matrix3d R_c0_bk1 = R_c0_ck[k + 1] * R_cb;
    const Eigen::Matrix3d R_vision_relative = R_c0_bk.transpose() * R_c0_bk1;

    // IMU'dan: SIFIR bias ile preintegrate edilmis rotasyon + bunun gyro
    // bias'a gore Jacobian'i (GTSAM bunu hazir hesapliyor).
    ImuPreintegrator preint(imu_calib_);
    preint.reset(gtsam::imuBias::ConstantBias());
    preint.integrateInterval(imu_between_keyframes[k], window[k].timestamp_ns, window[k + 1].timestamp_ns);

    const gtsam::Rot3 delta_R_imu = preint.preintegrated().deltaRij();
    const gtsam::Rot3 R_vision_relative_gtsam(R_vision_relative);

    // residual = Log(deltaR_imu^-1 * R_vision) - bu, gyro bias'imiz TAM
    // dogru olsaydi sifir olurdu. Birinci-mertebe yaklasimla residual ~ J*delta_bg.
    const Eigen::Vector3d residual = gtsam::Rot3::Logmap(delta_R_imu.between(R_vision_relative_gtsam));
    // preintegrated_H_biasOmega: 9x3 Jacobian (ilk 3 satiri rotasyon/theta
    // kismina ait - TangentPreintegration'in ic gösterimi [theta,pos,vel] seklindedir).
    const Eigen::Matrix3d J = preint.preintegrated().preintegrated_H_biasOmega().block<3, 3>(0, 0);

    J_blocks.push_back(J);
    residuals.push_back(residual);
  }

  Eigen::MatrixXd A(3 * J_blocks.size(), 3);
  Eigen::VectorXd b(3 * J_blocks.size());
  for (std::size_t i = 0; i < J_blocks.size(); ++i) {
    A.block<3, 3>(3 * i, 0) = J_blocks[i];
    b.segment<3>(3 * i) = residuals[i];
  }

  return A.colPivHouseholderQr().solve(b);
}

VioInitializer::LinearAlignmentResult VioInitializer::linearAlignment(
    const std::vector<PairwiseVision>& pairwise, const std::vector<bool>& position_reliable,
    const std::vector<Eigen::Matrix3d>& R_c0_ck,
    const std::vector<std::vector<ImuMeasurement>>& imu_between_keyframes,
    const std::vector<KeyframeObservation>& window, const gtsam::imuBias::ConstantBias& bias,
    const Eigen::Vector3d* fixed_gravity) const {
  LinearAlignmentResult result;
  const bool solve_gravity = (fixed_gravity == nullptr);
  const int n = static_cast<int>(R_c0_ck.size());
  const int num_pairs = n - 1;

  const Eigen::Matrix3d R_bc = camera_.T_BS().block<3, 3>(0, 0);
  const Eigen::Vector3d t_bc = camera_.T_BS().block<3, 1>(0, 3);
  const Eigen::Matrix3d R_cb = R_bc.transpose();

  // r_k = R_c0_ck * (-R_bc^T * t_bc): kamera-govde (extrinsic) ofsetinin,
  // govde pozisyonu hesabina kattigi BILINEN (metrik) katki.
  std::vector<Eigen::Vector3d> r(n);
  std::vector<Eigen::Matrix3d> R_c0_bk(n);
  for (int k = 0; k < n; ++k) {
    R_c0_bk[k] = R_c0_ck[k] * R_cb;
    r[k] = R_c0_ck[k] * (-R_bc.transpose() * t_bc);
  }

  // u_k: ardisik cift (k,k+1) icin GORSELDEN gelen (OLCEKSIZ, birim-benzeri)
  // c0-cercevesinde goreli kamera oteleme yonu - SADECE position_reliable[k]
  // ise anlamli/hesaplanir (bkz. header - saf-donus gibi dejenere ciftlerde
  // gorsel oteleme guvenilmez, o ciftin pozisyon denklemi tamamen atlanir).
  std::vector<Eigen::Vector3d> u(num_pairs, Eigen::Vector3d::Zero());
  std::vector<int> scale_col_of_pair(num_pairs, -1);
  int num_reliable = 0;
  for (int k = 0; k < num_pairs; ++k) {
    if (!position_reliable[k]) continue;
    u[k] = R_c0_ck[k] * (-pairwise[k].R_step.transpose() * pairwise[k].t_step);
    scale_col_of_pair[k] = num_reliable++;
  }

  if (num_reliable == 0) {
    result.failure_reason = "hicbir ardisik ciftte guvenilir gorsel oteleme yok (hepsi dejenere/saf-donus gibi)";
    return result;
  }

  // Bilinmeyenler: [v_0,...,v_{n-1} (3 her biri), (solve_gravity ise g_c0 (3)),
  // s_0,...,s_{num_reliable-1} (1 her biri, SADECE guvenilir ciftler)]
  const int gravity_cols = solve_gravity ? 3 : 0;
  const int num_unknowns = 3 * n + gravity_cols + num_reliable;
  // Satirlar: HER cift icin 3 hiz denklemi (her zaman) + SADECE guvenilir ciftler icin 3 pozisyon denklemi.
  const int num_rows = 3 * num_pairs + 3 * num_reliable;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(num_rows, num_unknowns);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(num_rows);

  const int g_col = 3 * n;  // SADECE solve_gravity=true iken gecerli (3 sutun kaplar)
  const int s_col_base = 3 * n + gravity_cols;

  int row_cursor = 0;
  for (int k = 0; k < num_pairs; ++k) {
    ImuPreintegrator preint(imu_calib_);
    preint.reset(bias);
    preint.integrateInterval(imu_between_keyframes[k], window[k].timestamp_ns, window[k + 1].timestamp_ns);

    const double dt = preint.deltaTimeSeconds();
    const Eigen::Vector3d alpha = preint.preintegrated().deltaPij();
    const Eigen::Vector3d beta = preint.preintegrated().deltaVij();

    // Hiz denklemi (Eq B_k): v_{k+1} - v_k - g_c0*dt = R_c0_bk*beta.
    // SADECE IMU'ya dayandigi icin HER ciftte (guvenilir olsun olmasin) eklenir.
    // fixed_gravity verilmisse, g_c0*dt terimi BILINEN kabul edilip denklemin
    // sag tarafina (b) tasinir (bilinmeyenlerden cikarilir).
    const int row_v = row_cursor;
    row_cursor += 3;
    A.block<3, 3>(row_v, 3 * k) = -Eigen::Matrix3d::Identity();
    A.block<3, 3>(row_v, 3 * (k + 1)) = Eigen::Matrix3d::Identity();
    if (solve_gravity) {
      A.block<3, 3>(row_v, g_col) = -dt * Eigen::Matrix3d::Identity();
      b.segment<3>(row_v) = R_c0_bk[k] * beta;
    } else {
      b.segment<3>(row_v) = R_c0_bk[k] * beta + dt * (*fixed_gravity);
    }

    if (!position_reliable[k]) continue;

    // Pozisyon denklemi (header'daki turetim - Eq A_k'):
    //   s_k*u_k - v_k*dt - 0.5*g_c0*dt^2 = R_c0_bk*alpha_k - r_{k+1} + r_k
    const int row_p = row_cursor;
    row_cursor += 3;
    const int s_col = s_col_base + scale_col_of_pair[k];
    A.block<3, 3>(row_p, 3 * k) = -dt * Eigen::Matrix3d::Identity();
    A.block<3, 1>(row_p, s_col) = u[k];
    if (solve_gravity) {
      A.block<3, 3>(row_p, g_col) = -0.5 * dt * dt * Eigen::Matrix3d::Identity();
      b.segment<3>(row_p) = R_c0_bk[k] * alpha - r[k + 1] + r[k];
    } else {
      b.segment<3>(row_p) = R_c0_bk[k] * alpha - r[k + 1] + r[k] + 0.5 * dt * dt * (*fixed_gravity);
    }
  }

  const Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

  // En az bir s_k'nin makul (pozitif) cikip cikmadigini kontrol ediyoruz -
  // hepsi negatif/gecersizse hizalama anlamsizdir.
  bool any_valid_scale = false;
  for (int k = 0; k < num_pairs; ++k) {
    if (scale_col_of_pair[k] < 0) continue;
    const double sk = x(s_col_base + scale_col_of_pair[k]);
    if (std::isfinite(sk) && sk > 1e-4) any_valid_scale = true;
  }
  if (!any_valid_scale) {
    result.failure_reason = "hicbir ardisik cift icin gecerli (pozitif) olcek bulunamadi";
    return result;
  }

  result.success = true;
  result.gravity_c0 = solve_gravity ? x.segment<3>(g_col) : *fixed_gravity;
  result.velocities_c0.resize(n);
  for (int k = 0; k < n; ++k) {
    result.velocities_c0[k] = x.segment<3>(3 * k);
  }
  return result;
}

Eigen::Matrix3d VioInitializer::computeGravityAlignmentRotation(const Eigen::Vector3d& g_c0) const {
  const Eigen::Vector3d g_dir = g_c0.normalized();
  const Eigen::Vector3d target(0.0, 0.0, -1.0);  // dunya cercevesinde yercekimi yonu (Z-up -> asagi = -Z)

  // g_dir'i target'a tasiyan rotasyonu bul: R_w_c0 * g_dir = target.
  Eigen::Matrix3d R_w_c0 = Eigen::Quaterniond::FromTwoVectors(g_dir, target).toRotationMatrix();

  // Yaw (Z ekseni etrafindaki donus) yercekiminden gozlemlenemez - keyfi
  // olarak sifirliyoruz ki dunya cercevesinin X ekseni c0'in orijinal X
  // eksenine (izdusumsel olarak) yakin kalsin. Bu SADECE yorumlanabilirlik
  // icindir, matematiksel dogrulugu etkilemez (herhangi bir yaw secimi
  // esdegerdir).
  const Eigen::Vector3d x_axis_in_w = R_w_c0 * Eigen::Vector3d::UnitX();
  const double yaw = std::atan2(x_axis_in_w.y(), x_axis_in_w.x());
  const Eigen::Matrix3d R_yaw_correction = Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  return R_yaw_correction * R_w_c0;
}

VioInitializationResult VioInitializer::initialize(
    const std::vector<KeyframeObservation>& window,
    const std::vector<std::vector<ImuMeasurement>>& imu_between_keyframes) const {
  VioInitializationResult out;
  const int n = static_cast<int>(window.size());

  if (n < 5) {
    out.failure_reason = "pencere cok kucuk (en az 5 keyframe gerekli, verilen: " + std::to_string(n) + ")";
    return out;
  }
  if (static_cast<int>(imu_between_keyframes.size()) != n - 1) {
    out.failure_reason = "imu_between_keyframes boyutu window.size()-1 olmali";
    return out;
  }

  // --- ADIM A: ardisik goreli pozlar + rotasyon zinciri ---
  // Bazi ardisik ciftler DEJENERE olabilir (orn. saf-donus / pure-rotation:
  // oteleme neredeyse sifirsa Essential Matrix matematiksel olarak
  // cozulemez hale gelir - deneyerek kesfedildi, bkz. header). Boyle bir
  // ciftte, o cift icin ROTASYONU sifir-bias IMU preintegration'indan alip
  // (kisa surede yuksek dogruluklu, bkz. ImuPreintegration testi) zincire
  // devam ediyoruz, ama o ciftin GORSEL OTELEME bilgisi guvenilmez oldugu
  // icin position_reliable[k]=false isaretleyip ADIM C'de pozisyon
  // denklemini o ciftte atliyoruz (hiz denklemi yine kullanilir, cunku o
  // SADECE IMU'ya dayanir).
  const Eigen::Matrix3d R_bc_a = camera_.T_BS().block<3, 3>(0, 0);
  const Eigen::Matrix3d R_cb_a = R_bc_a.transpose();

  std::vector<PairwiseVision> pairwise(n - 1);
  std::vector<bool> position_reliable(n - 1, false);
  std::vector<Eigen::Matrix3d> R_c0_ck(n, Eigen::Matrix3d::Identity());
  for (int k = 0; k < n - 1; ++k) {
    pairwise[k] = estimateRelativePose(window[k], window[k + 1]);

    Eigen::Matrix3d R_body_rel;
    if (pairwise[k].success) {
      position_reliable[k] = true;
      // R_bk_bk+1 = R_bc * R_step^T * R_cb (bkz. header'daki turetim notu).
      R_body_rel = R_bc_a * pairwise[k].R_step.transpose() * R_cb_a;
    } else {
      position_reliable[k] = false;
      ImuPreintegrator preint(imu_calib_);
      preint.reset(gtsam::imuBias::ConstantBias());
      preint.integrateInterval(imu_between_keyframes[k], window[k].timestamp_ns, window[k + 1].timestamp_ns);
      R_body_rel = preint.preintegrated().deltaRij().matrix();
    }

    // R_c0_c(k+1) = R_c0_bk * R_bk_bk+1 * R_cb (govde zincirinden kameraya donus).
    const Eigen::Matrix3d R_c0_bk_current = R_c0_ck[k] * R_cb_a;
    const Eigen::Matrix3d R_c0_bk_next = R_c0_bk_current * R_body_rel;
    R_c0_ck[k + 1] = R_c0_bk_next * R_bc_a;
  }

  // --- ADIM B: gyro bias ---
  const Eigen::Vector3d delta_bg = solveGyroBias(R_c0_ck, imu_between_keyframes, window);
  const gtsam::imuBias::ConstantBias bias_after_gyro(Eigen::Vector3d::Zero(), delta_bg);

  // --- ADIM C: yercekimi + hiz (per-cift olcek ile, duzeltilmis bias ile yeniden entegre edilerek) ---
  const LinearAlignmentResult align =
      linearAlignment(pairwise, position_reliable, R_c0_ck, imu_between_keyframes, window, bias_after_gyro);
  if (!align.success) {
    out.failure_reason = "Dogrusal hizalama basarisiz: " + align.failure_reason;
    return out;
  }

  // --- Mutlak pozisyonlar: IMU preintegration denklemiyle ILERI DOGRU entegre et ---
  // (gorsel oteleme zincirlemeye HIC gerek yok - bkz. header'daki aciklama)
  const Eigen::Matrix3d R_bc = camera_.T_BS().block<3, 3>(0, 0);
  const Eigen::Matrix3d R_cb = R_bc.transpose();

  // ONEMLI DUZELTME (deneyerek kesfedildi): align.gravity_c0'in BUYUKLUGU,
  // dogrusal en kucuk kareler cozumunden geldigi icin gurultu/hata yuzunden
  // TAM 9.81 CIKMAZ (olcum: ~8.2 gibi, ~%15-20 hata). Ama GTSAM'in
  // CombinedImuFactor'u (ve FactorGraphBackend'in her yerde kullandigi IMU
  // preintegration parametreleri) HER ZAMAN TAM 9.81 varsayiyor
  // (makePreintegrationParams -> MakeSharedU(9.81)). Bu ikisi arasindaki
  // uyumsuzluk, VioInitializer'in kendi ileri-entegrasyonuyla FactorGraphBackend'in
  // beklentisini TUTARSIZ hale getirir - baslangic tahmini artik CombinedImuFactor'un
  // residual'ini sifirlamaz, optimizasyon ciddi sekilde sapar (olcek/pozisyon
  // hatasi olarak gozlemlendi). COZUM: yonu SOLVE EDILMIS yonde tutup, buyuklugu
  // FIZIKSEL OLARAK BILINEN 9.81 degerine sabitliyoruz (VINS-Mono'nun da
  // yaptigi "gravity magnitude" duzeltmesi).
  const Eigen::Vector3d g_c0 = align.gravity_c0.normalized() * 9.81;

  // Yercekimi buyuklugu duzeltildigi icin, ILK cozumdeki hizlar artik hafifce
  // tutarsiz (yanlis buyuklukteki yercekimiyle cozulmustu). Ayni sistemi
  // yercekimi SABIT (duzeltilmis) olarak tutup SADECE hizlar/olcekler icin
  // YENIDEN cozuyoruz - boylece sonuc hem dogru yercekimi buyuklugune hem de
  // onunla TUTARLI hizlara sahip olur (deneyerek kesfedildi, bkz. yukaridaki not).
  const LinearAlignmentResult refined =
      linearAlignment(pairwise, position_reliable, R_c0_ck, imu_between_keyframes, window, bias_after_gyro, &g_c0);
  if (!refined.success) {
    out.failure_reason = "Yercekimi-duzeltmeli yeniden hizalama basarisiz: " + refined.failure_reason;
    return out;
  }

  std::vector<Eigen::Vector3d> p_c0_bk(n, Eigen::Vector3d::Zero());  // p_c0_b0 = 0 (referans)
  for (int k = 0; k < n - 1; ++k) {
    ImuPreintegrator preint(imu_calib_);
    preint.reset(bias_after_gyro);
    preint.integrateInterval(imu_between_keyframes[k], window[k].timestamp_ns, window[k + 1].timestamp_ns);

    const double dt = preint.deltaTimeSeconds();
    const Eigen::Vector3d alpha = preint.preintegrated().deltaPij();
    const Eigen::Matrix3d R_c0_bk_k = R_c0_ck[k] * R_cb;

    p_c0_bk[k + 1] = p_c0_bk[k] + refined.velocities_c0[k] * dt + 0.5 * g_c0 * dt * dt + R_c0_bk_k * alpha;
  }

  // --- Yercekimi hizalamasi: c0 cercevesinden gravity-aligned dunya cercevesine gec ---
  const Eigen::Matrix3d R_w_c0 = computeGravityAlignmentRotation(g_c0);

  out.keyframes.resize(n);
  for (int k = 0; k < n; ++k) {
    const Eigen::Matrix3d R_c0_bk_k = R_c0_ck[k] * R_cb;

    InitializedKeyframe& kf = out.keyframes[k];
    kf.keyframe_id = window[k].keyframe_id;
    kf.timestamp_ns = window[k].timestamp_ns;
    kf.R_w_b = R_w_c0 * R_c0_bk_k;
    kf.p_w_b = R_w_c0 * p_c0_bk[k];
    kf.v_w = R_w_c0 * refined.velocities_c0[k];
  }

  out.success = true;
  out.initial_bias = bias_after_gyro;
  return out;
}

}  // namespace vio
