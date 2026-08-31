// =====================================================================================
// GtAligner.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   VIO'nun kestirdigi trajectory (keyframe pozlari) ile ground truth (GT)
//   trajectory'sini KARSILASTIRILABILIR hale getirmek: once zaman
//   damgalarina gore ESLESTIRMEK, sonra bir Umeyama HIZALAMASI uygulamak.
//
//   NEDEN HIZALAMA GEREKLI? VIO'nun kendi dunya cercevesi (ilk keyframe'in
//   pozuyla tanimli) ile GT'nin dunya cercevesi (Vicon/Leica referans
//   sistemi) FARKLI bir yerde/yonde baslar - ikisini DOGRUDAN karsilastirmak
//   anlamsizdir. Standart pratik (Zhang & Scaramuzza, "A Tutorial on
//   Quantitative Trajectory Evaluation"): VIO trajectory'sini, GT'ye EN IYI
//   UYAN rijit donusumle (rotasyon+oteleme, Umeyama'nin scale=1 hali) hizala,
//   SONRA farklarin RMSE'sini (ATE) hesapla.
//
//   NEDEN ZAMAN ARALIGI KISITLAMASI? MH_01_easy'de GT verisi kamera
//   akisindan ~1.075s SONRA basliyor (bkz. proje notlari). Bu araliktaki
//   kareler icin nearestGroundTruth() hep AYNI (ilk) GT ornegine sabitlenir -
//   bu da yanlis/yaniltici bir karsilastirmaya yol acar. Bu yuzden SADECE
//   GT'nin gercekten kapsadigi zaman araligindaki keyframe'ler karsilastirmaya
//   dahil edilir.
// =====================================================================================

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "DataLoader.hpp"

namespace vio {

// Zaman damgasina gore eslestirilmis (ama HENUZ hizalanmamis) VIO/GT pozisyon
// ciftleri.
struct MatchedTrajectories {
  std::vector<TimestampNs> timestamps;
  std::vector<Eigen::Vector3d> vio_positions;  // VIO'nun KENDI dunya cercevesinde
  std::vector<Eigen::Vector3d> gt_positions;   // GT dunya cercevesinde
};

// Umeyama hizalamasi sonucu: R,t (ve varsa scale) + hizalanmis VIO noktalari
// (dogrudan GT ile karsilastirilabilir, ayni cerceve).
struct AlignmentResult {
  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  double scale = 1.0;  // with_scale=false ile cagrildiysa hep 1.0
  std::vector<Eigen::Vector3d> aligned_vio_positions;
};

class GtAligner {
 public:
  // vio_timestamps/vio_positions: VIO'nun kestirdigi HER keyframe icin
  // (ayni sirada) zaman damgasi ve pozisyon. loader: GT verisinin kendisi
  // (gt zaman araligini ve nearestGroundTruth'u saglamak icin).
  //
  // SADECE loader.groundTruth() dizisinin ILK ve SON zaman damgasi arasinda
  // kalan VIO keyframe'leri sonuca dahil edilir (bkz. header'daki aciklama).
  static MatchedTrajectories match(const std::vector<TimestampNs>& vio_timestamps,
                                    const std::vector<Eigen::Vector3d>& vio_positions,
                                    const EurocDataLoader& loader);

  // Umeyama ile en-kucuk-kareler hizalamasi. with_scale=false: RIJIT (SE3,
  // scale=1 sabit) - ATE'nin STANDART tanimi budur. with_scale=true: SIM3
  // (scale de serbest cozulur) - SADECE tanisal amacli, "olcegimiz 1'e ne
  // kadar yakin" sorusuna cevap verir (mono+IMU sisteminin dogru metrik
  // olcek kestirip kestirmedigini gosterir).
  static AlignmentResult align(const MatchedTrajectories& matched, bool with_scale);
};

}  // namespace vio
