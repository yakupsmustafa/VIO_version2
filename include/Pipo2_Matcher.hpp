// =====================================================================================
// Pipo2_Matcher.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GÖREVİ:
//   İki farklı karede (örn. ardışık iki kamera karesi) çıkarılmış ORB
//   tanımlayıcılarını (descriptor) birbirleriyle eşleştirmek ve YANLIŞ
//   eşleşmeleri (outlier) temizlemek.
//
//   Eşleştirme 3 aşamada yapılıyor (her aşama bir öncekinin ürettiği
//   eşleşmeleri daha da eler, doğruluk için katı davranıyoruz):
//
//   1) Oran testi (Lowe's ratio test): Her nokta için en yakın 2 aday bulunur;
//      en yakının mesafesi ikinciye çok yakınsa ("belirsiz eşleşme") atılır.
//      Bu, BRIEF/ORB gibi ikili (binary) tanımlayıcılarda da standart bir
//      pratiktir.
//   2) Çapraz kontrol (cross-check / simetrik eşleştirme): Eşleştirme hem
//      1->2 hem 2->1 yönünde yapılır, sadece HER İKİ yönde de birbirini
//      "en iyi eşleşme" olarak seçen çiftler tutulur. Tek yönlü eşleştirmeye
//      göre çok daha az ama çok daha güvenilir eşleşme verir.
//   3) Geometrik doğrulama (RANSAC + Essential Matrix): Kalan eşleşmelerin,
//      İKİ GÖRÜNTÜ ARASINDA TEK BİR KAMERA HAREKETİYLE (epipolar geometri)
//      tutarlı olup olmadığı kontrol edilir. Tutarsız (örn. hareketli bir
//      nesne üzerindeki veya yanlış eşleşmiş) noktalar RANSAC ile elenir.
//      Bu adımdan önce lens distorsiyonu mutlaka giderilmiş olmalı (bkz.
//      CameraModel.hpp), yoksa Essential Matrix hatalı çıkar.
// =====================================================================================

#pragma once

#include <vector>

#include <opencv2/features2d.hpp>

#include "CameraModel.hpp"

namespace vio {

struct MatcherParams {
  // DENENDI ama GERI ALINDI (2026-08-30, track omru arastirmasi): bu iki
  // esik hafifce gevsetilmisti (0.75->0.8, 1.5->2.0), amac frame-to-frame
  // eslesmenin daha fazla dogru eslesmeyi "hayatta tutmasi" ve track'lerin
  // daha uzun yasamasiydi. SONUC TAM TERSI oldu: tam veriseti testinde ATE
  // 5.64m -> 1052m'ye FIRLADI, crash daha erken oldu (keyframe 599). Neden:
  // gevsetilmis esikler daha fazla YANLIS eslesmenin (outlier) sizmasina
  // izin verdi - RANSAC (Essential Matrix) bunlarin hepsini temizleyemedi,
  // sizan yanlis eslesmeler SmartFactor'a bozuk geometri olarak girdi. Katı
  // orijinal degerlere GERI DONULDU.
  float ratio_test_threshold = 0.75f;  // Lowe oran testi esigi (kucuk = daha kati/az ama guvenilir eslesme)
  double ransac_threshold_px = 1.5;    // Essential Matrix RANSAC piksel esigi (undistort edilmis piksel biriminde)
  double ransac_confidence = 0.999;    // RANSAC'in dogru modeli bulma guven duzeyi
  int min_points_for_ransac = 8;       // bu sayidan az eslesme varsa RANSAC atlanir (guvenilir model kurulamaz)
};

// -------------------------------------------------------------------------------------
// OrbMatcher
// ---------------------------------------------------------------------------
// İki kare arasında ORB tanımlayıcılarını eşleştirip geometrik olarak
// doğrulanmış eşleşme listesini döndürür.
// -------------------------------------------------------------------------------------
class OrbMatcher {
 public:
  explicit OrbMatcher(const MatcherParams& params = MatcherParams());

  // kpts1/desc1: birinci karenin ozellik noktalari ve tanimlayicilari
  // kpts2/desc2: ikinci karenin ozellik noktalari ve tanimlayicilari
  // camera: distorsiyon giderme ve Essential Matrix icin K matrisi saglar
  //
  // Donen her cv::DMatch icin: queryIdx -> kpts1 icindeki indeks,
  //                             trainIdx -> kpts2 icindeki indeks.
  std::vector<cv::DMatch> match(const std::vector<cv::KeyPoint>& kpts1, const cv::Mat& desc1,
                                 const std::vector<cv::KeyPoint>& kpts2, const cv::Mat& desc2,
                                 const PinholeCameraModel& camera) const;

  // match()'in SADECE ilk 2 asamasi (oran testi + simetrik/karsilikli kontrol) -
  // 3. asama (RANSAC + Essential Matrix) UYGULANMAZ. Loop closure gibi iki
  // kare arasinda KEYFI BUYUK bir poz farki olabilecek (Essential Matrix'in
  // guvenilir calismayabilecegi) VE geometrik dogrulamanin BASKA bir yontemle
  // (orn. PnP-RANSAC, 3D-2D karsilik biliniyorsa) yapilacagi durumlar icin.
  // NOT: match() bu fonksiyonu KULLANMAZ, birbirinden BAGIMSIZDIR - bu yuzden
  // match()'in mevcut davranisi/cagiranlari HICBIR SEKILDE etkilenmez.
  std::vector<cv::DMatch> matchRaw(const std::vector<cv::KeyPoint>& kpts1, const cv::Mat& desc1,
                                    const std::vector<cv::KeyPoint>& kpts2, const cv::Mat& desc2) const;

 private:
  // query_desc'teki her tanimlayici icin train_desc icinde en yakin 2 komsuyu
  // bulur ve oran testini uygular. Donen eslesmelerde queryIdx=query_desc,
  // trainIdx=train_desc indeksidir.
  std::vector<cv::DMatch> ratioTestMatch(const cv::Mat& query_desc, const cv::Mat& train_desc) const;

  MatcherParams params_;
  cv::Ptr<cv::BFMatcher> matcher_;  // NORM_HAMMING: ORB'un ikili (binary) tanimlayicisi icin doğru mesafe olcusu
};

}  // namespace vio
