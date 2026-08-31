// =====================================================================================
// Pipo1_FeatureExtraction.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GÖREVİ:
//   Tek bir gri tonlamalı görüntüden ORB (Oriented FAST + Rotated BRIEF)
//   özellik noktalarını (keypoint) ve tanımlayıcılarını (descriptor) çıkarmak.
//
//   Sade bir "cv::ORB::create(...)->detectAndCompute(...)" çağrısı YETERLİ
//   DEĞİL çünkü OpenCV'nin varsayılan davranışı özellikleri görüntünün en
//   "köşeli" (yüksek tepkili) bölgelerinde kümeleyebiliyor (örn. bir binanın
//   köşesi). Bu, VIO için kötü bir durum: pose kestirimi (Essential Matrix,
//   triangulation) için özelliklerin görüntüye YAYGIN şekilde dağılmış olması
//   gerekir, yoksa geometri kötü şartlandırılır (ill-conditioned) ve doğruluk
//   düşer.
//
//   Bu yüzden burada ORB-SLAM ailesinin kullandığı yönteme benzer bir
//   "grid-tabanlı" (ızgara tabanlı) çıkarım uyguluyoruz:
//     1) Görüntüden bir piramit (pyramid) oluşturulur (farklı ölçeklerde
//        kopyalar) - böylece hem küçük hem büyük özellikler yakalanır
//        (ölçek-değişmezliği / scale invariance).
//     2) Her piramit seviyesi küçük hücrelere (grid cell) bölünür.
//     3) Her hücrede FAST köşe algılama çalıştırılır; hücre "boş" çıkarsa
//        (dokusuz bölge: gökyüzü, düz duvar gibi) eşik düşürülüp tekrar
//        denenir (adaptif eşik).
//     4) Her hücreden en iyi (en yüksek tepkili) birkaç nokta tutulur, böylece
//        toplam özellik sayısı hedeflenen değere yaklaşır ve görüntüye eşit
//        yayılmış olur.
//     5) Son olarak OpenCV'nin kendi ORB tanımlayıcı hesaplayıcısı
//        (orb_->compute) çağrılır; bu da her noktanın açısını (rotation)
//        hesaplayıp BRIEF tanımlayıcısını o açıya göre döndürerek çıkarır.
// =====================================================================================

#pragma once

#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>

namespace vio {

// ORB çıkarımını kontrol eden ayarlanabilir parametreler. Varsayılan değerler,
// kullanıcıyla birlikte kararlaştırılan değerlerdir (bkz. proje hafızası).
struct OrbExtractionParams {
  int target_features = 800;   // kare başına hedeflenen toplam özellik sayısı
  float scale_factor = 1.2f;   // piramit seviyeleri arası ölçek oranı (ORB-SLAM ile aynı)
  int n_levels = 8;             // piramit seviye sayısı (ölçek-değişmezliği için)
  int edge_threshold = 31;      // görüntü kenarından bu kadar piksel içeride özellik aranmaz
  int patch_size = 31;          // BRIEF tanımlayıcısının kullandığı yama (patch) boyutu
  int fast_threshold = 20;      // normal FAST köşe algılama eşiği
  int min_fast_threshold = 7;   // hücre boş çıkarsa denenecek daha düşük (daha toleranslı) eşik
  int grid_cell_size = 35;      // her piramit seviyesinde hücre boyutu (piksel, seviye-yerel)
};

// -------------------------------------------------------------------------------------
// OrbFeatureExtractor
// ---------------------------------------------------------------------------
// Grid-tabanlı, piramitli ORB özellik çıkarıcı. Durumsuz (stateless) çalışır:
// her extract() çağrısı bağımsızdır, önceki karelerle ilgili hiçbir bilgi
// tutmaz (kare-kareler arası takip işi FeatureTracker.hpp'nin görevi olacak).
// -------------------------------------------------------------------------------------
class OrbFeatureExtractor {
 public:
  explicit OrbFeatureExtractor(const OrbExtractionParams& params = OrbExtractionParams());

  // image_gray: tek kanallı (CV_8UC1) gri tonlamalı görüntü.
  // Çıktılar: keypoints (piksel konumu, ölçek/octave, açı, tepki değeri) ve
  // descriptors (Nx32 boyutlu, her satır bir noktanın 256-bit ORB tanımlayıcısı).
  void extract(const cv::Mat& image_gray,
               std::vector<cv::KeyPoint>& keypoints,
               cv::Mat& descriptors) const;

 private:
  // Görüntüden n_levels adet, her biri bir öncekinden scale_factor kat küçük
  // olan bir piramit oluşturur. pyramid[0] = orijinal görüntü.
  std::vector<cv::Mat> buildPyramid(const cv::Mat& image_gray) const;

  // Her piramit seviyesini hücrelere bölüp FAST ile köşe adaylarını toplar,
  // her hücreden en iyi adayları seçer ve tüm koordinatları seviye-0
  // (orijinal görüntü) koordinat sistemine çevirip döndürür.
  std::vector<cv::KeyPoint> detectGrid(const std::vector<cv::Mat>& pyramid) const;

  // ORB-SLAM2'de de kullanılan geometrik seri formülü: toplam özellik
  // sayısını piramit seviyelerine, her seviyenin "ağırlığına" göre (üst
  // seviyeler daha küçük olduğu için daha az özellik hedeflenir) paylaştırır.
  std::vector<int> computeFeaturesPerLevel() const;

  OrbExtractionParams params_;
  cv::Ptr<cv::ORB> orb_;                // sadece tanımlayıcı (descriptor) hesaplamak için kullanılır
  std::vector<float> scale_per_level_;  // scale_per_level_[L] = scale_factor^L
};

}  // namespace vio
