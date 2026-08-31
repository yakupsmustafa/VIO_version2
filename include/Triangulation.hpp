// =====================================================================================
// Triangulation.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   Iki farkli kameradan (iki keyframe'den) gorulen AYNI fiziksel noktanin
//   2D piksel gozlemlerinden, o noktanin 3D (dunya/world) konumunu hesaplamak
//   (triangulation / ucgenleme).
//
//   NEDEN GEREKLI? Mono kamera tek basina derinlik (uzaklik) bilgisi veremez -
//   tek bir goruntudeki bir piksel, o isinin (ray) uzerindeki HERHANGI bir
//   uzakliktaki bir noktaya ait olabilir. Ama AYNI noktayi FARKLI bir konumdan
//   (farkli bir keyframe'den) da gorursek, iki isinin kesistigi (veya en
//   yakin oldugu) nokta 3D konumu verir. Bu, VI-SfM baslatma (initialization)
//   ve sonrasinda yeni landmark eklemek icin temel islemdir.
//
//   KULLANILAN YONTEM: DLT (Direct Linear Transform) - cv::triangulatePoints
//   ile. Bu, iki goruntudeki noktalari ve kameralarin projeksiyon
//   matrislerini kullanarak dogrusal en kucuk kareler (linear least squares)
//   coz -umu ile 3D noktayi bulur. Literatürde standart ve sayisal olarak
//   kararli bir yontemdir (Hartley & Zisserman, "Multiple View Geometry").
//
//   KALITE KONTROLU: Her triangulate edilen nokta icin uc kontrol yapilir:
//     1) Pozitif derinlik: nokta HER IKI kameranin da ONUNDE olmali (arkasinda
//        degil) - aksi halde geometrik olarak imkansiz bir cozumdur.
//     2) Reprojeksiyon hatasi: 3D nokta tekrar goruntuye duşurulduğünde
//        (reproject) orijinal piksel konumuna yakin cikmali. Buyuk hata,
//        yanlis eslesme veya kotu poz tahmini anlamina gelir.
//     3) Parallax (paralaks) acisi: iki kameradan noktaya bakan isinlar
//        birbirine COK yakin acidaysa (neredeyse paralel), triangulation
//        sayisal olarak kararsizdir (kucuk bir gurultu devasa bir derinlik
//        hatasina yol acar) - bu yuzden minimum bir aci sarti koyuyoruz.
// =====================================================================================

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

#include "CameraModel.hpp"

namespace vio {

// Tek bir triangulation sonucunu ve kalite bilgilerini tasir.
struct TriangulatedPoint {
  Eigen::Vector3d position_world = Eigen::Vector3d::Zero();  // dunya (world) cercevesindeki 3D konum
  double reprojection_error_px_1 = 0.0;  // kare1'e geri izdusum hatasi (piksel, yaklasik)
  double reprojection_error_px_2 = 0.0;  // kare2'ye geri izdusum hatasi (piksel, yaklasik)
  double parallax_deg = 0.0;             // iki kameradan noktaya bakan isinlar arasindaki aci
  bool valid = false;                    // butun kalite kontrollerini gecti mi
};

struct TriangulationParams {
  double max_reprojection_error_px = 2.0;  // bu degerden buyuk hata -> gecersiz nokta
  double min_parallax_deg = 1.0;           // bu degerden kucuk parallax -> sayisal olarak guvenilmez
};

// -------------------------------------------------------------------------------------
// Triangulator
// ---------------------------------------------------------------------------
// ONEMLI KURAL (koordinat cercevesi yonu): Bu sinifta pozlar HER ZAMAN
// "T_w_c" formatinda verilir, yani "camera -> world" donusumu: kameranin
// KENDI cercevesindeki bir noktayi DUNYA cercevesine tasir (GTSAM'in Pose3
// kuralinin ayni, ileride FactorGraphBackend ile dogrudan uyumlu olacak).
// -------------------------------------------------------------------------------------
class Triangulator {
 public:
  Triangulator(const PinholeCameraModel& camera, const TriangulationParams& params = TriangulationParams());

  // Tek bir nokta cifti icin triangulation. pixel1/pixel2 HAM (distorsiyonlu)
  // piksel koordinatlaridir - distorsiyon giderme islemi bu fonksiyonun
  // icinde otomatik yapilir.
  TriangulatedPoint triangulate(const cv::Point2f& pixel1, const cv::Point2f& pixel2,
                                 const Eigen::Matrix4d& T_w_c1, const Eigen::Matrix4d& T_w_c2) const;

  // Coklu nokta cifti icin toplu (batch) triangulation - tek tek cagirmaktan
  // daha hizlidir (cv::triangulatePoints tek seferde tum noktalari cozer).
  std::vector<TriangulatedPoint> triangulateBatch(const std::vector<cv::Point2f>& pixels1,
                                                   const std::vector<cv::Point2f>& pixels2,
                                                   const Eigen::Matrix4d& T_w_c1,
                                                   const Eigen::Matrix4d& T_w_c2) const;

 private:
  const PinholeCameraModel& camera_;
  TriangulationParams params_;
};

}  // namespace vio
