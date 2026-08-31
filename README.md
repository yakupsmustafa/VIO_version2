# Drone VIO — Versiyon 2

Bu proje, bir drone için geliştirilen Visual-Inertial Odometry (VIO) sisteminin
**2. versiyonudur**. Unreal Engine tabanlı bir simülasyon ortamında denenmek
üzere burada paylaşılmıştır.

Tek kameralı (mono) + IMU verisini birleştirerek dronun konumunu ve yönelimini
gerçek zamanlı olarak kestirir. Bu branch (`main`) **loop closure içermez**;
loop closure eklenmiş versiyon için [`loop-closure`](../../tree/loop-closure)
branch'ine bakın.

## Ne Yapar

- **Özellik çıkarımı/takibi**: ORB tabanlı, kareler arası sürekli takip
- **Başlatma**: VI-SfM tarzı (ölçek + yerçekimi + hız + IMU bias kestirimi)
- **Backend**: GTSAM factor graph — `CombinedImuFactor` + `SmartProjectionPoseFactor`,
  `IncrementalFixedLagSmoother` ile kayan pencere (sliding window)
- **Değerlendirme**: Umeyama hizalama + ATE (Absolute Trajectory Error)
- **Görselleştirme**: OpenCV ile canlı kamera görüntüsü + trajectory grafiği

## Sonuçlar (EuRoC MH_01_easy, 3682 kare, tam veri seti)

| Metrik | Değer | Hedef |
|---|---|---|
| ATE RMSE (rijit hizalama) | ~2.21 m | 0.3–0.6 m |
| Umeyama scale | ~0.87 | ~1.0 |
| FPS | ~46–58 | 30–60 |

Koşunun büyük kısmında ATE 0.2–1.0 m aralığında kalıyor; veri setinin sonuna
yakın gerçek bir ani manevra (~2m dalış-çıkış) final sonucu yukarı çekiyor.

## Bağımlılıklar

- CMake ≥ 3.16, C++17 derleyici
- OpenCV
- Eigen3
- [GTSAM](https://github.com/borglab/gtsam) (**`GTSAM_BUILD_UNSTABLE=ON`** ile derlenmeli —
  `gtsam_unstable` modülü gerekli)
- yaml-cpp

GTSAM sudo yetkisi olmayan bir ortamda kullanıcı-lokal bir dizine
(`third_party/install`) kaynak koddan derlenecek şekilde kurulabilir; bu
depoda GTSAM kurulumu **dahil değildir** (boyutu nedeniyle `.gitignore`'da).

## Veri Seti

[EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)'in
`MH_01_easy` dizisini indirip proje kökünde `data/MH_01_easy/` olacak şekilde
yerleştirin (klasör içinde `mav0/` alt dizini olmalı).

## Derleme ve Çalıştırma

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
./drone_vio ../data/MH_01_easy
```

Pencere içinde `q` veya `ESC` ile çıkılabilir. Program sonunda ATE/scale
raporu konsola yazdırılır.
