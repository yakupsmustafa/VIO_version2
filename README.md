# Drone VIO — Versiyon 2 (Loop Closure)

Bu proje, bir drone için geliştirilen Visual-Inertial Odometry (VIO) sisteminin
**2. versiyonudur**. Unreal Engine tabanlı bir simülasyon ortamında denenmek
üzere burada paylaşılmıştır.

Tek kameralı (mono) + IMU verisini birleştirerek dronun konumunu ve yönelimini
gerçek zamanlı olarak kestirir. Bu branch (`loop-closure`), temel sisteme
**loop closure** (daha önce görülen bir yeri tanıyıp birikmiş sürüklenmeyi
düzeltme) ekler. Loop closure içermeyen sade hali için
[`main`](../../tree/main) branch'ine bakın.

## Ne Yapar

- **Özellik çıkarımı/takibi**: ORB tabanlı, kareler arası sürekli takip
- **Başlatma**: VI-SfM tarzı (ölçek + yerçekimi + hız + IMU bias kestirimi)
- **Backend**: GTSAM factor graph — `CombinedImuFactor` + `SmartProjectionPoseFactor`,
  `IncrementalFixedLagSmoother` ile kayan pencere (sliding window)
- **Loop closure** (bu branch'e özel): her keyframe'in ORB verisi ve geçerli
  3D landmark'ları ayrı bir veritabanında (`KeyframeDatabase`) saklanır; yeterince
  eski bir kareyle eşleşme bulunursa (`LoopClosureDetector`, PnP-RANSAC ile
  METRİK doğrulama) ayrı bir pose-graph (`PoseGraphOptimizer`) TÜM geçmiş
  trajectory'yi (ana backend'in kayan pencereden düşürdüğü eski kareler dahil)
  yeniden hizalar
- **Değerlendirme**: Umeyama hizalama + ATE (Absolute Trajectory Error)
- **Görselleştirme**: OpenCV ile canlı kamera görüntüsü + trajectory grafiği

## Sonuçlar (EuRoC MH_01_easy, 3682 kare, tam veri seti)

Loop closure bulunduğu bölgelerde sonucu belirgin şekilde iyileştiriyor
(örn. bir koşuda keyframe 220 civarında ATE=0.19m, scale≈1.0'a kadar
düşüyor/yaklaşıyor). Ancak aday arama, ölçeklenebilirlik için
sadeleştirilmiş (DBoW2 gibi bir "vocabulary" tabanlı indeks yerine
brute-force + örnekleme) olduğundan her zorlu segmenti yakalayamayabilir;
bu durumda final sonuç loop-closure'suz hale yakın kalır. FPS ortalama ~25
(hedef 30–60'ın bir miktar altında, loop closure'ın ek maliyeti nedeniyle).

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

Pencere içinde `q` veya `ESC` ile çıkılabilir. Bir loop closure bulunduğunda
konsola `[LOOP CLOSURE] ...` satırı yazdırılır. Program sonunda ATE/scale
raporu konsola yazdırılır.
