// =====================================================================================
// FpsCounter.hpp
// ---------------------------------------------------------------------------
// BU HEADER'IN GOREVI:
//   Islenen kare basina gecen SURE'yi olcup, bunu "saniyede kare" (FPS)
//   degerine cevirmek.
//
//   NEDEN "SON N KARENIN ORTALAMASI"? Tek bir karenin suresini olcup 1000/sure
//   seklinde FPS hesaplamak COK GURULTULU olur (bir kare disk I/O'su yuzunden
//   yavaslarsa FPS aniden dusuyormus gibi gorunur). Bunun yerine SON N
//   karenin (varsayilan 30) TOPLAM suresi uzerinden ortalama FPS hesaplamak,
//   goze daha "stabil" ve anlamli bir sayi verir - oyun motorlarinda ve
//   gercek-zamanli sistemlerde yaygin kullanilan bir teknik.
// =====================================================================================

#pragma once

#include <chrono>
#include <deque>

namespace vio {

class FpsCounter {
 public:
  explicit FpsCounter(int window_size = 30) : window_size_(window_size) {}

  // Her kare islendiginde (dongunun basinda veya sonunda, tutarli bir
  // noktada) bir kez cagrilmali.
  void tick() {
    const auto now = std::chrono::steady_clock::now();
    if (has_last_tick_) {
      const double dt_s = std::chrono::duration<double>(now - last_tick_).count();
      frame_times_s_.push_back(dt_s);
      if (static_cast<int>(frame_times_s_.size()) > window_size_) {
        frame_times_s_.pop_front();
      }
    }
    last_tick_ = now;
    has_last_tick_ = true;
  }

  // Su anki (son window_size kare uzerinden ortalama) FPS tahmini. Henuz
  // yeterli veri yoksa 0 doner.
  double fps() const {
    if (frame_times_s_.empty()) return 0.0;
    double total = 0.0;
    for (double dt : frame_times_s_) total += dt;
    const double avg_dt = total / static_cast<double>(frame_times_s_.size());
    return (avg_dt > 0.0) ? (1.0 / avg_dt) : 0.0;
  }

 private:
  int window_size_;
  std::deque<double> frame_times_s_;
  std::chrono::steady_clock::time_point last_tick_;
  bool has_last_tick_ = false;
};

}  // namespace vio
