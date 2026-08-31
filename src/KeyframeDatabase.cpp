// =====================================================================================
// KeyframeDatabase.cpp
// ---------------------------------------------------------------------------
// KeyframeDatabase.hpp icinde bildirilen sinifin implementasyonu.
// =====================================================================================

#include "KeyframeDatabase.hpp"

#include <stdexcept>

namespace vio {

void KeyframeDatabase::add(KeyframeRecord record) {
  const int id = record.keyframe_id;
  insertion_order_.push_back(id);
  records_.emplace(id, std::move(record));
}

std::vector<int> KeyframeDatabase::candidateIds(int current_keyframe_id, int min_keyframe_gap) const {
  std::vector<int> result;
  result.reserve(insertion_order_.size());
  for (int id : insertion_order_) {
    if (current_keyframe_id - id >= min_keyframe_gap) {
      result.push_back(id);
    }
  }
  return result;
}

const KeyframeRecord& KeyframeDatabase::at(int keyframe_id) const {
  auto it = records_.find(keyframe_id);
  if (it == records_.end()) {
    throw std::out_of_range("KeyframeDatabase::at: keyframe_id bulunamadi: " + std::to_string(keyframe_id));
  }
  return it->second;
}

}  // namespace vio
