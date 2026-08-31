cmake_minimum_required(VERSION 3.16)
project(drone_vio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# GTSAM was built from source into third_party/install (user-local prefix, no sudo used).
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/third_party/install")

# Bake the local GTSAM lib dir into rpath so running the binary doesn't
# require manually setting LD_LIBRARY_PATH.
set(CMAKE_BUILD_RPATH "${CMAKE_SOURCE_DIR}/third_party/install/lib")

# libgtsam.so itself depends on libmetis-gtsam.so but carries no rpath of its
# own. Modern linkers default to DT_RUNPATH, which (unlike the older
# DT_RPATH) is NOT inherited when resolving a dependency's own dependencies -
# so our executable's rpath would resolve libgtsam.so but not libgtsam.so's
# need for libmetis-gtsam.so. --disable-new-dtags forces classic DT_RPATH,
# which does propagate down the whole dependency chain.
add_link_options(-Wl,--disable-new-dtags)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(drone_vio
  main.cpp
  src/DataLoader.cpp
  src/CameraModel.cpp
  src/Pipo1_FeatureExtraction.cpp
  src/Pipo2_Matcher.cpp
  src/FeatureTracker.cpp
  src/Triangulation.cpp
  src/ImuPreintegration.cpp
  src/VioInitializer.cpp
  src/FactorGraphBackend.cpp
  src/GtAligner.cpp
  src/AteEvaluator.cpp
  src/Visualizer.cpp
  src/KeyframeDatabase.cpp
  src/LoopClosureDetector.cpp
  src/PoseGraphOptimizer.cpp
)

target_include_directories(drone_vio PRIVATE
  include
)

target_link_libraries(drone_vio PRIVATE
  ${OpenCV_LIBS}
  gtsam
  gtsam_unstable
  Eigen3::Eigen
  yaml-cpp
)
