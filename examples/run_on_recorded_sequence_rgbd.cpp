// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)

#include <filesystem/filesystem.h>
#include <icg/basic_depth_renderer.h>
#include <icg/body.h>
#include <icg/common.h>
#include <icg/loader_camera.h>
#include <icg/manual_detector.h>
#include <icg/static_detector.h>
#include <icg/normal_viewer.h>
#include <icg/region_modality.h>
#include <icg/depth_modality.h>
// #include <icg/depth_model.h>
#include <icg/renderer_geometry.h>
#include <icg/tracker.h>

#include <Eigen/Geometry>
#include <memory>

// Example script for the detector usage on the data provided in data/sequence
// with the object triangle
int main() {
  const std::filesystem::path body_directory{"./metal/"};
  // std::vector<std::string> body_names{"xiao_chilun", "040_large_marker", "005_tomato_soup_can", "007_tuna_fish_can"};
	// std::vector<std::string> body_names{"dizuo", "da_chilun", "xiao_chilun", "da_quan_hou", "xiao_quan_hou", "gaizi", "da_gaizi_hou", "xiao_gaizi_hou"};
  std::vector<std::string> body_names{"gaizi"};

  constexpr bool kUseColorViewer = false;
  constexpr bool kUseDepthViewer = false;
  constexpr bool kMeasureOcclusions = true;
  constexpr bool kModelOcclusions = false;
  constexpr bool kVisualizePoseResult = false;

  const std::filesystem::path save_path{"./raw_poses"};
	const std::filesystem::path sequence_path{"./" + body_names[0] + "_yuan"};
  const std::filesystem::path color_camera_metafile_path{sequence_path / "color_camera.yaml"};
  const std::filesystem::path depth_camera_metafile_path{sequence_path / "depth_camera.yaml"};

  // Set up tracker and renderer geometry
  auto tracker_ptr{std::make_shared<icg::Tracker>("tracker")};
  auto renderer_geometry_ptr{
      std::make_shared<icg::RendererGeometry>("renderer_geometry")};

  // Set up camera
  auto color_camera_ptr{std::make_shared<icg::LoaderColorCamera>(
      "color_camera", color_camera_metafile_path)};
  auto depth_camera_ptr{std::make_shared<icg::LoaderDepthCamera>(
      "depth_camera", depth_camera_metafile_path)};

  // Set up viewers
  if (kUseColorViewer) {
    auto color_viewer_ptr{std::make_shared<icg::NormalColorViewer>(
        "color_viewer", color_camera_ptr, renderer_geometry_ptr)};
    tracker_ptr->AddViewer(color_viewer_ptr);
  }
  if (kUseDepthViewer) {
    auto depth_viewer_ptr{std::make_shared<icg::NormalDepthViewer>(
        "depth_viewer", depth_camera_ptr, renderer_geometry_ptr, 0.3f, 1.0f)};
    tracker_ptr->AddViewer(depth_viewer_ptr);
  }

  // Set up depth renderer
  auto color_depth_renderer_ptr{
      std::make_shared<icg::FocusedBasicDepthRenderer>(
          "color_depth_renderer", renderer_geometry_ptr, color_camera_ptr)};
  auto depth_depth_renderer_ptr{
      std::make_shared<icg::FocusedBasicDepthRenderer>(
          "depth_depth_renderer", renderer_geometry_ptr, depth_camera_ptr)};

  for (const auto body_name : body_names) {
    // Set up body
    std::filesystem::path metafile_path{body_directory / (body_name + ".yaml")};
    auto body_ptr{std::make_shared<icg::Body>(body_name, metafile_path)};
    renderer_geometry_ptr->AddBody(body_ptr);
    color_depth_renderer_ptr->AddReferencedBody(body_ptr);
    depth_depth_renderer_ptr->AddReferencedBody(body_ptr);

    // Set up detector
    std::filesystem::path detector_path{body_directory /
                                        (body_name + "_detector.yaml")};
    auto detector_ptr{std::make_shared<icg::StaticDetector>(
        body_name + "_detector", detector_path, body_ptr)};
    tracker_ptr->AddDetector(detector_ptr);

    // Set up models
    auto region_model_ptr{std::make_shared<icg::RegionModel>(
        body_name + "_region_model", body_ptr,
        body_directory / (body_name + "_region_model.bin"))};
    auto depth_model_ptr{std::make_shared<icg::DepthModel>(
        body_name + "_depth_model", body_ptr,
        body_directory / (body_name + "_depth_model.bin"))};

    // Set up modalities
    auto region_modality_ptr{std::make_shared<icg::RegionModality>(
        body_name + "_region_modality", body_ptr, color_camera_ptr,
        region_model_ptr)};
    auto depth_modality_ptr{std::make_shared<icg::DepthModality>(
        body_name + "_depth_modality", body_ptr, depth_camera_ptr,
        depth_model_ptr)};
    if (kVisualizePoseResult) {
      region_modality_ptr->set_visualize_pose_result(true);
    }
    if (kMeasureOcclusions) {
      region_modality_ptr->MeasureOcclusions(depth_camera_ptr);
      depth_modality_ptr->MeasureOcclusions();
    }
    if (kModelOcclusions) {
      region_modality_ptr->ModelOcclusions(color_depth_renderer_ptr);
      depth_modality_ptr->ModelOcclusions(depth_depth_renderer_ptr);
    }

    // Set up optimizer
    auto body1_optimizer_ptr{
        std::make_shared<icg::Optimizer>(body_name + "_optimizer")};
    body1_optimizer_ptr->AddModality(region_modality_ptr);
    body1_optimizer_ptr->AddModality(depth_modality_ptr);
    tracker_ptr->AddOptimizer(body1_optimizer_ptr);
  }

  tracker_ptr->set_n_update_iterations(2);
  tracker_ptr->set_n_corr_iterations(2);

  // Start tracking
  if (!tracker_ptr->SetUp()) return 0;
  tracker_ptr->startPoseAnnotationAndSave(save_path);
  // if (!tracker_ptr->RunTrackerProcess(true, true)) return 0;
  tracker_ptr->RunTrackerProcess(true, true);
	tracker_ptr->savePose();
  return 0;
}
