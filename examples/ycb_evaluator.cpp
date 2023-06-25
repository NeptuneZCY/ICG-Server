// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)

#include "ycb_evaluator.h"
#include <icg/gm11.h>
#include <cmath>

YCBEvaluator::YCBEvaluator(const std::string &name,
                           const std::filesystem::path &dataset_directory,
                           const std::filesystem::path &external_directory,
                           const std::vector<int> &sequence_ids,
                           const std::vector<std::string> &evaluated_body_names)
    : name_{name},
      dataset_directory_{dataset_directory},
      external_directory_{external_directory},
      sequence_ids_{sequence_ids},
      evaluated_body_names_{evaluated_body_names} {
  // Compute thresholds used to compute AUC score
  float threshold_step = kThresholdMax / float(kNCurveValues);
  for (size_t i = 0; i < kNCurveValues; ++i) {
    thresholds_[i] = threshold_step * (0.5f + float(i));
  }
}

bool YCBEvaluator::SetUp() {
  set_up_ = false;

  if (!CreateRunConfigurations()) return false;
  AssambleTrackedBodyNames();
  if (!LoadBodies()) return false;
  if (!GenerateModels()) return false;
  if (!LoadKeyframes()) return false;
  LoadNFrames();
  LoadPoseBegin();
  GenderateReducedVertices();
  GenerateKDTrees();

  set_up_ = true;
  return true;
}

void YCBEvaluator::set_detector_folder(const std::string &detector_folder) {
  detector_folder_ = detector_folder;
}

void YCBEvaluator::set_evaluate_refinement(bool evaluate_refinement) {
  evaluate_refinement_ = evaluate_refinement;
}

void YCBEvaluator::set_use_detector_initialization(
    bool use_detector_initialization) {
  use_detector_initialization_ = use_detector_initialization;
}

void YCBEvaluator::set_use_matlab_gt_poses(bool use_matlab_gt_poses) {
  use_matlab_gt_poses_ = use_matlab_gt_poses;
}

void YCBEvaluator::set_run_sequentially(bool run_sequentially) {
  run_sequentially_ = run_sequentially;
}

void YCBEvaluator::set_use_random_seed(bool use_random_seed) {
  use_random_seed_ = use_random_seed;
}

void YCBEvaluator::set_n_vertices_evaluation(int n_vertices_evaluation) {
  n_vertices_evaluation_ = n_vertices_evaluation;
}

void YCBEvaluator::set_visualize_tracking(bool visualize_tracking) {
  visualize_tracking_ = visualize_tracking;
}

void YCBEvaluator::set_visualize_frame_results(bool visualize_frame_results) {
  visualize_frame_results_ = visualize_frame_results;
}

void YCBEvaluator::StartSavingImages(
    const std::filesystem::path &save_directory) {
  save_images_ = true;
  save_directory_ = save_directory;
}

void YCBEvaluator::StopSavingImages() { save_images_ = false; }

void YCBEvaluator::set_use_region_modality(bool use_region_modality) {
  use_region_modality_ = use_region_modality;
}

void YCBEvaluator::set_use_depth_modality(bool use_depth_modality) {
  use_depth_modality_ = use_depth_modality;
}

void YCBEvaluator::set_measure_occlusions_region(
    bool measure_occlusions_region) {
  measure_occlusions_region_ = measure_occlusions_region;
}

void YCBEvaluator::set_measure_occlusions_depth(bool measure_occlusions_depth) {
  measure_occlusions_depth_ = measure_occlusions_depth;
}

void YCBEvaluator::set_model_occlusions_region(bool model_occlusions_region) {
  model_occlusions_region_ = model_occlusions_region;
}

void YCBEvaluator::set_model_occlusions_depth(bool model_occlusions_depth) {
  model_occlusions_depth_ = model_occlusions_depth;
}

void YCBEvaluator::set_tracker_setter(
    const std::function<void(std::shared_ptr<icg::Tracker>)> &tracker_setter) {
  tracker_setter_ = tracker_setter;
}

void YCBEvaluator::set_refiner_setter(
    const std::function<void(std::shared_ptr<icg::Refiner>)> &refiner_setter) {
  refiner_setter_ = refiner_setter;
}

void YCBEvaluator::set_optimizer_setter(
    const std::function<void(std::shared_ptr<icg::Optimizer>)>
        &optimizer_setter) {
  optimizer_setter_ = optimizer_setter;
}

void YCBEvaluator::set_region_modality_setter(
    const std::function<void(std::shared_ptr<icg::RegionModality>)>
        &region_modality_setter) {
  region_modality_setter_ = region_modality_setter;
}

void YCBEvaluator::set_region_model_setter(
    const std::function<void(std::shared_ptr<icg::RegionModel>)>
        &region_model_setter) {
  region_model_setter_ = region_model_setter;
  set_up_ = false;
}

void YCBEvaluator::set_depth_modality_setter(
    const std::function<void(std::shared_ptr<icg::DepthModality>)>
        &depth_modality_setter) {
  depth_modality_setter_ = depth_modality_setter;
}

void YCBEvaluator::set_depth_model_setter(
    const std::function<void(std::shared_ptr<icg::DepthModel>)>
        &depth_model_setter) {
  depth_model_setter_ = depth_model_setter;
  set_up_ = false;
}

bool YCBEvaluator::Evaluate() {
  if (!set_up_) {
    std::cerr << "Set up evaluator " << name_ << " first" << std::endl;
    return false;
  }
  if (run_configurations_.empty()) return false;

  // Evaluate all run configurations
  results_.clear();
  final_results_.clear();

  // 多线程
  if (run_sequentially_ || visualize_tracking_ || visualize_frame_results_) {
  // if (0) {
    auto renderer_geometry_ptr{std::make_shared<icg::RendererGeometry>("rg")};
    renderer_geometry_ptr->SetUp();
    for (size_t i = 0; i < int(run_configurations_.size()); ++i) {
      std::vector<SequenceResult> sequence_results;
      if (!EvaluateRunConfiguration(run_configurations_[i],
                                    renderer_geometry_ptr, &sequence_results))
        continue;
      results_.insert(end(results_), begin(sequence_results),
                      end(sequence_results));
      for (const auto &sequence_result : sequence_results) {
        std::string title{sequence_result.sequence_name + ": " +
                          sequence_result.body_name};
        VisualizeResult(sequence_result.average_result, title);
        VisualizeResult(sequence_result.average_stable_result, title);
      }
    }
  } else {
    std::vector<std::shared_ptr<icg::RendererGeometry>> renderer_geometry_ptrs(
        omp_get_max_threads());
    for (auto &renderer_geometry_ptr : renderer_geometry_ptrs) {
      renderer_geometry_ptr = std::make_shared<icg::RendererGeometry>("rg");
      renderer_geometry_ptr->SetUp();
    }
#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < int(run_configurations_.size()); ++i) {
      std::vector<SequenceResult> sequence_results;
      if (!EvaluateRunConfiguration(
              run_configurations_[i],
              renderer_geometry_ptrs[omp_get_thread_num()], &sequence_results))
        continue;
#pragma omp critical
      {
        results_.insert(end(results_), begin(sequence_results),
                        end(sequence_results));
        for (const auto &sequence_result : sequence_results) {
          std::string title{sequence_result.sequence_name + ": " +
                            sequence_result.body_name};
          VisualizeResult(sequence_result.average_result, title);
          VisualizeResult(sequence_result.average_stable_result, title);
        }
      }
    }
  }

  float total_time = 0;
  // Calculate sequnce results
  std::cout << std::endl << std::string(80, '-') << std::endl;
  for (int sequence_id : sequence_ids_) {
    auto sequence_name{SequenceIDToName(sequence_id)};
    Result result{CalculateAverageSequenceResult({sequence_name})};
    VisualizeResult(result, sequence_name);
    total_time += result.execution_times.complete_cycle;
  }
  
  std::cout << std::endl << std::string(80, '-') << std::endl;
  int total_frame = 2949;
  std::cout << "total_time: " << total_time << "us" << std::endl;

  // Calculate body results
  std::cout << std::endl << std::string(80, '-') << std::endl;
  for (const auto &body_name : evaluated_body_names_) {
    Result result{CalculateAverageBodyResult({body_name})};
    final_results_.insert({body_name, result});
    VisualizeResult(result, body_name);
  }

  // Calculate average results
  Result final_result_all{CalculateAverageBodyResult({evaluated_body_names_})};
  final_results_.insert({"all", final_result_all});
  VisualizeResult(final_result_all, "all");
  std::cout << std::string(80, '-') << std::endl;

  // Calculate average results
  Result final_result_all_stable{CalculateAverageBodyResult2({evaluated_body_names_})};
  final_results_.insert({"all_stable", final_result_all_stable});
  VisualizeResult(final_result_all_stable, "all_stable");
  std::cout << std::string(80, '-') << std::endl;

  return true;
}

void YCBEvaluator::SaveResults(std::filesystem::path path) const {
  std::ofstream ofs{path};
  for (auto const &[body_name, result] : final_results_) {
    ofs << body_name << "," << result.add_auc << "," << result.adds_auc << ","
        << result.execution_times.complete_cycle << ","
        << result.execution_times.start_modalities << ","
        << result.execution_times.calculate_correspondences << ","
        << result.execution_times.calculate_gradient_and_hessian << ","
        << result.execution_times.calculate_optimization << ","
        << result.execution_times.calculate_results << std::endl;
  }
  ofs.flush();
  ofs.close();
}

float YCBEvaluator::add_auc() const { return final_results_.at("all").add_auc; }

float YCBEvaluator::adds_auc() const {
  return final_results_.at("all").adds_auc;
}

float YCBEvaluator::execution_time() const {
  return final_results_.at("all").execution_times.complete_cycle;
}

std::map<std::string, YCBEvaluator::Result> YCBEvaluator::final_results()
    const {
  return final_results_;
}

bool YCBEvaluator::EvaluateRunConfiguration(
    const RunConfiguration &run_configuration,
    const std::shared_ptr<icg::RendererGeometry> &renderer_geometry_ptr,
    std::vector<SequenceResult> *sequence_results) {
  const auto &sequence_name{run_configuration.sequence_name};
  const auto &evaluated_body_names{run_configuration.evaluated_body_names};
  const auto &tracked_body_names{run_configuration.tracked_body_names};
  const auto &keyframes{sequence2keyframes_map_.at(sequence_name)};
  // std::cout<< "sequence_name: " << sequence_name <<std::endl;
  // for(auto index: keyframes){
  //     std::cout<< "keyframe: " << index <<std::endl;
  // }

  size_t n_keyframes = keyframes.size();

  // Initialize tracker
  std::shared_ptr<icg::Tracker> tracker_ptr;
  if (!SetUpTracker(run_configuration, renderer_geometry_ptr, &tracker_ptr))
    return false;

  // Read gt poses and detector poses
  std::map<std::string, std::vector<icg::Transform3fA>> gt_body2world_poses;
  std::map<std::string, std::vector<icg::Transform3fA>>
      detector_body2world_poses;
  std::map<std::string, std::vector<bool>> body_detected;
  for (const auto &body_name : tracked_body_names) {
    if (use_matlab_gt_poses_) {
      // std::cout << "LoadMatlabGTPoses running......................................................." << std::endl;
      if (!LoadMatlabGTPoses(sequence_name, body_name,
                             &gt_body2world_poses[body_name]))
        return false;
    } else {
      // std::cout << "LoadGTPoses running......................................................." << std::endl;
      if (!LoadGTPoses(sequence_name, body_name,
                       &gt_body2world_poses[body_name]))
        return false;
    }
    if (evaluate_refinement_) {
      // std::cout << "LoadDetectorPoses running......................................................." << std::endl;
      if (!LoadDetectorPoses(sequence_name, body_name,
                             &detector_body2world_poses[body_name],
                             &body_detected[body_name]))
        return false;
    }
  }

  // Init results
  sequence_results->resize(evaluated_body_names.size());
  // std::cout << "evaluated_body_names.size(): " << evaluated_body_names.size() << std::endl;
  
  for (int i = 0; i < evaluated_body_names.size(); ++i) {
    (*sequence_results)[i].sequence_name = sequence_name;
    (*sequence_results)[i].body_name = evaluated_body_names[i];
    (*sequence_results)[i].frame_results.clear();
  }

  std::chrono::high_resolution_clock::time_point all_begin_time;
  all_begin_time = std::chrono::high_resolution_clock::now();
  // Iterate over all frames
  for (int i = 0; i < n_keyframes; ++i) {
    // if(i%2 == 1) continue;
    Result general_results;
    general_results.frame_index = i;

    // Execute different evaluations
    // std::cout << "evaluate_refinement_: " << evaluate_refinement_ << std::endl;
    if (evaluate_refinement_) {
      ResetBodies(tracker_ptr, tracked_body_names, detector_body2world_poses,
                  i);
      ExecuteMeasuredRefinementCycle(tracker_ptr, keyframes[i], i,
                                     &general_results.execution_times);
    } else {
      // 第0帧初始化
      if (i == 0) {
        ResetBodies(tracker_ptr, tracked_body_names, gt_body2world_poses, 0);
        UpdateCameras(tracker_ptr, keyframes[0]);
        tracker_ptr->StartModalities(0);
      }
      ExecuteMeasuredTrackingCycle(tracker_ptr, keyframes[i], i,
                                   &general_results.execution_times);
    }



    // Calculate results
    for (int j = 0; j < evaluated_body_names.size(); ++j) {
      const auto &body_name{evaluated_body_names[j]};
      Result result{general_results};
      Result stable_result{general_results}; // 抖动adds计算结果
      CalculatePoseResults(tracker_ptr, body_name,
                           gt_body2world_poses[body_name][i], &result, &stable_result);

      // visualize_frame_results_ = 1;
      if (visualize_frame_results_){
        std::cout<<"result: "<<std::endl;
        VisualizeResult(result, sequence_name + ": " + body_name);
        std::cout<<"stable_result: "<<std::endl;
        VisualizeResult(stable_result, sequence_name + ": " + body_name);
      }

      (*sequence_results)[j].frame_results.push_back(std::move(result));
      (*sequence_results)[j].stable_results.push_back(std::move(stable_result));
    }

    // 记录本帧位姿
    tracker_ptr -> RecordLastPose();
  }

  float sequence_time = ElapsedTime(all_begin_time);
  std::cout << "frames: " << n_keyframes << ", time: " << sequence_time / 1000000.0 << "us" << std::endl;
  all_time_ += sequence_time;
  all_frames_ += n_keyframes;

  // Calculate Average Results
  for (auto &sequence_result : *sequence_results){
    sequence_result.average_result =
        CalculateAverageResult(sequence_result.frame_results);
    sequence_result.average_stable_result =
        CalculateAverageResult(sequence_result.stable_results);
  }
  return true;
}

bool YCBEvaluator::SetUpTracker(
    const RunConfiguration &run_configuration,
    const std::shared_ptr<icg::RendererGeometry> &renderer_geometry_ptr,
    std::shared_ptr<icg::Tracker> *tracker_ptr) const {
  renderer_geometry_ptr->ClearBodies();
  *tracker_ptr = std::make_shared<icg::Tracker>("tracker");
  auto refiner_ptr{std::make_shared<icg::Refiner>("refiner")};

  // Init cameras
  std::filesystem::path camera_directory{dataset_directory_ / "data" /
                                         run_configuration.sequence_name};
  auto color_camera_ptr{std::make_shared<icg::LoaderColorCamera>(
      "color_camera", camera_directory, kYCBIntrinsics, "", 1, 6, "-color")};
  if (save_region_photo_){
    color_camera_ptr->StartSavingImages(photo_directory_ / "rgb");
  }
  color_camera_ptr->SetUp();
  auto depth_camera_ptr{std::make_shared<icg::LoaderDepthCamera>(
      "depth_camera", camera_directory, kYCBIntrinsics, 0.0001f, "", 1, 6,
      "-depth")};

  if (save_depth_photo_){
    depth_camera_ptr->StartSavingImages(photo_directory_ / "depth");
  }
  depth_camera_ptr->SetUp();

  // Init visualizer
  auto save_directory_root{save_directory_ / "icg"};
  if (save_images_) std::filesystem::create_directory(save_directory_root);



  auto save_directory_sequence{save_directory_root / 
                               run_configuration.sequence_name /
                               run_configuration.tracked_body_names[0]
                               };
  

  if (save_images_) {
    std::filesystem::create_directory(save_directory_root / 
                               run_configuration.sequence_name);
    std::filesystem::create_directory(save_directory_sequence);
    std::filesystem::create_directory(save_directory_sequence / "rgb");
    std::filesystem::create_directory(save_directory_sequence / "depth");
  }

  if (use_region_modality_ && (visualize_tracking_ || save_images_)) {
    auto color_viewer_ptr{std::make_shared<icg::NormalColorViewer>(
        "color_viewer", color_camera_ptr, renderer_geometry_ptr)};
    color_viewer_ptr->set_display_images(visualize_tracking_);
    if (save_images_)
      color_viewer_ptr->StartSavingImages(save_directory_sequence / "rgb");
    color_viewer_ptr->SetUp();
    (*tracker_ptr)->AddViewer(color_viewer_ptr);
  }
  if (use_depth_modality_ && (visualize_tracking_ || save_images_)) {
    auto depth_viewer_ptr{std::make_shared<icg::NormalDepthViewer>(
        "depth_viewer", depth_camera_ptr, renderer_geometry_ptr, 0.3f, 1.0f)};
    depth_viewer_ptr->set_display_images(visualize_tracking_);
    if (save_images_)
      depth_viewer_ptr->StartSavingImages(save_directory_sequence / "depth");
    depth_viewer_ptr->SetUp();
    (*tracker_ptr)->AddViewer(depth_viewer_ptr);
  }

  // Init renderer
  auto color_depth_renderer_ptr{
      std::make_shared<icg::FocusedBasicDepthRenderer>(
          "color_depth_renderer", renderer_geometry_ptr, color_camera_ptr)};
  auto depth_depth_renderer_ptr{
      std::make_shared<icg::FocusedBasicDepthRenderer>(
          "depth_depth_renderer", renderer_geometry_ptr, depth_camera_ptr)};

  // Iterate over tracked bodies
  for (const auto &body_name : run_configuration.tracked_body_names) {
    // Init body
    auto body_ptr{
        std::make_shared<icg::Body>(*body2body_ptr_map_.at(body_name))};
    renderer_geometry_ptr->AddBody(body_ptr);

    // Init optimizer
    auto optimizer_ptr{
        std::make_shared<icg::Optimizer>(body_name + "_optimizer")};

    // Init region modality
    if (use_region_modality_) {
      auto region_modality_ptr{std::make_shared<icg::RegionModality>(
          body_name + "_region_modality", body_ptr, color_camera_ptr,
          body2region_model_ptr_map_.at(body_name))};
      region_modality_ptr->set_n_unoccluded_iterations(0);
      region_modality_setter_(region_modality_ptr);
      if (measure_occlusions_region_)
        region_modality_ptr->MeasureOcclusions(depth_camera_ptr);
      if (model_occlusions_region_) {
        color_depth_renderer_ptr->AddReferencedBody(body_ptr);
        color_depth_renderer_ptr->SetUp();
        region_modality_ptr->ModelOcclusions(color_depth_renderer_ptr);
      }
      region_modality_ptr->SetUp();
      optimizer_ptr->AddModality(region_modality_ptr);
    }

    // Init depth modality
    if (use_depth_modality_) {
      auto depth_modality_ptr{std::make_shared<icg::DepthModality>(
          body_name + "_depth_modality", body_ptr, depth_camera_ptr,
          body2depth_model_ptr_map_.at(body_name))};
      depth_modality_ptr->set_n_unoccluded_iterations(0);
      depth_modality_setter_(depth_modality_ptr);
      if (measure_occlusions_depth_) depth_modality_ptr->MeasureOcclusions();
      if (model_occlusions_depth_) {
        depth_depth_renderer_ptr->AddReferencedBody(body_ptr);
        depth_depth_renderer_ptr->SetUp();
        depth_modality_ptr->ModelOcclusions(depth_depth_renderer_ptr);
      }
      depth_modality_ptr->SetUp();
      optimizer_ptr->AddModality(depth_modality_ptr);
    }

    // Init optimizer
    optimizer_setter_(optimizer_ptr);
    optimizer_ptr->SetUp();
    (*tracker_ptr)->AddOptimizer(optimizer_ptr);
    refiner_ptr->AddOptimizer(optimizer_ptr);
  }

  // Init refiner
  refiner_setter_(refiner_ptr);
  refiner_ptr->SetUp(false);
  (*tracker_ptr)->AddRefiner(refiner_ptr);

  // Init tracker
  tracker_setter_(*tracker_ptr);
  return (*tracker_ptr)->SetUp(false);
}

void YCBEvaluator::ResetBodies(
    const std::shared_ptr<icg::Tracker> &tracker_ptr,
    const std::vector<std::string> &body_names,
    const std::map<std::string, std::vector<icg::Transform3fA>>
        &body2world_poses,
    int idx) const {
  for (const auto &body_name : body_names) {
    for (auto &modality_ptr : tracker_ptr->modality_ptrs()) {
      if (modality_ptr->body_ptr()->name() == body_name)
        modality_ptr->body_ptr()->set_body2world_pose(
            body2world_poses.at(body_name)[idx]);
    }
  }
}

void YCBEvaluator::ExecuteMeasuredRefinementCycle(
    const std::shared_ptr<icg::Tracker> &tracker_ptr, int load_index,
    int iteration, ExecutionTimes *execution_times) const {
  std::chrono::high_resolution_clock::time_point begin_time;
  auto refiner_ptr{tracker_ptr->refiner_ptrs()[0]};

  // Update Cameras
  UpdateCameras(tracker_ptr, load_index);

  execution_times->start_modalities = 0.0f;
  execution_times->calculate_correspondences = 0.0f;
  execution_times->calculate_gradient_and_hessian = 0.0f;
  execution_times->calculate_optimization = 0.0f;
  for (int corr_iteration = 0;
       corr_iteration < refiner_ptr->n_corr_iterations(); ++corr_iteration) {
    // Start modalities
    begin_time = std::chrono::high_resolution_clock::now();
    refiner_ptr->StartModalities(corr_iteration);
    execution_times->start_modalities += ElapsedTime(begin_time);

    // Calculate correspondences
    begin_time = std::chrono::high_resolution_clock::now();
    refiner_ptr->CalculateCorrespondences(corr_iteration);
    execution_times->calculate_correspondences += ElapsedTime(begin_time);

    // Visualize correspondences
    int corr_save_idx = corr_iteration;
    refiner_ptr->VisualizeCorrespondences(corr_save_idx);

    for (int update_iteration = 0;
         update_iteration < refiner_ptr->n_update_iterations();
         ++update_iteration) {
      // Calculate gradient and hessian
      begin_time = std::chrono::high_resolution_clock::now();
      refiner_ptr->CalculateGradientAndHessian(corr_iteration,
                                               update_iteration);
      execution_times->calculate_gradient_and_hessian +=
          ElapsedTime(begin_time);

      // Calculate optimization
      begin_time = std::chrono::high_resolution_clock::now();
      refiner_ptr->CalculateOptimization(corr_iteration, update_iteration);
      execution_times->calculate_optimization += ElapsedTime(begin_time);

      // Visualize pose update
      int update_save_idx =
          corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
      refiner_ptr->VisualizeOptimization(update_save_idx);
    }
  }

  // Update viewer
  if (visualize_tracking_ || save_images_)
    tracker_ptr->UpdateViewers(iteration);

  execution_times->calculate_results = 0.0f;
  execution_times->complete_cycle =
      execution_times->start_modalities +
      execution_times->calculate_correspondences +
      execution_times->calculate_gradient_and_hessian +
      execution_times->calculate_optimization;
}

class RegionTrajectory
{
	cv::Mat1b  _pathMask;
	float  _delta;

	cv::Point2f _uv2Pt(const cv::Point2f& uv)
	{
		return cv::Point2f(uv.x / _delta + float(_pathMask.cols) / 2.f, uv.y / _delta + float(_pathMask.rows) / 2.f);
	}
public:
	RegionTrajectory(cv::Size regionSize, float delta)
	{
		_pathMask = cv::Mat1b::zeros(regionSize);
		_delta = delta;
	}
	bool  addStep(cv::Point2f start, cv::Point2f end)
	{
		start = _uv2Pt(start);
		end = _uv2Pt(end);

		auto dv = end - start;
		float len = sqrt(dv.dot(dv)) + 1e-6f;
		float dx = dv.x / len, dy = dv.y / len;
		const int  N = int(len) + 1;
		cv::Point2f p = start;
		for (int i = 0; i < N; ++i)
		{
			int x = int(p.x + 0.5), y = int(p.y + 0.5);
			if (uint(x) < uint(_pathMask.cols) && uint(y) < uint(_pathMask.rows))
			{
				if (_pathMask(y, x) != 0 && len > 1.f)
					return true;
				_pathMask(y, x) = 1;
			}
			else
				return false;

			p.x += dx; p.y += dy;
		}

		return false;
	}

};

void YCBEvaluator::ExecuteMeasuredTrackingCycle(
    const std::shared_ptr<icg::Tracker> &tracker_ptr, int load_index,
    int iteration, ExecutionTimes *execution_times) const {
  std::chrono::high_resolution_clock::time_point begin_time;
  std::chrono::high_resolution_clock::time_point all_begin_time;
  all_begin_time = std::chrono::high_resolution_clock::now();

  // Update Cameras
  UpdateCameras(tracker_ptr, load_index);

  execution_times->start_modalities = 0.0f;
  execution_times->calculate_correspondences = 0.0f;
  execution_times->calculate_gradient_and_hessian = 0.0f;
  execution_times->calculate_optimization = 0.0f;

  // 用当前body的位姿来更新位姿
  auto update = [&](){

    auto &modality_ptr = tracker_ptr->modality_ptrs()[0]; 
    for (int corr_iteration = 0;
        corr_iteration < tracker_ptr->n_corr_iterations(); ++corr_iteration) {

      // std::cout<<"corr_iteration: "<< corr_iteration <<std::endl;

      // Calculate correspondences
      begin_time = std::chrono::high_resolution_clock::now();
      tracker_ptr->CalculateCorrespondences(iteration, corr_iteration);
      execution_times->calculate_correspondences += ElapsedTime(begin_time);

      // Visualize correspondences
      int corr_save_idx =
          iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
      tracker_ptr->VisualizeCorrespondences(corr_save_idx);

      for (int update_iteration = 0;
          update_iteration < tracker_ptr->n_update_iterations();
          ++update_iteration) {

        // std::cout << "iteration, corr_iteration, update_iteration: "<< iteration << " " << corr_iteration << " " <<update_iteration <<std::endl;

        // Calculate gradient and hessian
        begin_time = std::chrono::high_resolution_clock::now();
        tracker_ptr->CalculateGradientAndHessian(iteration, corr_iteration,
                                                update_iteration);
        execution_times->calculate_gradient_and_hessian +=
            ElapsedTime(begin_time);

        // Calculate optimization
        begin_time = std::chrono::high_resolution_clock::now();
        tracker_ptr->CalculateOptimization(iteration, corr_iteration,
                                          update_iteration);
        execution_times->calculate_optimization += ElapsedTime(begin_time);

        // Visualize pose update
        int update_save_idx =
            corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
        tracker_ptr->VisualizeOptimization(update_save_idx);
      }
    }
  };

  auto updateByPara = [&](int outer_iterations, 
                          int inner_iterations, 
                          float learning_rate){

    auto &modality_ptr = tracker_ptr->modality_ptrs()[0]; 

    // std::cout<<"body_name: "<<modality_ptr->name() <<std::endl;
    icg::Modality* m_ptr = modality_ptr.get();
    icg::RegionModality* region_modality_ptr = dynamic_cast<icg::RegionModality*>(m_ptr);

    region_modality_ptr->set_learning_rate(learning_rate);

    for (int corr_iteration = 0;
        corr_iteration < outer_iterations; ++corr_iteration) {

      // std::cout<<"corr_iteration: "<< corr_iteration <<std::endl;

      // Calculate correspondences
      begin_time = std::chrono::high_resolution_clock::now();
      tracker_ptr->CalculateCorrespondences(iteration, corr_iteration);
      execution_times->calculate_correspondences += ElapsedTime(begin_time);

      // Visualize correspondences
      int corr_save_idx =
          iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
      tracker_ptr->VisualizeCorrespondences(corr_save_idx);

      for (int update_iteration = 0;
          update_iteration < inner_iterations;
          ++update_iteration) {

        // std::cout << "iteration, corr_iteration, update_iteration: "<< iteration << " " << corr_iteration << " " <<update_iteration <<std::endl;

        // Calculate gradient and hessian
        begin_time = std::chrono::high_resolution_clock::now();
        tracker_ptr->CalculateGradientAndHessian(iteration, corr_iteration,
                                                update_iteration);
        execution_times->calculate_gradient_and_hessian +=
            ElapsedTime(begin_time);

        // Calculate optimization
        begin_time = std::chrono::high_resolution_clock::now();
        tracker_ptr->CalculateOptimization(iteration, corr_iteration,
                                          update_iteration);
        execution_times->calculate_optimization += ElapsedTime(begin_time);

        // Visualize pose update
        int update_save_idx =
            corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
        tracker_ptr->VisualizeOptimization(update_save_idx);
      }
    }
    // std::cout << "inner: " << region_modality_ptr->learning_rate() << std::endl;
    region_modality_ptr->set_learning_rate(learning_rate_);
    // std::cout << "outer: " << region_modality_ptr->learning_rate() << std::endl;
  };

  update();

  // 启用全局搜索
  if(global_search_){

    float errMin = tracker_ptr->CalculateError(alpha_);

    // 第0,1帧不判断前n帧，因为第1帧优化结束后才可以记录theta
    if(iteration > 1){
      float errT = tracker_ptr->GetMedianOfLastNError(errT_num_);
      // std::cout << "errT: " << errT << std::endl;
      
      // err大于阈值则全局搜索
      if(errMin > errT){
        // std::cout << "Miss!" << std::endl;

        auto &body_ptr{tracker_ptr->body_ptrs()[0]};

        // 复制物体位姿
        icg::Transform3fA dpose0 = body_ptr->body2world_pose();
        // 当前匹配错误最小位姿
        icg::Transform3fA dpose = dpose0;
        
        // 记录初始旋转与位移
        const auto R0 = dpose0.rotation();
        const auto T0 = dpose0.translation();
        // std::cout << "R0: " << R0.matrix() << std::endl;
        // 搜索范围
        float	thetaT = tracker_ptr->GetMedianOfLastNTheta(5);
        // std::cout << "thetaT: " << thetaT << std::endl;
        // 采样的数量
        const int N = int(thetaT / (CV_PI / 12) + 0.5f) | 1;
        // std::cout << "N: " << N << std::endl;
        // 每份采样的变化量
        const float dc = thetaT / N;
        // std::cout << "dc: " << dc << std::endl;
        // 细分3倍
        // const int subDiv = 3;
        // const int subRegionSize = (N * 2 * 2 + 1) * subDiv;
        // RegionTrajectory traj(cv::Size(subRegionSize, subRegionSize), dc / subDiv);

        cv::Mat1b label = cv::Mat1b::zeros(2 * N + 1, 2 * N + 1);

        struct DSeed
        {
          cv::Point coord;
          //bool  isLocalMinima;
        };

        float inner_learning_rate = 1.30f;

        // 网格遍历搜索
        bool scan_grid = false;
        // 从近到远搜索
        bool near_to_far_grid = true;
        // 路径提前结束
        bool bfs_grid = false;

        if(scan_grid){
          // while循环队列bfs
          bool flag = false;
          for(int i=0; i<2 * N + 1; i++){
            for(int j=0; j<2 * N + 1; j++){
              // std::cout << "i, j: " << i << ", " << j << std::endl;
              DSeed curSeed = {cv::Point(i,j)};

              // 相对于当前旋转的旋转变化
              auto dR = icg::theta2OutofplaneRotation(float(curSeed.coord.x - N) * dc, float(curSeed.coord.y - N) * dc);

              // 复制一个初始位姿
              auto dposex = dpose;

              Eigen::Matrix3f dR_eigen;
              cv::cv2eigen(dR, dR_eigen); 
              // 得到新搜索起点姿态
              dposex.linear() = dR_eigen * R0;
              body_ptr->set_body2world_pose(dposex);

              // cv::Point2f start = icg::dir2Theta(icg::viewDirFromR(dposex.R * R0.t()));

              // // Transform translation to Eigen
              // icg::Transform3fA body2world_pose;
              // Eigen::Vector3f eigen_translation;
              // cv::cv2eigen(translation, eigen_translation);
              // body2world_pose.translation() = eigen_translation;

              // // Transform rotation to Eigen
              // cv::Mat rotation_matrix;
              // Eigen::Matrix3f eigen_rotation;
              // cv::Rodrigues(rotation, rotation_matrix);
              // cv::cv2eigen(rotation_matrix, eigen_rotation);
              // body2world_pose.linear() = eigen_rotation;

              // body_ptr_->set_body2world_pose(body2world_pose);

              // std::cout << "重新迭代" << std::endl;s
              
              // 用新参数优化
              updateByPara(tracker_ptr->n_corr_iterations(), tracker_ptr->n_update_iterations(), inner_learning_rate);
              dposex = body_ptr->body2world_pose();

              // 计算轮廓error
              float err = tracker_ptr->CalculateError(alpha_);
              // error小于阈值则提前结束while
              // 找最小错误
              if (err < errMin)
              {
                errMin = err;
                dpose = dposex;
              }
              // 最小错误大于前N中值提前结束搜索
              if (errMin < errT){
                // std::cout << "提前结束" << std::endl;
                flag = true;
                break;
              }

                
            }
            if(flag) break;
          }

          // 重新用小优化细化姿态
          update();

          

        }else if(near_to_far_grid){

          std::vector<DSeed> seed_vec;
          for(int i=0; i<2 * N + 1; i++){
            for(int j=0; j<2 * N + 1; j++){
              if(i==N && j==N) continue; // 不搜索原旋转
              seed_vec.push_back({cv::Point(i,j)});
            }
          }

          sort(seed_vec.begin(), seed_vec.end(), [&](auto& seed1, auto& seed2){
            return abs(seed1.coord.x - N) + abs(seed1.coord.y - N) < abs(seed2.coord.x - N) + abs(seed2.coord.y - N);
          });
          
          // std::cout << "N: " << N <<std::endl;
          for(auto& curSeed: seed_vec){
            // std::cout << "x, y: " << curSeed.coord.x << ", " << curSeed.coord.y << std::endl;

            // 相对于当前旋转的旋转变化
            auto dR = icg::theta2OutofplaneRotation(float(curSeed.coord.x - N) * dc, float(curSeed.coord.y - N) * dc);
            
            // 复制一个当前匹配错误最小的位姿
            auto dposex = dpose;
            // std::cout << "old dposex: " << dposex.matrix() << std::endl;
            Eigen::Matrix3f dR_eigen;
            cv::cv2eigen(dR, dR_eigen); 
            // std::cout << "dR: " << dR_eigen.matrix() << std::endl;

            // T取当前最小匹配错误位姿的T,R取初始R0按搜索dR变化后的R,得到新搜索起点姿态
            dposex.linear() = dR_eigen * R0;

            // dposex.translation() = T0; // T恢复初始位姿T0
            // std::cout << "dpose: " << dpose.matrix() <<std::endl;
            // std::cout << "R0: " << R0 <<std::endl;
            // std::cout << "T0: " << T0 <<std::endl;
            // std::cout << "dposex.translation(): " << dposex.translation() <<std::endl;

            // std::cout << "new dposex: " << dposex.matrix() << std::endl;
            body_ptr->set_body2world_pose(dposex);

            // std::cout << "old dposex: " << dposex.matrix() <<std::endl;
            // std::cout << "重新迭代" << std::endl;

            updateByPara(tracker_ptr->n_corr_iterations(), tracker_ptr->n_update_iterations(), inner_learning_rate_);
            // update();
            // update();
            // update();
            dposex = body_ptr->body2world_pose();
            // std::cout << "new dposex: " << dposex.matrix() <<std::endl;
            
            // 计算轮廓error
            float err = tracker_ptr->CalculateError(alpha_);
            // error小于阈值则提前结束while
            // 找最小错误
            if (err < errMin)
            {
              errMin = err;
              dpose = dposex;
            }
            // 最小错误大于前N中值提前结束搜索
            if (errMin < errT){
              // std::cout << "提前结束" << std::endl;
              break;
            }
          }

          // 若所有样本均不符合阈值，则取匹配错误最小位姿重新迭代
          body_ptr->set_body2world_pose(dpose);
          // 重新用小优化细化姿态
          update();

        }else if(bfs_grid){
          // while循环队列bfs

          // 获得一个新rotation替换物体旋转，重新迭代(用更快的收敛参数)，计算轮廓error

          // error小于阈值则提前结束while

          // 重新用小优化细化姿态
        }
      }

    }
    
    if(iteration > 0){
      float	theta = tracker_ptr->CalculateTheta();
      // 记录当前帧与前一帧的旋转差
      tracker_ptr->RecordTheta(theta);
      // std::cout << "theta: " << theta << std::endl;
      // float	thetaT = tracker_ptr->GetMedianOfLastNTheta(5);
      // std::cout << "thetaT: " << thetaT << std::endl;
    }
    
    
    // 记录当前帧的匹配错误
    tracker_ptr->RecordError(errMin);
    
  }

  

  
  // 记录当前帧的位姿
  // tracker_ptr->RecordLastPose();

  // Calculate results
  begin_time = std::chrono::high_resolution_clock::now();
  tracker_ptr->CalculateResults(iteration);
  execution_times->calculate_results = ElapsedTime(begin_time);

  // Visualize results and update viewers
  tracker_ptr->VisualizeResults(iteration);
  if (visualize_tracking_ || save_images_)
    tracker_ptr->UpdateViewers(iteration);

  // execution_times->complete_cycle =
  //     execution_times->calculate_correspondences +
  //     execution_times->calculate_gradient_and_hessian +
  //     execution_times->calculate_optimization +
  //     execution_times->calculate_results;

  execution_times->complete_cycle = ElapsedTime(all_begin_time);
}

void YCBEvaluator::UpdateCameras(
    const std::shared_ptr<icg::Tracker> &tracker_ptr, int load_index) const {
  if (use_region_modality_) {
    auto color_camera{std::static_pointer_cast<icg::LoaderColorCamera>(
        tracker_ptr->main_camera_ptrs()[0])};
    color_camera->set_load_index(load_index);
    color_camera->SetUp();
  }
  if (!use_region_modality_ && use_depth_modality_) {
    auto depth_camera{std::static_pointer_cast<icg::LoaderDepthCamera>(
        tracker_ptr->main_camera_ptrs()[0])};
    depth_camera->set_load_index(load_index);
    depth_camera->SetUp();
  }
  if (use_region_modality_ &&
      (use_depth_modality_ || measure_occlusions_region_)) {
    auto depth_camera{std::static_pointer_cast<icg::LoaderDepthCamera>(
        tracker_ptr->main_camera_ptrs()[1])};
    depth_camera->set_load_index(load_index);
    depth_camera->SetUp();
  }
}

YCBEvaluator::Result YCBEvaluator::CalculateAverageBodyResult(
    const std::vector<std::string> &body_names) const {
  size_t n_results = 0;
  std::vector<Result> result_sums;
  for (const auto &result : results_) {
    if (std::find(begin(body_names), end(body_names), result.body_name) !=
        end(body_names)) {
      n_results += result.frame_results.size();
      result_sums.push_back(SumResults(result.frame_results));
    }
  }
  return DivideResult(SumResults(result_sums), n_results);
}

YCBEvaluator::Result YCBEvaluator::CalculateAverageSequenceResult(
    const std::vector<std::string> &sequence_names) const {
  size_t n_results = 0;
  std::vector<Result> result_sums;
  for (const auto &result : results_) {
    if (std::find(begin(sequence_names), end(sequence_names),
                  result.sequence_name) != end(sequence_names)) {
      n_results += result.frame_results.size();
      result_sums.push_back(SumResults(result.frame_results));
    }
  }
  return DivideResult(SumResults(result_sums), n_results);
}

YCBEvaluator::Result YCBEvaluator::CalculateAverageBodyResult2(
    const std::vector<std::string> &body_names) const {
  size_t n_results = 0;
  std::vector<Result> result_sums;
  for (const auto &result : results_) {
    if (std::find(begin(body_names), end(body_names), result.body_name) !=
        end(body_names)) {
      n_results += result.stable_results.size();
      result_sums.push_back(SumResults(result.stable_results));
    }
  }
  return DivideResult(SumResults(result_sums), n_results);
}

YCBEvaluator::Result YCBEvaluator::CalculateAverageSequenceResult2(
    const std::vector<std::string> &sequence_names) const {
  size_t n_results = 0;
  std::vector<Result> result_sums;
  for (const auto &result : results_) {
    if (std::find(begin(sequence_names), end(sequence_names),
                  result.sequence_name) != end(sequence_names)) {
      n_results += result.stable_results.size();
      result_sums.push_back(SumResults(result.stable_results));
    }
  }
  return DivideResult(SumResults(result_sums), n_results);
}

YCBEvaluator::Result YCBEvaluator::CalculateAverageResult(
    const std::vector<Result> &results) {
  return DivideResult(SumResults(results), results.size());
}

YCBEvaluator::Result YCBEvaluator::SumResults(
    const std::vector<Result> &results) {
  Result sum;
  for (const auto &result : results) {
    sum.add_auc += result.add_auc;
    sum.adds_auc += result.adds_auc;
    std::transform(begin(result.add_curve), end(result.add_curve),
                   begin(sum.add_curve), begin(sum.add_curve),
                   [](float a, float b) { return a + b; });
    std::transform(begin(result.adds_curve), end(result.adds_curve),
                   begin(sum.adds_curve), begin(sum.adds_curve),
                   [](float a, float b) { return a + b; });
    sum.execution_times.start_modalities +=
        result.execution_times.start_modalities;
    sum.execution_times.calculate_correspondences +=
        result.execution_times.calculate_correspondences;
    sum.execution_times.calculate_gradient_and_hessian +=
        result.execution_times.calculate_gradient_and_hessian;
    sum.execution_times.calculate_optimization +=
        result.execution_times.calculate_optimization;
    sum.execution_times.calculate_results +=
        result.execution_times.calculate_results;
    sum.execution_times.complete_cycle += result.execution_times.complete_cycle;
  }
  return sum;
}

YCBEvaluator::Result YCBEvaluator::DivideResult(const Result &result,
                                                size_t n) {
  Result divided;
  divided.add_auc = result.add_auc / float(n);
  divided.adds_auc = result.adds_auc / float(n);
  std::transform(begin(result.add_curve), end(result.add_curve),
                 begin(divided.add_curve),
                 [&](float a) { return a / float(n); });
  std::transform(begin(result.adds_curve), end(result.adds_curve),
                 begin(divided.adds_curve),
                 [&](float a) { return a / float(n); });
  divided.execution_times.start_modalities =
      result.execution_times.start_modalities / float(n);
  divided.execution_times.calculate_correspondences =
      result.execution_times.calculate_correspondences / float(n);
  divided.execution_times.calculate_gradient_and_hessian =
      result.execution_times.calculate_gradient_and_hessian / float(n);
  divided.execution_times.calculate_optimization =
      result.execution_times.calculate_optimization / float(n);
  divided.execution_times.calculate_results =
      result.execution_times.calculate_results / float(n);
  divided.execution_times.complete_cycle =
      result.execution_times.complete_cycle / float(n);
  return divided;
}


void YCBEvaluator::CalculatePoseResults(
    const std::shared_ptr<icg::Tracker> tracker_ptr,
    const std::string &body_name, const icg::Transform3fA &gt_body2world_pose,
    Result *result, Result *stable_result) {
  const auto &vertices{body2reduced_vertice_map_.at(body_name)};
  const auto &kdtree_index{body2kdtree_ptr_map_.at(body_name)->index};

  // Calculate pose error
  icg::Transform3fA body2world_pose;
  
  for (const auto &body_ptr : tracker_ptr->body_ptrs()) {
    if (body_ptr->name() == body_name)
      body2world_pose = body_ptr->body2world_pose();
  }

  icg::Transform3fA last_pose = tracker_ptr->body2world_pose_map_[body_name];

  icg::Transform3fA delta_pose{body2world_pose.inverse() * gt_body2world_pose};
  // 此处用上一帧pose替换gt
  // icg::Transform3fA delta_pose{body2world_pose.inverse() * last_pose};

  // 灰色预测结果位姿
  icg::Transform3fA predict_Pose;
  // 卡尔曼结果位姿
  icg::Transform3fA kalman_Pose = body2world_pose;
  // 最终位姿
  icg::Transform3fA result_pose = body2world_pose;

  // 预测位姿相对本帧的变化值占上下界直接的百分比
  float deltaPercent = 0;

  //启用灰色预测  
  if(use_gray_){
    //灰色预测样本数
    int sampleNum = sample_num_;
    std::vector<vector<double>> data;
    
    auto& vec = tracker_ptr->body2world_pose_map_vector_[body_name];
    int v_size = vec.size();
    // sampleNum = min(5, v_size);
    
    // // 第一帧释放卡尔曼指针
    // if(v_size == 0){
    //   delete kalman;
    //   kalman = nullptr;
    // }

    if(v_size >= sampleNum){
      // 将sampleNum个样本的位姿矩阵转换为7维位姿
      for(int i=sampleNum; i>0; i--){
        auto& trans = vec[v_size - i];
        Eigen::Quaternionf q(trans.rotation());
        data.push_back({
          trans.translation().x(),
          trans.translation().y(),
          trans.translation().z(),
          q.w(),
          q.x(),
          q.y(),
          q.z()
        });
      }
      // std::cout<<"gray data: {"<<std::endl;
      // for(auto& v: data){
      //   std::cout<<"{"<<std::endl;
      //   for(auto& d: v){
      //     std::cout<<d<<" ,";
      //   }
      //   std::cout<<"},"<<std::endl;
      // }
      // std::cout<<"};"<<std::endl;

      GM11 gm(data, sampleNum);
      gm.run();

      // std::cout<<"gray predictPose: "<<std::endl;
      // for(auto d: gm.predictPose){
      //   std::cout<<d<<" ";
      // }
      // std::cout<<std::endl;

      Eigen::Translation3f translation{gm.predictPose[0], gm.predictPose[1], gm.predictPose[2]};
      Eigen::Quaternionf quaternion{gm.predictPose[3], gm.predictPose[4], gm.predictPose[5], gm.predictPose[6]};
      predict_Pose = translation * quaternion;

      // 用预测帧与当前帧的变化程度来改变学习率
      icg::Transform3fA delta_pose_f2f{predict_Pose.inverse() * body2world_pose};

      // Calcualte add and adds error
      float add_error = 0.0f;
      float adds_error = 0.0f;
      size_t ret_index;
      float dist_sqrt;
      Eigen::Vector3f v;
      for (const auto &vertice : vertices) {
        v = delta_pose_f2f * vertice;
        add_error += (vertice - v).norm();
        kdtree_index->knnSearch(v.data(), 1, &ret_index, &dist_sqrt);
        adds_error += std::sqrt(dist_sqrt);
        // std::cout << "add_error: " << add_error << ", adds_error: " << adds_error  <<std::endl;
      }
      // std::cout << "add_error: " << add_error << ", adds_error: " << adds_error  <<std::endl;

      add_error /= vertices.size();
      adds_error /= vertices.size();

      // Calculate area under curve
      float add_auc = 1.0f - std::min(add_error / kThresholdMax, 1.0f);
      float adds_auc = 1.0f - std::min(adds_error / kThresholdMax, 1.0f);
      // std::cout << "add_auc: " << add_auc << ", adds_auc: " << adds_auc <<std::endl;



      for (auto &modality_ptr : tracker_ptr->modality_ptrs()) {
        
        std::string modality_name = body_name + "_region_modality";

        if(modality_name == modality_ptr->name()){
          // std::cout<<"body_name: "<<modality_ptr->name() <<std::endl;
          icg::Modality* m_ptr = modality_ptr.get();
          icg::RegionModality* region_modality_ptr = dynamic_cast<icg::RegionModality*>(m_ptr);

          float new_learning_rate = 0;
          float pi = 3.14159;
          int n = 64;
          deltaPercent = pow(cos(pi/2 * (0.993 - max(min(0.993f, adds_auc), 0.9f)) / (0.993 - 0.9)), n);
          // std::cout<< "deltaPercent: " << deltaPercent << std::endl;
          // std::cout<< "cos: " << cos(pi/2 * (0.993 - max(min(0.993f, adds_auc), 0.9f)) / (0.993 - 0.9)) << std::endl;
          // new_learning_rate = (-pow(cos(pi/2 * (0.993 - max(min(0.993f, adds_auc), 0.9f)) / (0.993 - 0.9)), n) + 1)  * 1.3;
          // std::cout<< "cos learning_rate: " << new_learning_rate << std::endl;
          new_learning_rate = (0.993 - max(min(0.993f, adds_auc), 0.9f)) / (0.993 - 0.9) * 1.3;
          // std::cout<< "old learning_rate: " << new_learning_rate << std::endl;
          // 灰色预测改变学习率
          region_modality_ptr->set_learning_rate(new_learning_rate);

          // std::cout<<"adds_auc: "<< adds_auc <<std::endl;
          // std::cout<<"learning_rate: "<< region_modality_ptr->learning_rate() <<std::endl;
        }
        
      }
    }
  }

  // 卡尔曼初始化
  if(use_kalman_){
    auto& vec = tracker_ptr->body2world_pose_map_vector_[body_name];
    // 第一帧释放卡尔曼指针，第二帧开始卡尔曼滤波，因为第一帧没有上一帧的结果
    if(vec.size() == 0){
      delete kalman;
      kalman = nullptr;
    }else if(vec.size() >= kalman_start_frame_){ // 卡尔曼滤波从kalman_start_frame_+1帧开始生效
      if(!kalman){
        kalman = new Kalman(7,7,7);
        Eigen::VectorXd x;
        Eigen::MatrixXd P0;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd R;
        Eigen::MatrixXd A;
        Eigen::MatrixXd B;
        Eigen::MatrixXd H;
        Eigen::VectorXd u_v;

        // 状态量
        x.resize(7);
        auto& trans = vec.back();
        Eigen::Quaternionf q(trans.rotation());
        x(0) = trans.translation().x();
        x(1) = trans.translation().y();
        x(2) = trans.translation().z();
        x(3) = q.w();
        x(4) = q.x();
        x(5) = q.y();
        x(6) = q.z();

        // 先验估计协方差
        P0.resize(7,7);
        // P0.setZero();
        // P0(0,0)=100;
        // P0(1,1)=2;
        P0.setIdentity();

        // 过程噪声协方差
        Q.resize(7,7);
        Q.setIdentity();
        Q *= kalman_Q_;
        // Q.setZero();

        //测量噪声协方差
        R.resize(7,7);
        R.setIdentity();
        R *= kalman_R_;
        // R.setZero();

        A.resize(7,7);
        A.setIdentity();

        B.resize(7,7);
        B.setIdentity();

        H.resize(7,7);
        H.setIdentity();

        // 输入矩阵
        u_v.resize(7,7);
        u_v.setIdentity();

        kalman->Init_Par(x,P0,R,Q,A,B,H,u_v);
      }else{
        // 更新估计值
        Eigen::VectorXd x;
        x.resize(7);
        // for(int i=0; i<7; i++){
        //   x(i) = gm.predictPose[i];
        // }

        // 更新测量值，用当前帧icg算法结果作为测量值
        Eigen::VectorXd z0;
        z0.resize(7);
        Eigen::Quaternionf quaternion{body2world_pose.rotation()};
        z0(0) = body2world_pose.translation().x();
        z0(1) = body2world_pose.translation().y();
        z0(2) = body2world_pose.translation().z();
        z0(3) = quaternion.w();
        z0(4) = quaternion.x();
        z0(5) = quaternion.y();
        z0(6) = quaternion.z();

        kalman->Run(x, z0, deltaPercent); // 运行卡尔曼滤波
        Eigen::VectorXd kalman_res = kalman->Result(); // 滤波结果

        // 用卡尔曼滤波结果更新位姿更新位姿
        for (const auto &body_ptr : tracker_ptr->body_ptrs()) {
          if (body_ptr->name() == body_name){
            // std::cout<<"卡尔曼滤波更新前，icg算法位姿:"<<std::endl;
            // std::cout<<body_ptr->body2world_pose().translation()<<std::endl;
            // std::cout<<body_ptr->body2world_pose().rotation()<<std::endl;
            Eigen::Translation3f translation{kalman_res(0), kalman_res(1), kalman_res(2)};
            Eigen::Quaternionf quaternion{kalman_res(3), kalman_res(4), kalman_res(5), kalman_res(6)};
            kalman_Pose = translation * quaternion;

            body_ptr->set_body2world_pose(kalman_Pose);
            // std::cout<<"卡尔曼滤波更新后，位姿:"<<std::endl;
            // std::cout<<body_ptr->body2world_pose().translation()<<std::endl;
            // std::cout<<body_ptr->body2world_pose().rotation()<<std::endl;

            // std::cout<<"gt位姿:"<<std::endl;
            // std::cout<<gt_body2world_pose.translation()<<std::endl;
            // std::cout<<gt_body2world_pose.rotation()<<std::endl;

            // 用卡尔曼滤波结果更新位姿，用于计算adds
            result_pose = kalman_Pose;
            // delta_pose = kalman_Pose.inverse() * last_pose;
            // delta_pose = kalman_Pose.inverse() * gt_body2world_pose;
          }
        }
      }
    }
  }

  // 计算gt帧间变化量，将每帧结果用gt替换
  // result_pose = gt_body2world_pose;
  // for (const auto &body_ptr : tracker_ptr->body_ptrs()) {
  //   if (body_ptr->name() == body_name){
  //     body_ptr->set_body2world_pose(result_pose);
  //   }
  // }

  // std::cout << "body_name: " << body_name <<std::endl;
  // std::cout << "map_vector: " << tracker_ptr->body2world_pose_map_vector_[body_name].size() <<std::endl;

  // std::cout << "result_pose: " << result_pose.rotation().matrix() <<std::endl;

  // 6D定值阈值不上报防抖
  bool use_value_switch = false;
  if(use_value_switch){
    float t_thresholds = 0.01; // 坐标阈值(m)
    float r_thresholds = 1 / 180 * 3.14159; // 欧拉角阈值(度)
    bool flag = true;
    if(abs(result_pose.translation().x() - last_pose.translation().x()) < t_thresholds){
      // std::cout << "x1: " << result_pose.translation().x() << "x2: " << last_pose.translation().x() <<std::endl;
      flag = false;
    }
    if(abs(result_pose.translation().y() - last_pose.translation().y()) < t_thresholds){
      // std::cout << "y1: " << result_pose.translation().y() << "y2: " << last_pose.translation().y() <<std::endl;
      flag = false;
    }
    if(abs(result_pose.translation().z() - last_pose.translation().z()) < t_thresholds){
      // std::cout << "z1: " << result_pose.translation().z() << "z2: " << last_pose.translation().z() <<std::endl;
      flag = false;
    }
    Eigen::Vector3f result_euler_angle = result_pose.rotation().eulerAngles(0,1,2);
    Eigen::Vector3f last_euler_angle = last_pose.rotation().eulerAngles(0,1,2);
    if(abs(result_euler_angle(0) - last_euler_angle(0)) < r_thresholds){
      std::cout << "result_euler_angle(0): " << result_euler_angle(0) << "last_euler_angle(0): " << last_pose.translation().z() <<std::endl;
      flag = false;
    }
    if(abs(result_euler_angle(1) - last_euler_angle(1)) < r_thresholds){
      flag = false;
    }
    if(abs(result_euler_angle(2) - last_euler_angle(2)) < r_thresholds){
      flag = false;
    }

    std::cout << "flag: " << flag <<std::endl;


    // 若变化过小则不改变位姿
    if(!flag)
      result_pose = last_pose;
  }

  // 用addsj作为阈值不上报的防抖法
  bool use_addsj_switch = false;
  if(use_addsj_switch){
    float addsj_thresholds = 0.98;
    Result* result_test = new Result();

    // delta_pose用last_pose计算，用于计算抖动ADDSJ
    delta_pose = result_pose.inverse() * last_pose;

    // Calcualte and adds error
    float adds_error = 0.0f;
    size_t ret_index;
    float dist_sqrt;
    Eigen::Vector3f v;
    for (const auto &vertice : vertices) {
      v = delta_pose * vertice;
      kdtree_index->knnSearch(v.data(), 1, &ret_index, &dist_sqrt);
      adds_error += std::sqrt(dist_sqrt);
    }

    adds_error /= vertices.size();
    // std::cout << "kNCurveValues: " << kNCurveValues <<std::endl;
  
    // Calculate curve (tracking loss distribution)
    std::fill(begin(result_test->adds_curve), end(result_test->adds_curve), 1.0f);
    for (size_t i = 0; i < kNCurveValues; ++i) {
      if (adds_error < thresholds_[i]) break;
      result_test->adds_curve[i] = 0.0f;
    }

    // Calculate area under curve
    result_test->adds_auc = 1.0f - std::min(adds_error / kThresholdMax, 1.0f);
    // std::cout << "adds_auc: " << result_test->adds_auc  <<std::endl;

    // 若变化过小则不改变位姿
    if(result_test->adds_auc > addsj_thresholds){
      result_pose = last_pose;
    }

  }

  // 用addsj作为阈值不上报的防抖法
  bool use_switch = false;
  if(use_switch){
    float addsj_thresholds = 0.90;
    Result* result_test = new Result();

    // delta_pose用last_pose计算，用于计算抖动ADDSJ
    delta_pose = result_pose.inverse() * last_pose;

    // Calcualte and adds error
    float adds_error = 0.0f;
    size_t ret_index;
    float dist_sqrt;
    Eigen::Vector3f v;
    for (const auto &vertice : vertices) {
      v = delta_pose * vertice;
      kdtree_index->knnSearch(v.data(), 1, &ret_index, &dist_sqrt);
      adds_error += std::sqrt(dist_sqrt);
    }

    adds_error /= vertices.size();
    // std::cout << "kNCurveValues: " << kNCurveValues <<std::endl;
  
    // Calculate curve (tracking loss distribution)
    std::fill(begin(result_test->adds_curve), end(result_test->adds_curve), 1.0f);
    for (size_t i = 0; i < kNCurveValues; ++i) {
      if (adds_error < thresholds_[i]) break;
      result_test->adds_curve[i] = 0.0f;
    }

    // Calculate area under curve
    result_test->adds_auc = 1.0f - std::min(adds_error / kThresholdMax, 1.0f);
    // std::cout << "adds_auc: " << result_test->adds_auc  <<std::endl;

    // 若变化过小则不改变位姿
    if(result_test->adds_auc > addsj_thresholds){
      result_pose = last_pose;
    }

  }

  // 用前n帧结果做平均的防抖法
  bool use_mean_pose = false;
  if(use_mean_pose){
    int n = 4;
    auto& vec = tracker_ptr->body2world_pose_map_vector_[body_name];
    auto mean_pose = result_pose;
    Eigen::Vector3f mean_euler_angle = result_pose.rotation().eulerAngles(0,1,2);

    for(int i=vec.size()-1; i>=vec.size()-n && i>=0; i--){
      auto& trans = vec[i];

      for(int x=0; x<4; x++){
        for(int y=0; y<4; y++){
          mean_pose(x,y) += trans(x,y);
        }
      }

      Eigen::Vector3f euler_angle = trans.rotation().eulerAngles(0,1,2);
      for(int x=0; x<3; x++){
        mean_euler_angle(x) += euler_angle(x);
      }
    }

    for(int x=0; x<4; x++){
      for(int y=0; y<4; y++){
        mean_pose(x,y) /= n+1;
      }
    }

    for(int x=0; x<3; x++){
      mean_euler_angle(x) /= n+1;
    }

    Eigen::Matrix3d mean_rotation;
    mean_rotation = Eigen::AngleAxisd(mean_euler_angle(0), Eigen::Vector3d::UnitX()) *
                    Eigen::AngleAxisd(mean_euler_angle(1), Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(mean_euler_angle(2), Eigen::Vector3d::UnitZ());

    // Eigen::Quaternionf mean_quaternion{mean_rotation};
    for(int x=0; x<3; x++){
      for(int y=0; y<3; y++){
        mean_pose(x,y) = mean_rotation(x,y);
      }
    }

    // mean_pose.rotation = mean_rotation;

    result_pose = mean_pose;

    // 用更新后的位姿替换物体位姿
    // for (const auto &body_ptr : tracker_ptr->body_ptrs()) {
    //   if (body_ptr->name() == body_name)
    //     body_ptr->set_body2world_pose(result_pose);
    // }
  }

  // bool print_ = false;
  if(false){
    auto& vec = tracker_ptr->body2world_pose_map_vector_[body_name];
    int v_size = vec.size();
    std::cout << "frame index: " << v_size+1  << " ----------------------------" <<std::endl;

    std::cout << "gt_body2world_pose: " <<std::endl;
    std::cout << gt_body2world_pose.translation() <<std::endl;
    // Eigen::Quaternionf quaternion_gt{gt_body2world_pose.rotation()};
    // std::cout << quaternion_gt.x() << ' ' << quaternion_gt.y() << ' ' << quaternion_gt.z() << ' ' << quaternion_gt.w() << std::endl;
    // 旋转矩阵转欧拉角ZYX
    Eigen::Vector3f euler_angle = gt_body2world_pose.rotation().eulerAngles(2,1,0);
    std::cout <<"euler_angle_ZYX: " <<  euler_angle.transpose()/3.14159*180 << std::endl;

    // std::cout << "last_pose: " << last_pose.matrix()  <<std::endl;

    std::cout << "result_pose: " <<std::endl;
    std::cout << result_pose.translation() <<std::endl;
    // Eigen::Quaternionf quaternion_result{result_pose.rotation()};
    // std::cout << quaternion_result.x() << ' ' << quaternion_result.y() << ' ' << quaternion_result.z() << ' ' << quaternion_result.w() << std::endl;
    // 旋转矩阵转欧拉角ZYX
    euler_angle = result_pose.rotation().eulerAngles(2,1,0);
    std::cout <<"euler_angle_ZYX: " <<  euler_angle.transpose()/3.14159*180 << std::endl;
  }
  

  // delta_pose用ground truth计算，用于计算准确adds
  delta_pose = result_pose.inverse() * gt_body2world_pose;

  // Calcualte add and adds error
  float add_error = 0.0f;
  float adds_error = 0.0f;
  size_t ret_index;
  float dist_sqrt;
  Eigen::Vector3f v;
  for (const auto &vertice : vertices) {
    v = delta_pose * vertice;
    add_error += (vertice - v).norm();
    kdtree_index->knnSearch(v.data(), 1, &ret_index, &dist_sqrt);
    adds_error += std::sqrt(dist_sqrt);
  }

  add_error /= vertices.size();
  adds_error /= vertices.size();

  // std::cout << "div: add_error: " << add_error << ", adds_error: " << adds_error  <<std::endl;
  // std::cout << "kNCurveValues: " << kNCurveValues <<std::endl;
 
  // Calculate curve (tracking loss distribution)
  std::fill(begin(result->add_curve), end(result->add_curve), 1.0f);
  for (size_t i = 0; i < kNCurveValues; ++i) {
    if (add_error < thresholds_[i]) break;
    result->add_curve[i] = 0.0f;
  }
  std::fill(begin(result->adds_curve), end(result->adds_curve), 1.0f);
  for (size_t i = 0; i < kNCurveValues; ++i) {
    if (adds_error < thresholds_[i]) break;
    result->adds_curve[i] = 0.0f;
  }
  // std::cout << "kThresholdMax: " << kThresholdMax <<std::endl;

  // Calculate area under curve
  result->add_auc = 1.0f - std::min(add_error / kThresholdMax, 1.0f);
  result->adds_auc = 1.0f - std::min(adds_error / kThresholdMax, 1.0f);
  // std::cout << "add_auc: " << result->add_auc << ", adds_auc: " << result->adds_auc  <<std::endl;

  // delta_pose用last_pose计算，用于计算抖动ADDSJ
  delta_pose = result_pose.inverse() * last_pose;
  // std::cout << "last_pose: " << last_pose.matrix()  <<std::endl;


  // Calcualte add and adds error
  add_error = 0.0f;
  adds_error = 0.0f;
  // size_t ret_index;
  // float dist_sqrt;
  // Eigen::Vector3f v;
  for (const auto &vertice : vertices) {
    v = delta_pose * vertice;
    add_error += (vertice - v).norm();
    kdtree_index->knnSearch(v.data(), 1, &ret_index, &dist_sqrt);
    adds_error += std::sqrt(dist_sqrt);
    // std::cout << "add_error: " << add_error << ", adds_error: " << adds_error  <<std::endl;
  }
  // std::cout << "add_error: " << add_error << ", adds_error: " << adds_error  <<std::endl;

  add_error /= vertices.size();
  adds_error /= vertices.size();

  // std::cout << "div: add_error: " << add_error << ", adds_error: " << adds_error  <<std::endl;
  // std::cout << "kNCurveValues: " << kNCurveValues <<std::endl;
 
  // Calculate curve (tracking loss distribution)
  std::fill(begin(stable_result->add_curve), end(stable_result->add_curve), 1.0f);
  for (size_t i = 0; i < kNCurveValues; ++i) {
    if (add_error < thresholds_[i]) break;
    stable_result->add_curve[i] = 0.0f;
  }
  std::fill(begin(stable_result->adds_curve), end(stable_result->adds_curve), 1.0f);
  for (size_t i = 0; i < kNCurveValues; ++i) {
    if (adds_error < thresholds_[i]) break;
    stable_result->adds_curve[i] = 0.0f;
  }
  // std::cout << "kThresholdMax: " << kThresholdMax <<std::endl;

  // Calculate area under curve
  stable_result->add_auc = 1.0f - std::min(add_error / kThresholdMax, 1.0f);
  stable_result->adds_auc = 1.0f - std::min(adds_error / kThresholdMax, 1.0f);
  // std::cout << "add_auc: " << stable_result->add_auc << ", adds_auc: " << stable_result->adds_auc  <<std::endl;
}

bool YCBEvaluator::LoadGTPoses(
    const std::string &sequence_name, const std::string &body_name,
    std::vector<icg::Transform3fA> *gt_body2world_poses) const {
  // Open poses file
  std::filesystem::path path{dataset_directory_ / "poses" /
                             (body_name + ".txt")};
  std::ifstream ifs{path.string(), std::ios::binary};
  if (!ifs.is_open() || ifs.fail()) {
    ifs.close();
    std::cerr << "Could not open file stream1 " << path.string() << std::endl;
    return false;
  }

  // Define begin index, n_frames and keyframes
  const auto idx_begin =
      body2sequence2pose_begin_.at(body_name).at(sequence_name);
  const int n_frames = sequence2nframes_map_.at(sequence_name);
  const auto keyframes = sequence2keyframes_map_.at(sequence_name);

  // Skip poses of other sequences
  std::string parsed;
  for (int idx = 0; idx < idx_begin; ++idx) {
    std::getline(ifs, parsed);
  }

  // Read poses
  int i_keyframe = 0;
  gt_body2world_poses->clear();
  for (int idx = 1; idx <= n_frames && i_keyframe < keyframes.size(); ++idx) {
    if (idx == keyframes[i_keyframe]) {
      Eigen::Quaternionf quaternion;
      Eigen::Translation3f translation;
      std::getline(ifs, parsed, ' ');
      quaternion.w() = stof(parsed);
      std::getline(ifs, parsed, ' ');
      quaternion.x() = stof(parsed);
      std::getline(ifs, parsed, ' ');
      quaternion.y() = stof(parsed);
      std::getline(ifs, parsed, ' ');
      quaternion.z() = stof(parsed);
      std::getline(ifs, parsed, ' ');
      translation.x() = stof(parsed);
      std::getline(ifs, parsed, ' ');
      translation.y() = stof(parsed);
      std::getline(ifs, parsed);
      translation.z() = stof(parsed);
      quaternion.normalize();
      icg::Transform3fA pose{translation * quaternion};
      gt_body2world_poses->push_back(std::move(pose));
      i_keyframe++;
    } else {
      std::getline(ifs, parsed);
    }
  }
  return true;
}

bool YCBEvaluator::LoadMatlabGTPoses(
    const std::string &sequence_name, const std::string &body_name,
    std::vector<icg::Transform3fA> *gt_body2world_poses) const {
  // Open poses file
  std::filesystem::path path{external_directory_ / "poses" / "ground_truth" /
                             (sequence_name + "_" + body_name + ".txt")};
  std::ifstream ifs{path.string(), std::ios::binary};
  if (!ifs.is_open() || ifs.fail()) {
    ifs.close();
    std::cerr << "Could not open file stream2 " << path.string() << std::endl;
    return false;
  }

  // Read poses
  gt_body2world_poses->clear();
  std::string parsed;
  while (std::getline(ifs, parsed)) {
    std::stringstream ss{parsed};
    Eigen::Quaternionf quaternion;
    Eigen::Translation3f translation;
    std::getline(ss, parsed, ' ');
    quaternion.w() = stof(parsed);
    std::getline(ss, parsed, ' ');
    quaternion.x() = stof(parsed);
    std::getline(ss, parsed, ' ');
    quaternion.y() = stof(parsed);
    std::getline(ss, parsed, ' ');
    quaternion.z() = stof(parsed);
    std::getline(ss, parsed, ' ');
    translation.x() = stof(parsed);
    std::getline(ss, parsed, ' ');
    translation.y() = stof(parsed);
    std::getline(ss, parsed);
    translation.z() = stof(parsed);
    quaternion.normalize();
    icg::Transform3fA pose{translation * quaternion};
    gt_body2world_poses->push_back(std::move(pose));
    // std::cout<<"---------------------------------------------------------------------------------"<<std::endl;
    // std::cout<<"translation:"<<std::endl;
    // std::cout << translation.x() << " " << translation.y() << " " << translation.z() << std::endl;
    // std::cout<<"quaternion:"<<std::endl;
    // std::cout << quaternion.w() << " " << quaternion.x() << " "  << quaternion.y() << " "  << quaternion.z() << " " <<std::endl;
    // std::cout << quaternion.matrix() <<std::endl;
    // Eigen::Quaternionf q = Eigen::Quaternionf(quaternion.matrix());
    // std::cout << q.w() << " " << q.x() << " "  << q.y() << " "  << q.z() << " " <<std::endl;

  }
  return true;
}

bool YCBEvaluator::LoadDetectorPoses(
    const std::string &sequence_name, const std::string &body_name,
    std::vector<icg::Transform3fA> *detector_body2world_poses,
    std::vector<bool> *body_detected) const {
  // Open detector poses file
  std::filesystem::path path{external_directory_ / "poses" / detector_folder_ /
                             (sequence_name + "_" + body_name + ".txt")};
  std::ifstream ifs{path.string(), std::ios::binary};
  if (!ifs.is_open() || ifs.fail()) {
    ifs.close();
    std::cerr << "Could not open file stream3 " << path.string() << std::endl;
    return false;
  }

  // Read poses
  body_detected->clear();
  detector_body2world_poses->clear();
  std::string parsed;
  while (std::getline(ifs, parsed)) {
    std::stringstream ss{parsed};
    Eigen::Quaternionf quaternion;
    Eigen::Translation3f translation;
    std::getline(ss, parsed, ' ');
    quaternion.w() = stof(parsed);
    std::getline(ss, parsed, ' ');
    quaternion.x() = stof(parsed);
    std::getline(ss, parsed, ' ');
    quaternion.y() = stof(parsed);
    std::getline(ss, parsed, ' ');
    quaternion.z() = stof(parsed);
    std::getline(ss, parsed, ' ');
    translation.x() = stof(parsed);
    std::getline(ss, parsed, ' ');
    translation.y() = stof(parsed);
    std::getline(ss, parsed);
    translation.z() = stof(parsed);
    if (translation.vector().isZero()) {
      detector_body2world_poses->push_back(icg::Transform3fA::Identity());
      body_detected->push_back(false);
    } else {
      quaternion.normalize();
      icg::Transform3fA pose{translation * quaternion};
      detector_body2world_poses->push_back(std::move(pose));
      body_detected->push_back(true);
    }
  }
  return true;
}

void YCBEvaluator::VisualizeResult(const Result &result,
                                   const std::string &title) {
                                    
  std::cout << title << ": ";
  if (result.frame_index) std::cout << "frame " << result.frame_index << ": ";
  std::cout << "execution_time = " << result.execution_times.complete_cycle
            << " us, "
            << "add auc = " << result.add_auc << ", "
            << "adds auc = " << result.adds_auc << std::endl;
}

bool YCBEvaluator::CreateRunConfigurations() {
  run_configurations_.clear();
  if (run_sequentially_ && !model_occlusions_region_ &&
      !model_occlusions_depth_) {
    for (int sequence_id : sequence_ids_) {
      std::string sequence_name{SequenceIDToName(sequence_id)};
      for (const auto &body_name : evaluated_body_names_) {
        if (BodyExistsInSequence(sequence_name, body_name)) {
          RunConfiguration run_configuration;
          run_configuration.sequence_id = sequence_id;
          run_configuration.sequence_name = sequence_name;
          run_configuration.evaluated_body_names.push_back(body_name);
          run_configuration.tracked_body_names.push_back(body_name);
          run_configurations_.push_back(std::move(run_configuration));
        }
      }
    }
  } else {
    for (int sequence_id : sequence_ids_) {
      RunConfiguration run_configuration;
      run_configuration.sequence_id = sequence_id;
      run_configuration.sequence_name = SequenceIDToName(sequence_id);
      for (const auto &body_name : evaluated_body_names_) {
        if (BodyExistsInSequence(run_configuration.sequence_name, body_name))
          run_configuration.evaluated_body_names.push_back(body_name);
      }
      if (run_configuration.evaluated_body_names.empty()) continue;
      if (model_occlusions_region_ || model_occlusions_depth_) {
        if (!SequenceBodyNames(run_configuration.sequence_name,
                               &run_configuration.tracked_body_names))
          return false;
      } else {
        run_configuration.tracked_body_names =
            run_configuration.evaluated_body_names;
      }
      run_configurations_.push_back(std::move(run_configuration));
    }
  }
  return true;
}

void YCBEvaluator::AssambleTrackedBodyNames() {
  tracked_body_names_.clear();
  for (const auto &run_configuration : run_configurations_) {
    for (const auto &tracked_body_name : run_configuration.tracked_body_names) {
      if (std::find(begin(tracked_body_names_), end(tracked_body_names_),
                    tracked_body_name) == end(tracked_body_names_))
        tracked_body_names_.push_back(tracked_body_name);
    }
  }
}

bool YCBEvaluator::LoadBodies() {
  bool success = true;
  body2body_ptr_map_.clear();
#pragma omp parallel for
  for (int i = 0; i < tracked_body_names_.size(); ++i) {
    auto &body_name{tracked_body_names_[i]};
    std::filesystem::path directory{dataset_directory_ / "models" / body_name};
    auto body_ptr{
        std::make_shared<icg::Body>(body_name, directory / "textured.obj", 1.0f,
                                    true, true, icg::Transform3fA::Identity())};
    if (!body_ptr->SetUp()) {
      success = false;
      continue;
    }
#pragma omp critical
    body2body_ptr_map_.insert({body_name, std::move(body_ptr)});

    const auto &vertices{body2body_ptr_map_[body_name]->vertices()};
  }
  return success;
}

bool YCBEvaluator::GenerateModels() {
  std::filesystem::path directory{external_directory_ / "models"};
  std::filesystem::create_directory(directory);
  for (auto body_name : tracked_body_names_) {
    if (use_region_modality_) {
      auto model_ptr{std::make_shared<icg::RegionModel>(
          body_name + "_region_model", body2body_ptr_map_[body_name],
          directory / (body_name + "_region_model.bin"), 0.8f, 4, 500, 0.05f,
          0.002f, false, 2000)};
      region_model_setter_(model_ptr);
      if (!model_ptr->SetUp()) return false;
      body2region_model_ptr_map_.insert({body_name, std::move(model_ptr)});
    }
    if (use_depth_modality_) {
      auto model_ptr{std::make_shared<icg::DepthModel>(
          body_name + "_depth_model", body2body_ptr_map_[body_name],
          directory / (body_name + "_depth_model.bin"), 0.8f, 4, 500, 0.05f,
          0.002f, false, 2000)};
      depth_model_setter_(model_ptr);
      if (!model_ptr->SetUp()) return false;
      body2depth_model_ptr_map_.insert({body_name, std::move(model_ptr)});
    }
  }
  return true;
}

bool YCBEvaluator::LoadKeyframes() {
  bool success = true;
  sequence2keyframes_map_.clear();
#pragma omp parallel for
  for (int i = 0; i < sequence_ids_.size(); ++i) {
    std::cout << "imgid: " << i << std::endl;
    std::string sequence_name{SequenceIDToName(sequence_ids_[i])};

    // Open poses file
    std::filesystem::path path{dataset_directory_ / "image_sets" /
                               "keyframe.txt"};
    std::ifstream ifs{path.string(), std::ios::binary};
    if (!ifs.is_open() || ifs.fail()) {
      ifs.close();
      std::cerr << "Could not open file stream4 " << path.string() << std::endl;
      success = false;
      continue;
    }

    std::vector<int> keyframes;
    std::string parsed;
    while (std::getline(ifs, parsed, '/')) {
      if (parsed == sequence_name) {
        std::getline(ifs, parsed);
        keyframes.push_back(stoi(parsed));
      } else {
        std::getline(ifs, parsed);
      }
    }
#pragma omp critical
    sequence2keyframes_map_.insert({sequence_name, keyframes});
  }
  return success;
}

void YCBEvaluator::LoadNFrames() {
  sequence2nframes_map_.clear();
  int max_sequence_id =
      *std::max_element(begin(sequence_ids_), end(sequence_ids_));
#pragma omp parallel for
  for (int sequence_id = 0; sequence_id <= max_sequence_id; ++sequence_id) {
    std::string sequence_name{SequenceIDToName(sequence_id)};
    int nframes = NFramesInSequence(sequence_name);
#pragma omp critical
    sequence2nframes_map_.insert({sequence_name, nframes});
  }
}

void YCBEvaluator::LoadPoseBegin() {
  body2sequence2pose_begin_.clear();
#pragma omp parallel for
  for (int i = 0; i < tracked_body_names_.size(); ++i) {
    auto &body_name{tracked_body_names_[i]};
    std::map<std::string, int> sequence2pose_begin;
    for (int max_sequence_id : sequence_ids_) {
      int idx_begin = 0;
      for (int sequence_id = 0; sequence_id < max_sequence_id; ++sequence_id) {
        std::string sequence_name{SequenceIDToName(sequence_id)};
        if (BodyExistsInSequence(sequence_name, body_name))
          idx_begin += sequence2nframes_map_[sequence_name];
      }
      std::string max_sequence_name{SequenceIDToName(max_sequence_id)};
      sequence2pose_begin.insert({max_sequence_name, idx_begin});
    }
#pragma omp critical
    body2sequence2pose_begin_.insert({body_name, sequence2pose_begin});
  }
}

void YCBEvaluator::GenderateReducedVertices() {
#pragma omp parallel for
  for (int i = 0; i < tracked_body_names_.size(); ++i) {
    auto &body_name{tracked_body_names_[i]};
    std::cout<< "body_name: " << body_name << std::endl;
    const auto &vertices{body2body_ptr_map_[body_name]->vertices()};
    std::cout<< "vertices.size(): " << vertices.size() << std::endl;
    if (n_vertices_evaluation_ <= 0 ||
        n_vertices_evaluation_ >= vertices.size()) {
      body2reduced_vertice_map_[body_name] = vertices;
      continue;
    }

    std::mt19937 generator{7};
    if (use_random_seed_)
      generator.seed(unsigned(
          std::chrono::system_clock::now().time_since_epoch().count()));

    std::vector<Eigen::Vector3f> reduced_vertices(n_vertices_evaluation_);
    int n_vertices = vertices.size();
    for (auto &v : reduced_vertices) {
      int idx = int(generator() % n_vertices);
      v = vertices[idx];
    }
#pragma omp critical
    body2reduced_vertice_map_.insert({body_name, std::move(reduced_vertices)});
  }
}

void YCBEvaluator::GenerateKDTrees() {
#pragma omp parallel for
  for (int i = 0; i < tracked_body_names_.size(); ++i) {
    auto &body_name{tracked_body_names_[i]};
    const auto &vertices{body2body_ptr_map_[body_name]->vertices()};
    auto kdtree_ptr{std::make_unique<KDTreeVector3f>(3, vertices, 10)};
    kdtree_ptr->index->buildIndex();
#pragma omp critical
    body2kdtree_ptr_map_.insert({body_name, std::move(kdtree_ptr)});
  }
}

bool YCBEvaluator::BodyExistsInSequence(const std::string &sequence_name,
                                        const std::string &body_name) const {
  std::filesystem::path path{dataset_directory_ / "data" / sequence_name /
                             "000001-box.txt"};
  std::ifstream ifs{path.string(), std::ios::binary};
  if (!ifs.is_open() || ifs.fail()) {
    ifs.close();
    std::cerr << "Could not open file stream5 " << path.string() << std::endl;
    return false;
  }

  std::string parsed;
  while (std::getline(ifs, parsed, ' ')) {
    if (parsed == body_name) return true;
    std::getline(ifs, parsed);
  }
  return false;
}

bool YCBEvaluator::SequenceBodyNames(
    const std::string &sequence_name,
    std::vector<std::string> *sequence_body_names) const {
  std::filesystem::path path{dataset_directory_ / "data" / sequence_name /
                             "000001-box.txt"};
  std::ifstream ifs{path.string(), std::ios::binary};
  if (!ifs.is_open() || ifs.fail()) {
    ifs.close();
    std::cerr << "Could not open file stream6 " << path.string() << std::endl;
    return false;
  }

  sequence_body_names->clear();
  std::string parsed;
  while (std::getline(ifs, parsed, ' ')) {
    sequence_body_names->push_back(parsed);
    std::getline(ifs, parsed);
  }
  return true;
}

int YCBEvaluator::NFramesInSequence(const std::string &sequence_name) const {
  std::filesystem::path directory{dataset_directory_ / "data" / sequence_name};
  for (int i = 1; true; ++i) {
    int n_zeros = 6 - int(std::to_string(i).length());
    std::filesystem::path path{directory / (std::string(n_zeros, '0') +
                                            std::to_string(i) + "-box.txt")};
    if (!std::filesystem::exists(path)) return i - 1;
  }
}

std::string YCBEvaluator::SequenceIDToName(int sequence_id) const {
  int n_zeros = 4 - int(std::to_string(sequence_id).length());
  return std::string(n_zeros, '0') + std::to_string(sequence_id);
}

float YCBEvaluator::ElapsedTime(
    const std::chrono::high_resolution_clock::time_point &begin_time) {
  auto end_time{std::chrono::high_resolution_clock::now()};
  return float(std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                     begin_time)
                   .count());
}


//保存结果
void YCBEvaluator::SaveMyResults(std::filesystem::path path) {
  path += "/";
  // path += std::to_string(learning_rate_);
  // path += std::to_string(id_);
  path += "inner_learning_rate_:";
  path += std::to_string(inner_learning_rate_);
  // path += ",use_gray:";
  // path += std::to_string(use_gray_);
  // path += ",sample_num:";
  // path += std::to_string(sample_num_);
  // path += ",use_kalman:";
  // path += std::to_string(use_kalman_);
  // path += ",kalman_Q:";
  // path += std::to_string(kalman_Q_).substr(0,4);
  // path += ",kalman_R:";
  // path += std::to_string(kalman_R_).substr(0,4);
  // path += ",kalman_start_frame:";
  // path += std::to_string(kalman_start_frame_);
  path += ".txt";

  std::ofstream ofs{path};
  for (auto const &[body_name, result] : final_results_) {
    ofs << body_name << "," << result.add_auc << "," << result.adds_auc << ","
        << result.execution_times.complete_cycle << ","
        << result.execution_times.start_modalities << ","
        << result.execution_times.calculate_correspondences << ","
        << result.execution_times.calculate_gradient_and_hessian << ","
        << result.execution_times.calculate_optimization << ","
        << result.execution_times.calculate_results << std::endl;
  }
  ofs << use_gray_ << "," << use_kalman_ << "," << sample_num_ << "," << kalman_Q_ << "," << kalman_R_ << "," << kalman_start_frame_ << std::endl;

  ofs.flush();
  ofs.close();
}

void YCBEvaluator::set_learning_rate(float learning_rate){
  learning_rate_ = learning_rate;
}

void YCBEvaluator::set_use_gray(bool use_gray){
  use_gray_ = use_gray;
}

void YCBEvaluator::set_use_kalman(bool use_kalman){
  use_kalman_ = use_kalman;
}

void YCBEvaluator::set_sample_num(int sample_num){
  sample_num_ = sample_num;
}

void YCBEvaluator::set_kalman_Q(double kalman_Q){
  kalman_Q_ = kalman_Q;
}

void YCBEvaluator::set_kalman_R(double kalman_R){
  kalman_R_ = kalman_R;
}

void YCBEvaluator::set_kalman_start_frame(int kalman_start_frame){
  kalman_start_frame_ = kalman_start_frame;
}

void YCBEvaluator::set_id(int id){
  id_ = id;
}

void YCBEvaluator::set_errT_num(int errT_num){
  errT_num_ = errT_num;
}