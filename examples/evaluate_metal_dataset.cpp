// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)
// #include <Python.h>
#include "metal_evaluator.h"
#include <fstream>
#include <cmath>
// #include "eval_rbot.h"

// evaluator的运行参数整合
class RunPara{
public:
  int id;
	float learning_rate;
  bool use_gray;
  int sample_num;
  bool use_kalman;
  double kalman_Q;
  double kalman_R;
  int kalman_start_frame;
  std::string time;

  // 全局搜索
  bool global_search;
  // errT自适应样本数
  int errT_num;
  float alpha;
  float inner_learning_rate;
};

void run(RunPara);

int main() {
  time_t rawtime;
  struct tm * timeinfo;
  time ( &rawtime );
  timeinfo = localtime ( &rawtime );

  // for(int i=130; i<=130; i++){
  //   run(0.01 * i, asctime (timeinfo));
  // }

  RunPara runPara;
  runPara.id = 1;
  runPara.learning_rate = 1.3f;

  // 灰色预测参数
  runPara.use_gray = false;
  runPara.sample_num = 8;
  
  // 卡尔曼滤波参数
  runPara.use_kalman = false;
  runPara.kalman_Q = 1;
  runPara.kalman_R = 1;
  runPara.kalman_start_frame = 1;
  runPara.time = asctime (timeinfo);

  // 全局搜索参数
  runPara.global_search = true;
  runPara.errT_num = 15;
  runPara.alpha = 0.085;
  runPara.inner_learning_rate = 1.3;
  for(float i=0.92; i>=0.01; i-=0.01){
    runPara.inner_learning_rate = i;
    run(runPara);
  }
  run(runPara);
}


void run(RunPara runPara){

  std::string result_path = "./record/" + runPara.time;
  if(!std::filesystem::exists(result_path))
    std::filesystem::create_directory(result_path);

  const std::filesystem::path dataset_directory{"./metal/"};
  // const std::filesystem::path body_directory{"./metal/"};
  // std::vector<std::string> body_names{"dizuo", "da_chilun"};
	// std::vector<std::string> body_names{"dizuo", "da_chilun", "xiao_chilun", "da_quan_hou", "xiao_quan_hou", "gaizi", "da_gaizi_hou", "xiao_gaizi_hou"};
  std::vector<std::string> body_names{"dizuo", "da_chilun", "xiao_chilun", "gaizi"};

  constexpr bool kUseColorViewer = false;
  constexpr bool kUseDepthViewer = false;
  constexpr bool kMeasureOcclusions = true;
  constexpr bool kModelOcclusions = false;
  constexpr bool kVisualizePoseResult = false;

	const std::filesystem::path sequence_path{"./metal"};
  const std::filesystem::path color_camera_metafile_path{ dataset_directory / "ground_truth"};
  const std::filesystem::path depth_camera_metafile_path{ dataset_directory / "ground_truth"};

  std::vector<int> sequence_ids{1};
  // std::vector<int> sequence_ids{1, 2, 3, 4};

  // Run experiments
  MetalEvaluator evaluator{"evaluator", dataset_directory,
                         sequence_ids, body_names};

  // 自定义参数
  evaluator.set_id(runPara.id); //设置学习率
  evaluator.set_learning_rate(runPara.learning_rate); //设置学习率
  // 灰色预测参数
  evaluator.set_use_gray(runPara.use_gray); //true: 启用灰色预测，false: 不启用
  evaluator.set_sample_num(runPara.sample_num); //灰色样本数
  // 卡尔曼滤波参数
  evaluator.set_use_kalman(runPara.use_kalman); //true: 启用卡尔曼滤波，false: 不启用
  evaluator.set_kalman_Q(runPara.kalman_Q); // 卡尔曼滤波Q：过程噪声协方差
  evaluator.set_kalman_R(runPara.kalman_R); // 卡尔曼滤波R：测量噪声协方差
  evaluator.set_kalman_start_frame(runPara.kalman_start_frame); // 卡尔曼滤波从n+1帧执行开始执行，参数不为0，因为第1帧不能滤波
  // 网格搜索参数
  evaluator.set_errT_num(runPara.errT_num); 
  evaluator.alpha_ = runPara.alpha; 
  evaluator.global_search_ = runPara.global_search; 
  evaluator.inner_learning_rate_ = runPara.inner_learning_rate;

  // evaluator.set_region_modality_setter([&](auto r) {
  //   r->set_n_lines(200); // 对应线个数
  //   r->set_min_continuous_distance(3.0f);
  //   r->set_function_length(8);  //预计算数
  //   r->set_distribution_length(12); //概率分布的离散数
  //   r->set_function_amplitude(0.43f); //振幅参数 0.43f
  //   r->set_function_slope(0.5f); //斜率参数
  //   r->set_learning_rate(runPara.learning_rate); //总学习率1.3f
  //   r->set_scales({7, 4, 2}); //对应线计算标度
  //   r->set_standard_deviations({25.0f, 15.0f, 10.0f}); //标准差
  //   r->set_n_histogram_bins(16); //直方图个数
  //   r->set_learning_rate_f(0.2f);//前景学习率0.2f
  //   r->set_learning_rate_b(0.2f);//后景学习率0.2f
  //   r->set_unconsidered_line_length(0.5f);
  //   r->set_max_considered_line_length(20.0f);
  //   r->set_measured_depth_offset_radius(0.01f);
  //   r->set_measured_occlusion_radius(0.01f); //遮挡半径
  //   r->set_measured_occlusion_threshold(0.03f); //遮挡阈值
  // });
  // evaluator.set_depth_modality_setter([&](auto d) {
  //   d->set_n_points(200);
  //   d->set_stride_length(0.005f); //对应点采样间隔 grid 0.005f
  //   d->set_considered_distances({0.07f, 0.05f, 0.04f}); // rt: radius threshold, 对应点的半径
  //   d->set_standard_deviations({0.05f, 0.03f, 0.02f});//标准差
  //   d->set_measured_depth_offset_radius(0.01f);
  //   d->set_measured_occlusion_radius(0.01f); //遮挡半径
  //   d->set_measured_occlusion_threshold(0.03f); //遮挡阈值
  // });
  // evaluator.set_optimizer_setter([&](auto o) {
  //   o->set_tikhonov_parameter_rotation(1000.0f); //旋转正则化参数Laputa r
  //   o->set_tikhonov_parameter_translation(30000.0f); //平移正则化参数Laputa t
  // });
  // evaluator.set_tracker_setter([&](auto t) {
  //   t->set_n_update_iterations(2);
  //   t->set_n_corr_iterations(5);
  // });
  // evaluator.set_evaluate_refinement(false);
  // evaluator.set_use_matlab_gt_poses(true);
  // evaluator.set_run_sequentially(true);
  // evaluator.set_use_random_seed(false);
  // evaluator.set_n_vertices_evaluation(1000);
  // evaluator.set_visualize_frame_results(false);
  // evaluator.set_visualize_tracking(false);
  // evaluator.set_use_region_modality(true);
  // evaluator.set_use_depth_modality(true);
  // evaluator.set_measure_occlusions_region(true);
  // evaluator.set_measure_occlusions_depth(true);
  // evaluator.set_model_occlusions_region(false);
  // evaluator.set_model_occlusions_depth(false);
  // evaluator.SetUp();
  // evaluator.Evaluate();
  //保存用时
  // evaluator.SaveResults(result_path);
  //保存adds
  // evaluator.SaveMyResults(result_path);

  evaluator.results_.clear();
  evaluator.final_results_.clear();


// #pragma omp parallel for 
  for (const auto body_name : body_names) {
    evaluator.tracked_body_names_ = {body_name};

    // Set up tracker and renderer geometry
    auto tracker_ptr{std::make_shared<icg::Tracker>("tracker")};
    auto renderer_geometry_ptr{
        std::make_shared<icg::RendererGeometry>("renderer_geometry")};

    // Set up camera
    auto color_camera_ptr{std::make_shared<icg::LoaderColorCamera>(
        "color_camera", color_camera_metafile_path / ("color_camera_" + body_name + ".yaml"))};
    auto depth_camera_ptr{std::make_shared<icg::LoaderDepthCamera>(
        "depth_camera", depth_camera_metafile_path / ("depth_camera_" + body_name + ".yaml"))};

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

    // Set up body
    std::filesystem::path metafile_path{dataset_directory / (body_name + ".yaml")};
    auto body_ptr{std::make_shared<icg::Body>(body_name, metafile_path)};
    renderer_geometry_ptr->AddBody(body_ptr);
    color_depth_renderer_ptr->AddReferencedBody(body_ptr);
    depth_depth_renderer_ptr->AddReferencedBody(body_ptr);

    // Set up detector
    std::filesystem::path detector_path{dataset_directory /
                                        (body_name + "_detector.yaml")};

    auto detector_ptr{std::make_shared<icg::StaticDetector>(
        body_name + "_detector", detector_path, body_ptr)};
    tracker_ptr->AddDetector(detector_ptr);

    // Set up models
    auto region_model_ptr{std::make_shared<icg::RegionModel>(
        body_name + "_region_model", body_ptr,
        dataset_directory / (body_name + "_region_model.bin"))};
    auto depth_model_ptr{std::make_shared<icg::DepthModel>(
        body_name + "_depth_model", body_ptr,
        dataset_directory / (body_name + "_depth_model.bin"))};

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

    tracker_ptr->set_n_update_iterations(2);
    tracker_ptr->set_n_corr_iterations(6);

    evaluator.body2body_ptr_map_.clear();
    evaluator.body2body_ptr_map_.insert({body_name, std::move(body_ptr)});

    evaluator.SetUp();

    evaluator.EvaluateTracker(tracker_ptr);

    // #pragma omp critical
    {
      evaluator.results_.insert(end(evaluator.results_), begin(evaluator.sequence_results),
                      end(evaluator.sequence_results));
      for (const auto &sequence_result : evaluator.sequence_results) {
        std::string title{sequence_result.sequence_name + ": " +
                          sequence_result.body_name};
        evaluator.VisualizeResult(sequence_result.average_result, title);
        evaluator.VisualizeResult(sequence_result.average_stable_result, title);
      }
    }
    
  }

  evaluator.CalculateAllResult();

  evaluator.SaveMyResults(result_path);

}