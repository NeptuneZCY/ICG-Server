// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)
// #include <Python.h>
#include "ycb_evaluator.h"
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
  runPara.kalman_R = 0.2;
  runPara.kalman_start_frame = 1;
  runPara.time = asctime (timeinfo);

  // 全局搜索参数
  runPara.global_search = false;
  runPara.errT_num = 15;
  runPara.alpha = 0.08;
  runPara.inner_learning_rate = 0.33;
  // for(int i=1; i<=200; i+=1){
  //   runPara.inner_learning_rate = i*0.01;
  //   run(runPara);
  // }
  run(runPara);
}


void run(RunPara runPara){

  std::string res_path = "./record/" + runPara.time;
  if(!std::filesystem::exists(res_path))
    std::filesystem::create_directory(res_path);

  // Directories
  std::filesystem::path dataset_directory{"./ycb/dataset/"};// ./ycb/dataset/
  std::filesystem::path external_directory{"./ycb/external/"};// ./../../dataset/ycb/dataset/
  std::filesystem::path result_path{res_path};

  // Dataset configuration
  std::vector<std::string> body_names{"002_master_chef_can",
                                      "003_cracker_box",
                                      "004_sugar_box",
                                      "005_tomato_soup_can",
                                      "006_mustard_bottle",
                                      "007_tuna_fish_can",
                                      "008_pudding_box",
                                      "009_gelatin_box",
                                      "010_potted_meat_can",
                                      "011_banana",
                                      "019_pitcher_base",
                                      "021_bleach_cleanser",
                                      "024_bowl",
                                      "025_mug",
                                      "035_power_drill",
                                      "036_wood_block",
                                      "037_scissors",
                                      "040_large_marker",
                                      "051_large_clamp",
                                      "052_extra_large_clamp",
                                      "061_foam_brick"};

  // std::vector<std::string> body_names{"dizuo",
  //                                   "da_chilun",
  //                                   "xiao_chilun",
  //                                   "gaizi"};

  std::vector<int> sequence_ids{48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
  // std::vector<int> sequence_ids{54};

  // Run experiments
  YCBEvaluator evaluator{"evaluator", dataset_directory, external_directory,
                         sequence_ids, body_names};

  // 自定义参数
  evaluator.set_id(runPara.id); //设置学习率
  evaluator.set_learning_rate(runPara.learning_rate); //设置学习率
  evaluator.photo_directory_ = result_path;
  evaluator.save_region_photo_ = false;
  evaluator.save_images_ = false;

  if(!std::filesystem::exists(evaluator.photo_directory_))
    std::filesystem::create_directory(evaluator.photo_directory_);

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

  evaluator.set_region_modality_setter([&](auto r) {
    r->set_n_lines(200); // 对应线个数
    r->set_min_continuous_distance(3.0f);
    r->set_function_length(8);  //预计算数
    r->set_distribution_length(12); //概率分布的离散数
    r->set_function_amplitude(0.43f); //振幅参数 0.43f
    r->set_function_slope(0.5f); //斜率参数
    r->set_learning_rate(runPara.learning_rate); //总学习率1.3f
    r->set_scales({7, 4, 2}); //对应线计算标度
    r->set_standard_deviations({25.0f, 15.0f, 10.0f}); //标准差
    r->set_n_histogram_bins(16); //直方图个数
    r->set_learning_rate_f(0.2f);//前景学习率0.2f
    r->set_learning_rate_b(0.2f);//后景学习率0.2f
    r->set_unconsidered_line_length(0.5f);
    r->set_max_considered_line_length(20.0f);
    r->set_measured_depth_offset_radius(0.01f);
    r->set_measured_occlusion_radius(0.01f); //遮挡半径
    r->set_measured_occlusion_threshold(0.03f); //遮挡阈值
  });
  evaluator.set_depth_modality_setter([&](auto d) {
    d->set_n_points(200);
    d->set_stride_length(0.005f); //对应点采样间隔 grid 0.005f
    d->set_considered_distances({0.07f, 0.05f, 0.04f}); // rt: radius threshold, 对应点的半径
    d->set_standard_deviations({0.05f, 0.03f, 0.02f});//标准差
    d->set_measured_depth_offset_radius(0.01f);
    d->set_measured_occlusion_radius(0.01f); //遮挡半径
    d->set_measured_occlusion_threshold(0.03f); //遮挡阈值
  });
  evaluator.set_optimizer_setter([&](auto o) {
    o->set_tikhonov_parameter_rotation(1000.0f); //旋转正则化参数Laputa r
    o->set_tikhonov_parameter_translation(30000.0f); //平移正则化参数Laputa t
  });
  evaluator.set_tracker_setter([&](auto t) {
    t->set_n_update_iterations(2);
    t->set_n_corr_iterations(3);
  });
  evaluator.set_evaluate_refinement(false);
  evaluator.set_use_matlab_gt_poses(true);
  evaluator.set_run_sequentially(true);
  evaluator.set_use_random_seed(false);
  evaluator.set_n_vertices_evaluation(1000);
  evaluator.set_visualize_frame_results(false);
  evaluator.set_visualize_tracking(false);
  evaluator.set_use_region_modality(true);
  evaluator.set_use_depth_modality(true);
  evaluator.set_measure_occlusions_region(true);
  evaluator.set_measure_occlusions_depth(true);
  evaluator.set_model_occlusions_region(false);
  evaluator.set_model_occlusions_depth(false);
  evaluator.SetUp();
  evaluator.Evaluate();
  std::cout << "frames: " << evaluator.all_frames_ << ", time: " << evaluator.all_time_ / 1000000.0 << "s" << std::endl;
  std::cout << "per frame: " << evaluator.all_time_ / 1000000.0 / evaluator.all_frames_ << "s" << std::endl;
  std::cout << "fps: " << evaluator.all_frames_ / evaluator.all_time_  * 1000000.0 << std::endl;
  //保存用时
  // evaluator.SaveResults(result_path);
  //保存adds
  evaluator.SaveMyResults(result_path);
}