// SPDX-License-Identifier: MIT
// Copyright (c) 2022 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef ICG_INCLUDE_ICG_COMMON_H_
#define ICG_INCLUDE_ICG_COMMON_H_

#include <filesystem/filesystem.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cmath>
#include <opencv2/core/eigen.hpp>


namespace icg {

// Commonly used types
typedef Eigen::Transform<float, 3, Eigen::Affine> Transform3fA;

// Commonly used constants
constexpr float kPi = 3.1415926535897f;

// Commonly used structs
struct Intrinsics {
  float fu, fv;
  float ppu, ppv;
  int width, height;
};

// Commonly used mathematical functions
template <typename T>
inline int sgn(T value) {
  if (value < T(0))
    return -1;
  else if (value > T(0))
    return 1;
  else
    return 0;
}

template <typename T>
inline float sgnf(T value) {
  if (value < T(0))
    return -1.0f;
  else if (value > T(0))
    return 1.0f;
  else
    return 0.0f;
}

template <typename T>
inline T square(T value) {
  return value * value;
}

inline int pow_int(int x, int p) {
  if (p == 0) return 1;
  if (p == 1) return x;
  return pow_int(x, p - 1) * x;
}

inline Eigen::Matrix3f Vector2Skewsymmetric(const Eigen::Vector3f &vector) {
  Eigen::Matrix3f skew_symmetric;
  skew_symmetric << 0.0f, -vector(2), vector(1), vector(2), 0.0f, -vector(0),
      -vector(1), vector(0), 0.0f;
  return skew_symmetric;
}

// Commonly used functions to compare paths
bool Equivalent(const std::filesystem::path &path1,
                const std::filesystem::path &path2);

// Commonly used functions to read and write value to txt file
void ReadValueFromTxt(std::ifstream &ifs, bool *value);
void ReadValueFromTxt(std::ifstream &ifs, int *value);
void ReadValueFromTxt(std::ifstream &ifs, float *value);
void ReadValueFromTxt(std::ifstream &ifs, std::string *value);
void ReadValueFromTxt(std::ifstream &ifs, Transform3fA *value);
void ReadValueFromTxt(std::ifstream &ifs, Eigen::MatrixXf *value);
void ReadValueFromTxt(std::ifstream &ifs, Intrinsics *value);
void ReadValueFromTxt(std::ifstream &ifs, std::filesystem::path *value);

void WriteValueToTxt(std::ofstream &ofs, const std::string &name, bool value);
void WriteValueToTxt(std::ofstream &ofs, const std::string &name, int value);
void WriteValueToTxt(std::ofstream &ofs, const std::string &name, float value);
void WriteValueToTxt(std::ofstream &ofs, const std::string &name,
                     const std::string &value);
void WriteValueToTxt(std::ofstream &ofs, const std::string &name,
                     const Transform3fA &value);
void WriteValueToTxt(std::ofstream &ofs, const std::string &name,
                     const Eigen::MatrixXf &value);
void WriteValueToTxt(std::ofstream &ofs, const std::string &name,
                     const Intrinsics &value);
void WriteValueToTxt(std::ofstream &ofs, const std::string &name,
                     const std::filesystem::path &value);

// Commonly used functions to open file storage for .yaml file
bool OpenYamlFileStorage(const std::filesystem::path &metafile_path,
                         cv::FileStorage *file_storage);

// Commonly used functions to read optional value from .yaml file
template <typename T>
void ReadValueFromYaml(const cv::FileStorage &file_storage,
                       const std::string &name, T *value) {
  file_storage[name] >> *value;
}
void ReadValueFromYaml(const cv::FileStorage &file_storage,
                       const std::string &name, std::filesystem::path *value);
void ReadValueFromYaml(const cv::FileStorage &file_storage,
                       const std::string &name, Intrinsics *value);
void ReadValueFromYaml(const cv::FileStorage &file_storage,
                       const std::string &name, Transform3fA *value);
void ReadValueFromYaml(const cv::FileStorage &file_storage,
                       const std::string &name, Eigen::MatrixXf *value);
void ReadValueFromYaml(const cv::FileStorage &file_storage,
                       const std::string &name,
                       std::vector<cv::Point3f> *value);

// Commonly used functions to read and evaluate value from .yaml file
template <typename T>
inline void ReadOptionalValueFromYaml(const cv::FileStorage &file_storage,
                                      const std::string &name, T *value) {
  if (!file_storage[name].empty()) ReadValueFromYaml(file_storage, name, value);
}

template <typename T>
inline bool ReadRequiredValueFromYaml(const cv::FileStorage &file_storage,
                                      const std::string &name, T *value) {
  if (file_storage[name].empty()) {
    std::cerr << "Could not read value " << name << std::endl;
    return false;
  }
  ReadValueFromYaml(file_storage, name, value);
  return true;
}

// Commonly used functions to write value to .yaml file
void WriteValueToYaml(cv::FileStorage file_storage, const std::string &name,
                      Intrinsics intrinsics);
void WriteValueToYaml(cv::FileStorage file_storage, const std::string &name,
                      Transform3fA transformation);

// Commonly used functions to plot points to image
void DrawPointInImage(const Eigen::Vector3f &point_f_camera,
                      const cv::Vec3b &color, const Intrinsics &intrinsics,
                      cv::Mat *image);
void DrawLineInImage(const Eigen::Vector3f &point1_f_camera,
                     const Eigen::Vector3f &point2_f_camera,
                     const cv::Vec3b &color, const Intrinsics &intrinsics,
                     cv::Mat *image);
void DrawFocusedPointInImage(const Eigen::Vector3f &point_f_camera,
                             const cv::Vec3b &color,
                             const Intrinsics &intrinsics, float corner_u,
                             float corner_v, float scale, cv::Mat *image);

// Commonly used functions to access and print vector elements
template <typename T>
inline T LastValidValue(const std::vector<T> &values, int idx) {
  if (idx < values.size())
    return values[idx];
  else
    return values.back();
}

template <typename T>
inline void PrintValue(const std::string &name, const std::vector<T> &values,
                       int idx) {
  if (values.size() >= 1)
    std::cout << name << " = " << icg::LastValidValue(values, idx) << ", " << std::endl;
}

template <typename T>
inline void PrintValues(const std::string &name,
                        const std::vector<std::vector<T>> &values, int idx) {
  if (values.size() >= 1) {
    std::cout << name << " = {";
    for (const auto &v : icg::LastValidValue(values, idx))
      std::cout << v << ",";
    std::cout << "...}, " << std::endl;
  }
}

// Commonly used functions to assamble object pointers
template <typename T>
bool GetPtrIfNameExists(const std::string &name, const std::vector<T> &ptrs,
                        T *ptr) {
  auto it = std::find_if(begin(ptrs), end(ptrs),
                         [&name](const T &p) { return p->name() == name; });
  if (it == end(ptrs)) return false;
  *ptr = *it;
  return true;
}

template <typename T>
bool AddPtrIfNameNotExists(const T &ptr, std::vector<T> *dest_ptrs) {
  if (!ptr) return true;
  if (!std::none_of(begin(*dest_ptrs), end(*dest_ptrs),
                    [&ptr](const T &p) { return p->name() == ptr->name(); }))
    return false;
  dest_ptrs->push_back(ptr);
  return true;
}

template <typename T>
bool AddPtrsIfNameNotExists(const std::vector<T> &ptrs,
                            std::vector<T> *dest_ptrs) {
  bool ptrs_added = true;
  for (const T &ptr : ptrs) {
    if (!AddPtrIfNameNotExists(ptr, dest_ptrs)) ptrs_added = false;
  }
  return ptrs_added;
}

template <typename T>
bool DeletePtrIfNameExists(const std::string &name, std::vector<T> *ptrs) {
  auto it{std::remove_if(begin(*ptrs), end(*ptrs),
                         [&name](const T &p) { return p->name() == name; })};
  if (it == end(*ptrs)) return false;
  ptrs->erase(it, end(*ptrs));
  return true;
}

// Commonly used functions to set up object pointers
template <typename T>
bool SetUpObjectPtrs(std::vector<T> *ptrs) {
  for (T &ptr : *ptrs) {
    if (!ptr->SetUp()) {
      std::cout << "Error setting up " << ptr->name() << std::endl;
      return false;
    }
  }
  return true;
}

template <typename T>
bool AreObjectPtrsSetUp(std::vector<T> *ptrs) {
  for (T &ptr : *ptrs) {
    if (!ptr->set_up()) {
      std::cout << ptr->name() << " not set up" << std::endl;
      return false;
    }
  }
  return true;
}


/*
  平面外旋转转旋转矩阵
*/

inline Eigen::Matrix4f lookAt(Eigen::Vector3f const& eye, Eigen::Vector3f const& center, Eigen::Vector3f const& up)
{
    Eigen::Vector3f f(center - eye);
    Eigen::Vector3f s(f.cross(up));
    Eigen::Vector3f  u(s.cross(f));
    f.normalize();
    s.normalize();
    u.normalize();

    Eigen::Matrix4f  Result = Eigen::Matrix4f::Identity();
    Result(0, 0) = s.x();
    Result(0, 1) = s.y();
    Result(0, 2) = s.z();
    Result(1, 0) = u.x();
    Result(1, 1) = u.y();
    Result(1, 2) = u.z();
    Result(2, 0) = -f.x();
    Result(2, 1) = -f.y();
    Result(2, 2) = -f.z();
    Result(0, 3) = -s.dot(eye);
    Result(1, 3) = -u.dot(eye);
    Result(2, 3) = f.dot(eye);
    return Result;
}

inline cv::Matx33f getRFromGLM(const cv::Matx44f& m)
{
	const float* v = m.val;

	return  cv::Matx33f(v[0], v[4], v[8], v[1], v[5], v[9], v[2], v[6], v[10]);
}

inline cv::Matx33f dir2OutofplaneRotation(const cv::Vec3f& dir)
{
	auto eyePos = dir;
	// auto m = lookat(eyePos[0], eyePos[1], eyePos[2], 0, 0, 0, 0, 1, 0);
  auto m_eigen = lookAt(Eigen::Vector3f( eyePos[0], eyePos[1], eyePos[2]), Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 1, 0));
  cv::Matx44f m;
  cv::eigen2cv(m_eigen, m);
	return getRFromGLM(m);
}

inline cv::Vec3f theta2Dir(float theta_x, float theta_y)
{
	CV_Assert(fabs(theta_x) < CV_PI / 2 || fabs(theta_y) < CV_PI / 2);

	if (fabs(theta_x) < 1e-6f)
	{
		return cv::Vec3f(0.f, sin(theta_y), cos(theta_y));
	}
	else if (fabs(theta_y) < 1e-6f)
	{
		return cv::Vec3f(sin(theta_x), 0.f, cos(theta_x));
	}

	float a = 1.f/tan(theta_x);
	float b = 1.f/tan(theta_y);
	float z = sqrt(1.f / (a * a + b * b + 1.f));

	return cv::normalize(cv::Vec3f(z/a, z/b, z));
}

inline cv::Matx33f theta2OutofplaneRotation(float theta_x, float theta_y)
{
	return dir2OutofplaneRotation(theta2Dir(theta_x, theta_y));
}

inline cv::Vec3f viewDirFromR(const cv::Matx33f& R)
{
	return cv::normalize(cv::Vec3f(R(2, 0), R(2, 1), R(2, 2)));
}

inline cv::Vec2f dir2Theta(const cv::Vec3f& dir)
{
	float theta_x = atan2(dir[0], dir[2]);
	float theta_y = atan2(dir[1], dir[2]);
	return cv::Vec2f(theta_x, theta_y);
}

}  // namespace icg

#endif  // ICG_INCLUDE_ICG_COMMON_H_
