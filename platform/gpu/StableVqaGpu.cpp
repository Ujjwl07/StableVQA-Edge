#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

//  BUILD-TIME CONFIGURATION

// Default directory holding the ONNX files (override at runtime with argv[2]).
static string MODEL_DIR = "onnx_models/";

// Per-branch filenames (edit these to swap precision per branch).
static string FLOW_MODEL = "flow_model_quant.onnx";
static string BACKBONE_MODEL = "backbone_fp16.onnx";
static string MOTION_MODEL = "motion_analyzer_fp16.onnx";
static string DEBLUR_MODEL = "deblur_net_fp16.onnx"; // used only if USE_BLUR
static string HEAD_MODEL = "quality_head_fp16.onnx";

static const bool USE_BLUR = true;

static const bool USE_DUAL_GPU = false; // flip this ONE line to test x2

static const bool USE_CONCURRENT = true; // set false for sequential execution

static const int FLOW_DEVICE = 0;   // GPU 0 in both modes
static const int MOTION_DEVICE = 0; // GPU 0 in both modes
static const int BACKBONE_DEVICE = USE_DUAL_GPU ? 1 : 0; // GPU 1 when dual
static const int DEBLUR_DEVICE =
    USE_DUAL_GPU ? 1 : 0; // GPU 1 when dual (only if USE_BLUR)
static const int HEAD_DEVICE = USE_DUAL_GPU ? 1 : 0; // GPU 1 when dual

static const bool USE_TENSORRT = false;
static const string TRT_CACHE_DIR = "./trt_engine_cache";

float map_to_mos(float raw) {
  const double b1 = 92.91053891;
  const double b2 = 0.35856351;
  const double b3 = -2.3526431;
  const double b4 = 63.48772513;
  double z = b2 * (raw - b3);
  z = max(-60.0, min(60.0, z));
  double mos = b1 * (0.5 - 1.0 / (1.0 + exp(z))) + b4;
  return (float)mos;
}

string stability_label(float mos) {
  if (mos <= 20)
    return "Very Unstable";
  if (mos <= 40)
    return "Unstable";
  if (mos <= 60)
    return "Moderately Stable";
  if (mos <= 80)
    return "Stable";
  return "Highly Stable";
}

float mean_abs(const vector<float> &v) {
  if (v.empty())
    return 0.f;
  double sum = 0.0;
  for (float x : v)
    sum += std::abs(double(x));
  return (float)(sum / v.size());
}

float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

float normalize_score(float x, float xmin, float xmax) {
  if (xmax <= xmin)
    return 50.f;
  return clamp01((x - xmin) / (xmax - xmin)) * 100.f;
}

void compute_branch_indicators(float spatial_raw, float blur_raw,
                               float motion_raw, float &out_spatial,
                               float &out_blur, float &out_motion) {
  out_spatial = 100.f - normalize_score(spatial_raw, 0.1479f, 0.4086f);
  out_blur = 100.f - normalize_score(blur_raw, 0.5129f, 2.8472f);
  out_motion = 100.f - normalize_score(motion_raw, 0.2580f, 3.6100f);
}

size_t get_rss_kb() {
  long rss = 0L;
  std::ifstream statm("/proc/self/statm");
  if (statm.is_open()) {
    long size;
    statm >> size >> rss;
    statm.close();
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE) / 1024;
  }
  return 0;
}

vector<float> global_avg_pool(const vector<float> &data,
                              const vector<int64_t> &shape) {
  if (shape.size() != 4)
    return data;
  int batch = shape[0];
  int channels = shape[1];
  int spatial = shape[2] * shape[3];
  vector<float> pooled;
  pooled.reserve(batch * channels);
  for (int b = 0; b < batch; ++b)
    for (int c = 0; c < channels; ++c) {
      const float *ptr = &data[(b * channels * spatial) + (c * spatial)];
      float sum = 0.0f;
      for (int k = 0; k < spatial; ++k)
        sum += ptr[k];
      pooled.push_back(sum / spatial);
    }
  return pooled;
}

struct ClipMetrics {
  float score;
  double latency_s;
  double ms_backbone, ms_deblur, ms_flow, ms_motion, ms_quality;
  size_t rss_start, rss_after_back, rss_after_deblur, rss_after_flow,
      rss_after_motion;

  float spatial_indicator;
  float blur_indicator;
  float motion_indicator;

  float spatial_raw;
  float blur_raw;
  float motion_raw;
};

struct VideoResult {
  string video_name;
  float mean_score;
  float std_dev;
  vector<ClipMetrics> clips;
};

void write_csv(const string &filename, const vector<VideoResult> &results) {
  ofstream f(filename);
  f << "VideoName,Clip,StartPos,Score,"
    << "SpatialIndicator,BlurIndicator,MotionIndicator,"
    << "TotalLatency(s),Backbone(ms),Deblur(ms),Flow(ms),"
    << "Motion(ms),QualityHead(ms),"
    << "RSS_Start(KB),RSS_AfterBackbone(KB),RSS_AfterDeblur(KB),"
    << "RSS_AfterFlow(KB),RSS_AfterMotion(KB),"
    << "BackboneRAMDelta(KB),DeblurRAMDelta(KB),"
    << "FlowRAMDelta(KB),MotionRAMDelta(KB),"
    << "VideoMeanScore,VideoStdDev,StabilityLabel\n";

  vector<string> clip_labels = {"12.5%", "37.5%", "62.5%", "87.5%"};

  for (const auto &r : results)
    for (size_t i = 0; i < r.clips.size(); i++) {
      const auto &c = r.clips[i];
      long long back_delta =
          (long long)c.rss_after_back - (long long)c.rss_start;
      long long deblur_delta =
          (long long)c.rss_after_deblur - (long long)c.rss_after_back;
      long long flow_delta =
          (long long)c.rss_after_flow - (long long)c.rss_after_deblur;
      long long motion_delta =
          (long long)c.rss_after_motion - (long long)c.rss_after_flow;

      f << r.video_name << "," << (i + 1) << ","
        << (i < clip_labels.size() ? clip_labels[i] : "?") << "," << fixed
        << setprecision(4) << c.score << "," << setprecision(2)
        << c.spatial_indicator << "," << c.blur_indicator << ","
        << c.motion_indicator << "," << setprecision(3) << c.latency_s << ","
        << setprecision(0) << c.ms_backbone << "," << c.ms_deblur << ","
        << c.ms_flow << "," << c.ms_motion << "," << c.ms_quality << ","
        << c.rss_start << "," << c.rss_after_back << "," << c.rss_after_deblur
        << "," << c.rss_after_flow << "," << c.rss_after_motion << ","
        << back_delta << "," << deblur_delta << "," << flow_delta << ","
        << motion_delta << "," << setprecision(4) << r.mean_score << ","
        << r.std_dev << "," << stability_label(r.mean_score) << "\n";
    }
}

void write_calibration_csv(const string &filename,
                           const vector<VideoResult> &results) {
  ofstream f(filename);
  f << "VideoName,Clip,MOS,SpatialRaw,BlurRaw,MotionRaw\n";

  float sp_min = 1e30f, sp_max = -1e30f;
  float bl_min = 1e30f, bl_max = -1e30f;
  float mo_min = 1e30f, mo_max = -1e30f;

  vector<string> labels = {"12.5%", "37.5%", "62.5%", "87.5%"};

  for (const auto &r : results)
    for (size_t i = 0; i < r.clips.size(); i++) {
      const auto &c = r.clips[i];
      f << r.video_name << "," << (i < labels.size() ? labels[i] : "?") << ","
        << fixed << setprecision(4) << c.score << "," << c.spatial_raw << ","
        << c.blur_raw << "," << c.motion_raw << "\n";

      sp_min = min(sp_min, c.spatial_raw);
      sp_max = max(sp_max, c.spatial_raw);
      bl_min = min(bl_min, c.blur_raw);
      bl_max = max(bl_max, c.blur_raw);
      mo_min = min(mo_min, c.motion_raw);
      mo_max = max(mo_max, c.motion_raw);
    }

  f << "\nSUMMARY,,,"
    << "spatial_min=" << sp_min << " spatial_max=" << sp_max << ","
    << "blur_min=" << bl_min << " blur_max=" << bl_max << ","
    << "motion_min=" << mo_min << " motion_max=" << mo_max << "\n";

  cout << "\nCalibration Summary (" << results.size() << " videos)\n"
       << fixed << setprecision(4) << "  spatial_raw : min=" << sp_min
       << "  max=" << sp_max << "\n"
       << "  blur_raw    : min=" << bl_min << "  max=" << bl_max << "\n"
       << "  motion_raw  : min=" << mo_min << "  max=" << mo_max << "\n";
}

//  ACCURACY METRICS ESTIMATION (SROCC, PLCC, KROCC, RMSE)
string trim(const string &str) {
  size_t first = str.find_first_not_of(" \t\r\n");
  if (first == string::npos)
    return "";
  size_t last = str.find_last_not_of(" \t\r\n");
  return str.substr(first, (last - first + 1));
}

map<string, float> load_labels(const string &list_path) {
  map<string, float> label_map;
  ifstream file(list_path);
  if (!file.is_open()) {
    cerr << "Warning: Could not open label list file: " << list_path << endl;
    return label_map;
  }
  string line;
  while (getline(file, line)) {
    if (line.empty())
      continue;
    stringstream ss(line);
    string cell;
    vector<string> row;
    while (getline(ss, cell, ',')) {
      row.push_back(cell);
    }
    if (row.size() >= 4) {
      string video_name = trim(row[0]);
      try {
        float score = stof(trim(row[3]));
        label_map[video_name] = score;
      } catch (...) {
        // ignore parsing error
      }
    }
  }
  return label_map;
}

vector<float> rescale(const vector<float> &pr, const vector<float> &gt) {
  size_t n = pr.size();
  if (n == 0)
    return pr;

  double sum_pr = 0.0, sum_gt = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum_pr += pr[i];
    sum_gt += gt[i];
  }
  double mean_pr = sum_pr / n;
  double mean_gt = sum_gt / n;

  double var_pr = 0.0, var_gt = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d_pr = pr[i] - mean_pr;
    double d_gt = gt[i] - mean_gt;
    var_pr += d_pr * d_pr;
    var_gt += d_gt * d_gt;
  }
  double std_pr = sqrt(var_pr / n);
  double std_gt = sqrt(var_gt / n);

  vector<float> res(n);
  for (size_t i = 0; i < n; ++i) {
    if (std_pr == 0.0) {
      res[i] = mean_gt;
    } else {
      res[i] = ((pr[i] - mean_pr) / std_pr) * std_gt + mean_gt;
    }
  }
  return res;
}

double compute_plcc(const vector<float> &x, const vector<float> &y) {
  size_t n = x.size();
  if (n <= 1)
    return 0.0;
  double sum_x = 0.0, sum_y = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum_x += x[i];
    sum_y += y[i];
  }
  double mean_x = sum_x / n;
  double mean_y = sum_y / n;

  double num = 0.0;
  double den_x = 0.0, den_y = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double dx = x[i] - mean_x;
    double dy = y[i] - mean_y;
    num += dx * dy;
    den_x += dx * dx;
    den_y += dy * dy;
  }
  if (den_x == 0.0 || den_y == 0.0)
    return 0.0;
  return num / sqrt(den_x * den_y);
}

vector<double> get_ranks(const vector<float> &v) {
  size_t n = v.size();
  vector<size_t> idx(n);
  iota(idx.begin(), idx.end(), 0);
  sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });

  vector<double> ranks(n);
  size_t i = 0;
  while (i < n) {
    size_t j = i + 1;
    while (j < n && v[idx[j]] == v[idx[i]]) {
      j++;
    }
    double rank = 1.0 + (double)(i + j - 1) / 2.0;
    for (size_t k = i; k < j; ++k) {
      ranks[idx[k]] = rank;
    }
    i = j;
  }
  return ranks;
}

double compute_srocc(const vector<float> &x, const vector<float> &y) {
  vector<double> rx = get_ranks(x);
  vector<double> ry = get_ranks(y);
  size_t n = x.size();
  if (n <= 1)
    return 0.0;
  double sum_x = 0.0, sum_y = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum_x += rx[i];
    sum_y += ry[i];
  }
  double mean_x = sum_x / n;
  double mean_y = sum_y / n;

  double num = 0.0;
  double den_x = 0.0, den_y = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double dx = rx[i] - mean_x;
    double dy = ry[i] - mean_y;
    num += dx * dy;
    den_x += dx * dx;
    den_y += dy * dy;
  }
  if (den_x == 0.0 || den_y == 0.0)
    return 0.0;
  return num / sqrt(den_x * den_y);
}

double compute_krocc(const vector<float> &x, const vector<float> &y) {
  size_t n = x.size();
  if (n <= 1)
    return 0.0;
  long long Nc = 0, Nd = 0;
  long long N1 = 0, N2 = 0;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      if (dx == 0.0)
        N1++;
      if (dy == 0.0)
        N2++;
      if (dx != 0.0 && dy != 0.0) {
        if (dx * dy > 0.0)
          Nc++;
        else
          Nd++;
      }
    }
  }
  long long N0 = (long long)n * (n - 1) / 2;
  double den = sqrt((double)(N0 - N1) * (double)(N0 - N2));
  if (den == 0.0)
    return 0.0;
  return (double)(Nc - Nd) / den;
}

double compute_rmse(const vector<float> &x, const vector<float> &y) {
  size_t n = x.size();
  if (n == 0)
    return 0.0;
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = x[i] - y[i];
    sum += d * d;
  }
  return sqrt(sum / n);
}

class ONNXRunner {
  Ort::Env &env;
  Ort::Session session;
  Ort::AllocatorWithDefaultOptions allocator;
  vector<string> input_node_names;
  vector<string> output_node_names;
  vector<const char *> in_ptrs, out_ptrs;

public:
  ONNXRunner(Ort::Env &env, const string &model_path, bool use_trt,
             int device_id = 0)
      : env(env), session(nullptr) {
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_trt) {
      try {
        OrtTensorRTProviderOptions trt_opts{};
        trt_opts.device_id = device_id;
        trt_opts.trt_fp16_enable = 1;
        trt_opts.trt_engine_cache_enable = 1;
        trt_opts.trt_engine_cache_path = TRT_CACHE_DIR.c_str();
        options.AppendExecutionProvider_TensorRT(trt_opts);
      } catch (const exception &e) {
        cerr << "  [TensorRT unavailable, falling back to CUDA] " << e.what()
             << endl;
      }
    }

    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = device_id;
    cuda_opts.gpu_mem_limit = SIZE_MAX;
    cuda_opts.arena_extend_strategy = 0;
    cuda_opts.do_copy_in_default_stream = 1;
    cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    options.AppendExecutionProvider_CUDA(cuda_opts);

    session = Ort::Session(env, model_path.c_str(), options);

    for (size_t i = 0; i < session.GetInputCount(); i++) {
      auto p = session.GetInputNameAllocated(i, allocator);
      string name = (p && strlen(p.get()) > 0) ? p.get() : "input";
      input_node_names.push_back(name);
      in_ptrs.push_back(input_node_names.back().c_str());
    }
    for (size_t i = 0; i < session.GetOutputCount(); i++) {
      auto p = session.GetOutputNameAllocated(i, allocator);
      string name = (p && strlen(p.get()) > 0) ? p.get() : "output";
      output_node_names.push_back(name);
      out_ptrs.push_back(output_node_names.back().c_str());
    }
  }

  pair<vector<float>, vector<int64_t>> forward(vector<float> &data,
                                               const vector<int64_t> &shape) {
    Ort::MemoryInfo cpu_mem_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        cpu_mem_info, data.data(), data.size(), shape.data(), shape.size());

    Ort::IoBinding io_binding(session);
    io_binding.BindInput(input_node_names[0].c_str(), input_tensor);
    io_binding.BindOutput(output_node_names[0].c_str(), cpu_mem_info);

    session.Run(Ort::RunOptions{nullptr}, io_binding);

    auto outputs = io_binding.GetOutputValues();
    float *f = outputs.front().GetTensorMutableData<float>();
    auto out_shape = outputs.front().GetTensorTypeAndShapeInfo().GetShape();
    size_t count =
        outputs.front().GetTensorTypeAndShapeInfo().GetElementCount();

    return {vector<float>(f, f + count), out_shape};
  }
};

void preprocess_dual(const Mat &frame, vector<float> &norm_buf,
                     vector<float> &raw_buf) {
  Mat resized, float_mat;
  resize(frame, resized, Size(224, 224));
  cvtColor(resized, float_mat, COLOR_BGR2RGB);
  float_mat.convertTo(float_mat, CV_32F);

  const float mean[] = {123.675f, 116.28f, 103.53f};
  const float std_val[] = {58.395f, 57.12f, 57.375f};
  int pixels = 224 * 224;

  size_t start_idx = norm_buf.size();
  norm_buf.resize(start_idx + 3 * pixels);
  raw_buf.resize(start_idx + 3 * pixels);
  float *norm_ptr = norm_buf.data() + start_idx;
  float *raw_ptr = raw_buf.data() + start_idx;

  vector<Mat> channels_m;
  split(float_mat, channels_m);

  for (int c = 0; c < 3; ++c) {
    Mat norm_chan = (channels_m[c] - mean[c]) / std_val[c];
    memcpy(norm_ptr + c * pixels, norm_chan.data, pixels * sizeof(float));
    memcpy(raw_ptr + c * pixels, channels_m[c].data, pixels * sizeof(float));
  }
}

vector<float> make_backbone_input_like_evaluator(const vector<float> &clip,
                                                 int frames, int channels,
                                                 int pixels) {
  vector<float> out(frames * channels * pixels);
  for (int row = 0; row < frames; ++row)
    for (int out_c = 0; out_c < channels; ++out_c) {
      int q = row * channels + out_c;
      int src_c = q / frames;
      int src_t = q % frames;
      const float *src = clip.data() + (src_t * channels + src_c) * pixels;
      float *dst = out.data() + (row * channels + out_c) * pixels;
      memcpy(dst, src, pixels * sizeof(float));
    }
  return out;
}

vector<float> make_flow_pairs_like_evaluator(const vector<float> &clip,
                                             int frames, int channels,
                                             int pixels) {
  const int frame_size = channels * pixels;
  vector<float> out;
  out.reserve(frames * 2 * frame_size);
  for (int t = 0; t < frames; ++t) {
    int next_t = min(t + 1, frames - 1);
    const float *f1 = clip.data() + t * frame_size;
    const float *f2 = clip.data() + next_t * frame_size;
    out.insert(out.end(), f1, f1 + frame_size);
    out.insert(out.end(), f2, f2 + frame_size);
  }
  return out;
}

vector<float> make_deblur_input(const vector<float> &clip, int frames,
                                int channels, int pixels, int blur_frames) {
  const int frame_size = channels * pixels;
  const int stride = frames / blur_frames; // 32 / 8 = 4
  vector<float> out(blur_frames * frame_size);
  for (int b = 0; b < blur_frames; ++b) {
    const float *src = clip.data() + (b * stride) * frame_size;
    memcpy(out.data() + b * frame_size, src, frame_size * sizeof(float));
  }
  return out;
}

bool extract_clip(VideoCapture &cap, int start_frame,
                  vector<float> &norm_frames, vector<float> &raw_frames) {
  norm_frames.clear();
  raw_frames.clear();
  const int frame_size = 3 * 224 * 224;
  norm_frames.reserve(32 * frame_size);
  raw_frames.reserve(32 * frame_size);

  cap.set(CAP_PROP_POS_FRAMES, start_frame);
  Mat f;
  int collected = 0;

  for (int i = 0; i < 64 && collected < 32; i++) {
    if (i % 2 == 0) {
      cap >> f;
      if (f.empty())
        break;
      preprocess_dual(f, norm_frames, raw_frames);
      collected++;
    } else {
      cap.grab();
    }
  }

  if (collected == 0)
    return false;

  while (norm_frames.size() < (size_t)(32 * frame_size) &&
         !norm_frames.empty()) {
    norm_frames.insert(norm_frames.end(), norm_frames.end() - frame_size,
                       norm_frames.end());
    raw_frames.insert(raw_frames.end(), raw_frames.end() - frame_size,
                      raw_frames.end());
  }
  return true;
}

ClipMetrics infer_clip(const vector<float> &norm_frames,
                       const vector<float> &raw_frames, ONNXRunner *flow,
                       ONNXRunner *back, ONNXRunner *deblur, ONNXRunner *mot,
                       ONNXRunner *head) {
  ClipMetrics metrics = {};
  const int NUM_FRAMES = 32;
  const int CHANNELS = 3;
  const int BLUR_FRAMES = 8;
  const int FLOW_PAIRS = NUM_FRAMES;
  const int pixels = 224 * 224;
  (void)raw_frames;

  metrics.rss_start = get_rss_kb();

  auto launch_policy = USE_CONCURRENT ? launch::async : launch::deferred;

  auto fut_back = async(launch_policy, [&]() {
    auto t0 = chrono::high_resolution_clock::now();
    vector<float> b_in = make_backbone_input_like_evaluator(
        norm_frames, NUM_FRAMES, CHANNELS, pixels);
    auto out = back->forward(b_in, {NUM_FRAMES, CHANNELS, 224, 224}).first;
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    return make_pair(std::move(out), ms);
  });

  auto fut_flow = async(launch_policy, [&]() {
    auto t0 = chrono::high_resolution_clock::now();
    vector<float> flow_in = make_flow_pairs_like_evaluator(
        norm_frames, NUM_FRAMES, CHANNELS, pixels);
    auto out = flow->forward(flow_in, {FLOW_PAIRS, 6, 224, 224}).first;
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
    return make_pair(std::move(out), ms);
  });

  future<pair<vector<float>, double>> fut_blur;
  if (USE_BLUR && deblur) {
    fut_blur = async(launch_policy, [&]() {
      auto t0 = chrono::high_resolution_clock::now();
      vector<float> d_in = make_deblur_input(norm_frames, NUM_FRAMES, CHANNELS,
                                             pixels, BLUR_FRAMES);
      auto res = deblur->forward(d_in, {BLUR_FRAMES, CHANNELS, 224, 224});
      auto pooled = global_avg_pool(res.first, res.second); // [8*320] = 2560
      auto t1 = chrono::high_resolution_clock::now();
      double ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
      return make_pair(std::move(pooled), ms);
    });
  }

  auto back_res = fut_back.get();
  vector<float> &b_out = back_res.first;
  metrics.ms_backbone = back_res.second;
  metrics.rss_after_back = get_rss_kb();

  vector<float> d_out;
  if (USE_BLUR && deblur) {
    auto blur_res = fut_blur.get();
    d_out = std::move(blur_res.first);
    metrics.ms_deblur = blur_res.second;
  } else {
    d_out.assign(BLUR_FRAMES * 320, 0.0f); // 2560 zeros
    metrics.ms_deblur = 0.0;
  }
  metrics.rss_after_deblur = get_rss_kb();

  auto flow_out_res = fut_flow.get();
  vector<float> &flow_res = flow_out_res.first;
  metrics.ms_flow = flow_out_res.second;
  metrics.rss_after_flow = get_rss_kb();

  if (flow_res.size() != (size_t)NUM_FRAMES * 2 * pixels)
    throw runtime_error("flow output is not [32,2,224,224]; re-export NeuFlow "
                        "with FLOW_PAIRS=NUM_FRAMES");

  auto t_mot0 = chrono::high_resolution_clock::now();
  vector<float> m_input(2 * NUM_FRAMES * pixels);
  for (int k = 0; k < NUM_FRAMES; ++k) {
    const float *src = flow_res.data() + (k * 2 * pixels);
    memcpy(m_input.data() + k * pixels, src, pixels * sizeof(float));
    memcpy(m_input.data() + NUM_FRAMES * pixels + k * pixels, src + pixels,
           pixels * sizeof(float));
  }
  auto m_out = mot->forward(m_input, {1, 2, NUM_FRAMES, 224, 224}).first;
  auto t_mot1 = chrono::high_resolution_clock::now();
  metrics.ms_motion =
      chrono::duration_cast<chrono::milliseconds>(t_mot1 - t_mot0).count();
  metrics.rss_after_motion = get_rss_kb();

  metrics.spatial_raw = mean_abs(b_out);
  metrics.blur_raw = USE_BLUR ? mean_abs(d_out) : 0.0f;
  metrics.motion_raw = mean_abs(m_out);
  compute_branch_indicators(metrics.spatial_raw, metrics.blur_raw,
                            metrics.motion_raw, metrics.spatial_indicator,
                            metrics.blur_indicator, metrics.motion_indicator);

  vector<float> fused;
  fused.reserve(d_out.size() + b_out.size() + m_out.size());
  fused.insert(fused.end(), d_out.begin(), d_out.end());
  fused.insert(fused.end(), b_out.begin(), b_out.end());
  fused.insert(fused.end(), m_out.begin(), m_out.end());
  auto t5 = chrono::high_resolution_clock::now();
  float raw_score = head->forward(fused, {1, (int64_t)fused.size()}).first[0];
  auto t6 = chrono::high_resolution_clock::now();
  metrics.ms_quality =
      chrono::duration_cast<chrono::milliseconds>(t6 - t5).count();

  metrics.score = map_to_mos(raw_score);
  return metrics;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    cout
        << "Usage: ./stablevqa_exp1 <folder_path> [model_dir] [label_list_path]"
        << endl;
    return -1;
  }

  string folder_path = argv[1];
  if (argc >= 3) {
    MODEL_DIR = argv[2];
    if (!MODEL_DIR.empty() && MODEL_DIR.back() != '/')
      MODEL_DIR += '/';
  }

  string label_list_path = "";
  if (argc >= 4) {
    label_list_path = argv[3];
  }

  vector<string> video_paths;
  vector<string> valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

  try {
    for (const auto &entry : fs::directory_iterator(folder_path)) {
      if (entry.is_regular_file()) {
        string ext = entry.path().extension().string();
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (find(valid_exts.begin(), valid_exts.end(), ext) != valid_exts.end())
          video_paths.push_back(entry.path().string());
      }
    }
  } catch (const fs::filesystem_error &e) {
    cerr << "Folder Error: " << e.what() << endl;
    return -1;
  }

  if (video_paths.empty()) {
    cout << "No video files found in: " << folder_path << endl;
    return 0;
  }
  sort(video_paths.begin(), video_paths.end());

  auto t_start_total = chrono::high_resolution_clock::now();

  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "StableVQA");
  string csv_file = "stablevqa_results.csv";
  string calib_file = "stablevqa_calibration.csv";

  cout << "StableVQA GPU inference\n";
  cout << "  Model dir : " << MODEL_DIR << "\n";
  cout << "  backbone  : " << BACKBONE_MODEL << "\n";
  cout << "  flow      : " << FLOW_MODEL << "\n";
  cout << "  motion    : " << MOTION_MODEL << "\n";
  cout << "  head      : " << HEAD_MODEL << "\n";
  cout << "  blur      : "
       << (USE_BLUR ? DEBLUR_MODEL : string("DISABLED (zeros)")) << "\n";
  cout << "  provider  : " << (USE_TENSORRT ? "TensorRT+CUDA" : "CUDA") << "\n";

  cout << "  gpu mode  : "
       << (USE_DUAL_GPU ? "DUAL  (flow+motion->GPU0, backbone+blur+head->GPU1)"
                        : "SINGLE (all branches->GPU0)")
       << "\n";

  auto f_flow = async(launch::async, [&] {
    return new ONNXRunner(env, MODEL_DIR + FLOW_MODEL, USE_TENSORRT,
                          FLOW_DEVICE);
  });
  auto f_back = async(launch::async, [&] {
    return new ONNXRunner(env, MODEL_DIR + BACKBONE_MODEL, USE_TENSORRT,
                          BACKBONE_DEVICE);
  });
  auto f_mot = async(launch::async, [&] {
    return new ONNXRunner(env, MODEL_DIR + MOTION_MODEL, USE_TENSORRT,
                          MOTION_DEVICE);
  });
  auto f_head = async(launch::async, [&] {
    return new ONNXRunner(env, MODEL_DIR + HEAD_MODEL, USE_TENSORRT,
                          HEAD_DEVICE);
  });

  future<ONNXRunner *> f_deblur;
  if (USE_BLUR)
    f_deblur = async(launch::async, [&] {
      return new ONNXRunner(env, MODEL_DIR + DEBLUR_MODEL, USE_TENSORRT,
                            DEBLUR_DEVICE);
    });

  ONNXRunner *flow = f_flow.get(), *back = f_back.get(), *mot = f_mot.get(),
             *head = f_head.get();
  ONNXRunner *deblur = USE_BLUR ? f_deblur.get() : nullptr;

  auto t_loaded = chrono::high_resolution_clock::now();
  cout << "Models loaded in "
       << chrono::duration_cast<chrono::milliseconds>(t_loaded - t_start_total)
                  .count() /
              1000.0
       << "s" << endl;

  cout << "Performing warm-up run" << endl;
  {
    vector<float> dummy(32 * 3 * 224 * 224, 0.0f);
    infer_clip(dummy, dummy, flow, back, deblur, mot, head);
  }
  cout << "Warm-up complete. Starting batch processing.\n" << endl;

  int video_count = 0;
  vector<VideoResult> all_results;

  map<string, float> label_map;
  bool has_labels = false;
  vector<float> pr_scores;
  vector<float> gt_scores;
  if (!label_list_path.empty()) {
    label_map = load_labels(label_list_path);
    if (!label_map.empty()) {
      has_labels = true;
      cout << "Loaded " << label_map.size()
           << " ground-truth labels from: " << label_list_path << "\n";
    }
  }

  try {
    for (const string &vid_path : video_paths) {
      string video_name = fs::path(vid_path).filename().string();
      cout << "[" << ++video_count << "/" << video_paths.size() << "] "
           << video_name << endl;

      VideoCapture cap(vid_path);
      if (!cap.isOpened()) {
        cerr << "  Cannot open, skipping.\n";
        continue;
      }

      int total = (int)cap.get(CAP_PROP_FRAME_COUNT);
      int max_start = max(0, total - 64);

      VideoResult vres;
      vres.video_name = video_name;
      vector<float> clip_scores;

      vector<string> pos_labels = {"12.5%", "37.5%", "62.5%", "87.5%"};

      for (int ci = 0; ci < 4; ci++) {
        float pos = (ci + 0.5f) / 4;
        int start_frame = (int)(pos * max_start);

        vector<float> norm_frames, raw_frames;
        if (!extract_clip(cap, start_frame, norm_frames, raw_frames)) {
          cerr << "  Clip " << (ci + 1) << ": extraction failed, skipping.\n";
          continue;
        }

        auto t_clip = chrono::high_resolution_clock::now();
        ClipMetrics m =
            infer_clip(norm_frames, raw_frames, flow, back, deblur, mot, head);
        m.latency_s = chrono::duration_cast<chrono::milliseconds>(
                          chrono::high_resolution_clock::now() - t_clip)
                          .count() /
                      1000.0;

        clip_scores.push_back(m.score);
        vres.clips.push_back(m);

        cout << fixed << setprecision(2) << "  Clip " << (ci + 1) << " ["
             << pos_labels[ci] << "]"
             << "  MOS=" << m.score
             << "  SpatialIndicator=" << m.spatial_indicator
             << "  BlurIndicator=" << m.blur_indicator
             << "  MotionIndicator=" << m.motion_indicator
             << "  Time=" << m.latency_s << "s" << endl;
      }

      if (!clip_scores.empty()) {
        float mean = accumulate(clip_scores.begin(), clip_scores.end(), 0.0f) /
                     clip_scores.size();
        float var = 0.0f;
        for (float s : clip_scores)
          var += (s - mean) * (s - mean);
        vres.mean_score = mean;
        vres.std_dev = sqrt(var / clip_scores.size());
        all_results.push_back(vres);

        cout << "\n  Mean MOS : " << fixed << setprecision(2) << mean << "  ("
             << stability_label(mean) << ")"
             << "  StdDev: " << vres.std_dev << "\n"
             << endl;

        write_csv(csv_file, all_results);

        if (has_labels) {
          if (label_map.find(video_name) != label_map.end()) {
            pr_scores.push_back(mean);
            gt_scores.push_back(label_map[video_name]);
          }
        }
      }
    }

    auto t_end = chrono::high_resolution_clock::now();
    double total_s =
        chrono::duration_cast<chrono::milliseconds>(t_end - t_start_total)
            .count() /
        1000.0;
    cout << "Done. " << video_count << " videos in " << total_s << "s.\n";
    cout << "Results     : " << csv_file << "\n";

    write_calibration_csv(calib_file, all_results);
    cout << "Calibration : " << calib_file << "\n";

    if (has_labels && !gt_scores.empty()) {
      vector<float> rescaled_pr = rescale(pr_scores, gt_scores);
      double srocc = compute_srocc(rescaled_pr, gt_scores);
      double plcc = compute_plcc(rescaled_pr, gt_scores);
      double krocc = compute_krocc(rescaled_pr, gt_scores);
      double rmse = compute_rmse(rescaled_pr, gt_scores);

      cout << "\nTesting result on: [" << gt_scores.size() << "] videos:\n"
           << fixed << setprecision(4) << "            SROCC: " << srocc << "\n"
           << "            PLCC:  " << plcc << "\n"
           << "            KROCC: " << krocc << "\n"
           << "            RMSE:  " << rmse << ".\n"
           << endl;
    }
  } catch (const exception &e) {
    cerr << "Runtime Error: " << e.what() << endl;
  }

  delete flow;
  delete back;
  delete mot;
  delete head;
  if (deblur)
    delete deblur;
  return 0;
}
