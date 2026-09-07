#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <string>
#include <cstring>
#include <stdexcept>
#include <future>
#include <tuple>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

static const bool USE_CONCURRENCY = true;  
static const int  TRIO_THREADS    = 3;     
static const int  MOTION_THREADS  = 8;     
static const int  HEAD_THREADS    = 1;     
static const int  SERIAL_THREADS  = 4;     
static const bool USE_BLUR        = true;  

float map_to_mos(float raw)
{
  const double b1 = 92.91053891;
  const double b2 = 0.35856351;
  const double b3 = -2.3526431;
  const double b4 = 63.48772513;
  double z = b2 * (raw - b3);
  z = max(-60.0, min(60.0, z));
  double mos = b1 * (0.5 - 1.0 / (1.0 + exp(z))) + b4;
  return (float)mos;
}

string stability_label(float mos)
{
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

float mean_abs(const vector<float> &v)
{
    if (v.empty())
        return 0.f;
    double sum = 0.0;
    for (float x : v)
        sum += std::abs(double(x));
    return (float)(sum / v.size());
}

float clamp01(float x)
{
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

float normalize_score(float x, float xmin, float xmax)
{
    if (xmax <= xmin)
        return 50.f;
    return clamp01((x - xmin) / (xmax - xmin)) * 100.f;
}

void compute_branch_indicators(float spatial_raw, float blur_raw, float motion_raw,
                               float &out_spatial, float &out_blur, float &out_motion)
{
    out_spatial = 100.f - normalize_score(spatial_raw, 0.1479f, 0.4086f);
    out_blur = 100.f - normalize_score(blur_raw, 0.5129f, 2.8472f);
    out_motion = 100.f - normalize_score(motion_raw, 0.2580f, 3.6100f);
}

size_t get_rss_kb()
{
#ifdef __APPLE__
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS)
        return info.resident_size / 1024;
#endif
    return 0;
}

vector<float> global_avg_pool(const vector<float> &data, const vector<int64_t> &shape)
{
    if (shape.size() != 4)
        return data;
    int B = (int)shape[0], C = (int)shape[1];
    int spatial = (int)(shape[2] * shape[3]);
    vector<float> out;
    out.reserve(B * C);
    for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c)
        {
            const float *p = data.data() + (b * C * spatial) + (c * spatial);
            float s = 0.0f;
            for (int k = 0; k < spatial; ++k)
                s += p[k];
            out.push_back(s / spatial);
        }
    return out;
}

struct ClipMetrics
{
    float score;
    double latency_s;
    double ms_decode, ms_preprocess;
    double ms_backbone, ms_deblur, ms_flow, ms_motion, ms_quality;
    size_t rss_start, rss_after_back, rss_after_deblur, rss_after_flow, rss_after_motion;

    float spatial_indicator;
    float blur_indicator;
    float motion_indicator;

    float spatial_raw;
    float blur_raw;
    float motion_raw;
};

struct VideoResult
{
    string video_name;
    float mean_score;
    float std_dev;
    vector<ClipMetrics> clips;
};

void write_csv(const string &filename, const vector<VideoResult> &results)
{
    ofstream f(filename);
    f << "VideoName,Clip,StartPos,Score,"
      << "SpatialIndicator,BlurIndicator,MotionIndicator,"
      << "TotalLatency(s),FrameDecode(ms),Preprocess(ms),"
      << "Backbone(ms),Deblur(ms),Flow(ms),Motion(ms),QualityHead(ms),"
      << "RSS_Start(KB),RSS_AfterBackbone(KB),RSS_AfterDeblur(KB),"
      << "RSS_AfterFlow(KB),RSS_AfterMotion(KB),"
      << "BackboneRAMDelta(KB),DeblurRAMDelta(KB),FlowRAMDelta(KB),MotionRAMDelta(KB),"
      << "VideoMeanScore,VideoStdDev,StabilityLabel\n";

    vector<string> labels = {"12.5%", "37.5%", "62.5%", "87.5%"};

    for (const auto &r : results)
        for (size_t i = 0; i < r.clips.size(); i++)
        {
            const auto &c = r.clips[i];
            long long bd = (long long)c.rss_after_back - (long long)c.rss_start;
            long long dd = (long long)c.rss_after_deblur - (long long)c.rss_after_back;
            long long fd = (long long)c.rss_after_flow - (long long)c.rss_after_deblur;
            long long md = (long long)c.rss_after_motion - (long long)c.rss_after_flow;

            f << r.video_name << ","
              << (i + 1) << ","
              << (i < labels.size() ? labels[i] : "?") << ","
              << fixed << setprecision(4) << c.score << ","
              << setprecision(2) << c.spatial_indicator << ","
              << c.blur_indicator << ","
              << c.motion_indicator << ","
              << setprecision(3) << c.latency_s << ","
              << setprecision(0) << c.ms_decode << ","
              << c.ms_preprocess << ","
              << c.ms_backbone << ","
              << c.ms_deblur << ","
              << c.ms_flow << ","
              << c.ms_motion << ","
              << c.ms_quality << ","
              << c.rss_start << ","
              << c.rss_after_back << ","
              << c.rss_after_deblur << ","
              << c.rss_after_flow << ","
              << c.rss_after_motion << ","
              << bd << "," << dd << "," << fd << "," << md << ","
              << setprecision(4) << r.mean_score << ","
              << r.std_dev << ","
              << stability_label(r.mean_score) << "\n";
        }
}

void write_calibration_csv(const string &filename, const vector<VideoResult> &results)
{
    ofstream f(filename);
    f << "VideoName,Clip,MOS,SpatialRaw,BlurRaw,MotionRaw\n";

    float sp_min = 1e30f, sp_max = -1e30f;
    float bl_min = 1e30f, bl_max = -1e30f;
    float mo_min = 1e30f, mo_max = -1e30f;

    vector<string> labels = {"12.5%", "37.5%", "62.5%", "87.5%"};

    for (const auto &r : results)
        for (size_t i = 0; i < r.clips.size(); i++)
        {
            const auto &c = r.clips[i];
            f << r.video_name << ","
              << (i < labels.size() ? labels[i] : "?") << ","
              << fixed << setprecision(4) << c.score << ","
              << c.spatial_raw << ","
              << c.blur_raw << ","
              << c.motion_raw << "\n";

            sp_min = min(sp_min, c.spatial_raw);
            sp_max = max(sp_max, c.spatial_raw);
            bl_min = min(bl_min, c.blur_raw);
            bl_max = max(bl_max, c.blur_raw);
            mo_min = min(mo_min, c.motion_raw);
            mo_max = max(mo_max, c.motion_raw);
        }

    f << "\nSUMMARY"
      << "spatial_min=" << sp_min << " spatial_max=" << sp_max << ","
      << "blur_min=" << bl_min << " blur_max=" << bl_max << ","
      << "motion_min=" << mo_min << " motion_max=" << mo_max << "\n";

    cout << "\nCalibration Summary (" << results.size() << " videos)\n"
         << fixed << setprecision(4)
         << "  spatial_raw : min=" << sp_min << "  max=" << sp_max << "\n"
         << "  blur_raw    : min=" << bl_min << "  max=" << bl_max << "\n"
         << "  motion_raw  : min=" << mo_min << "  max=" << mo_max << "\n";
}

class ONNXRunner
{
    Ort::Env &env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    vector<string> in_str, out_str;
    vector<const char *> in_ptr, out_ptr;

public:
    ONNXRunner(Ort::Env &env, const string &path, int threads)
        : env(env), session(nullptr)
    {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetIntraOpNumThreads(threads);
        opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        // When branches run concurrently, spinning threads of a finished branch
        // would steal cores from the branches still running -> disable spinning.
        opts.AddConfigEntry("session.intra_op.allow_spinning",
                            USE_CONCURRENCY ? "0" : "1");
        session = Ort::Session(env, path.c_str(), opts);

        for (size_t i = 0; i < session.GetInputCount(); i++)
        {
            in_str.push_back(session.GetInputNameAllocated(i, allocator).get());
            in_ptr.push_back(in_str.back().c_str());
        }
        for (size_t i = 0; i < session.GetOutputCount(); i++)
        {
            out_str.push_back(session.GetOutputNameAllocated(i, allocator).get());
            out_ptr.push_back(out_str.back().c_str());
        }
    }

    pair<vector<float>, vector<int64_t>> forward(vector<float> &data, const vector<int64_t> &shape)
    {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value t = Ort::Value::CreateTensor<float>(
            mem, data.data(), data.size(), shape.data(), shape.size());
        auto outs = session.Run(Ort::RunOptions{nullptr},
                                in_ptr.data(), &t, 1,
                                out_ptr.data(), 1);
        float *arr = outs.front().GetTensorMutableData<float>();
        auto shp = outs.front().GetTensorTypeAndShapeInfo().GetShape();
        size_t cnt = outs.front().GetTensorTypeAndShapeInfo().GetElementCount();
        return {vector<float>(arr, arr + cnt), shp};
    }
};

void preprocess_dual(const Mat &frame, vector<float> &norm_buf, vector<float> &raw_buf)
{
    Mat resized, fmat;
    resize(frame, resized, Size(224, 224));
    cvtColor(resized, fmat, COLOR_BGR2RGB);
    fmat.convertTo(fmat, CV_32F);

    const float mean_v[] = {123.675f, 116.28f, 103.53f};
    const float std_v[] = {58.395f, 57.12f, 57.375f};
    const int pixels = 224 * 224;

    size_t base = norm_buf.size();
    norm_buf.resize(base + 3 * pixels);
    raw_buf.resize(base + 3 * pixels);
    float *np = norm_buf.data() + base;
    float *rp = raw_buf.data() + base;
    float *src = (float *)fmat.data;

    for (int i = 0; i < pixels; ++i)
        for (int c = 0; c < 3; ++c)
        {
            float val = src[i * 3 + c];
            rp[c * pixels + i] = val;
            np[c * pixels + i] = (val - mean_v[c]) / std_v[c];
        }
}

vector<float> make_backbone_input_like_evaluator(const vector<float> &clip,
                                                 int frames, int channels, int pixels)
{
    vector<float> out(frames * channels * pixels);
    for (int row = 0; row < frames; ++row)
        for (int out_c = 0; out_c < channels; ++out_c)
        {
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
                                             int frames, int channels, int pixels)
{
    const int frame_size = channels * pixels;
    vector<float> out;
    out.reserve(frames * 2 * frame_size);
    for (int t = 0; t < frames; ++t)
    {
        int next_t = min(t + 1, frames - 1);
        const float *f1 = clip.data() + t * frame_size;
        const float *f2 = clip.data() + next_t * frame_size;
        out.insert(out.end(), f1, f1 + frame_size);
        out.insert(out.end(), f2, f2 + frame_size);
    }
    return out;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "Usage: ./stablevqa <video_folder>" << endl;
        return -1;
    }

    string folder = argv[1];
    // Model directory: argv[2] if given, else a local ./onnx_models_int8/ .
    string dir = (argc >= 3) ? argv[2] : "onnx_models/";
    if (!dir.empty() && dir.back() != '/') dir += '/';

    vector<string> video_paths;
    vector<string> valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"};

    try
    {
        for (const auto &e : fs::directory_iterator(folder))
        {
            if (!e.is_regular_file())
                continue;
            string ext = e.path().extension().string();
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (find(valid_exts.begin(), valid_exts.end(), ext) != valid_exts.end())
                video_paths.push_back(e.path().string());
        }
    }
    catch (const fs::filesystem_error &e)
    {
        cerr << "Folder error: " << e.what() << endl;
        return -1;
    }

    if (video_paths.empty())
    {
        cerr << "No videos found in: " << folder << endl;
        return 0;
    }
    sort(video_paths.begin(), video_paths.end());

    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "StableVQA");

    try
    {
        cout << "Loading ONNX models from: " << dir << endl;
        cout << "  mode      : " << (USE_CONCURRENCY ? "CONCURRENT (backbone|deblur|flow)" : "SERIAL") << "\n";
        cout << "  threads   : trio=" << (USE_CONCURRENCY ? TRIO_THREADS : SERIAL_THREADS)
             << "  motion=" << MOTION_THREADS << "\n";
        cout << "  blur      : " << (USE_BLUR ? "ON" : "DISABLED (zeros)") << "\n";

        // Threads per branch. When concurrent, backbone/deblur/flow get TRIO_THREADS
        // each (they run at the same time); motion runs alone afterwards at MOTION_THREADS.
        const int trio_threads = USE_CONCURRENCY ? TRIO_THREADS : SERIAL_THREADS;

        ONNXRunner flow_model(env,      dir + "flow_model_quant.onnx",      trio_threads);
        ONNXRunner backbone_model(env,  dir + "backbone_quant.onnx",        trio_threads);
        ONNXRunner deblur_model(env,    dir + "deblur_net_quant.onnx",      trio_threads);
        ONNXRunner motion_analyzer(env, dir + "motion_analyzer_quant.onnx", MOTION_THREADS);
        ONNXRunner quality_head(env,    dir + "quality_head_quant.onnx",    HEAD_THREADS);

        cout << "Models loaded. Found " << video_paths.size() << " videos.\n"
             << endl;

        const int NUM_FRAMES = 32;
        const int FLOW_PAIRS = NUM_FRAMES;
        const int BLUR_STRIDE = 4;
        const int BLUR_FRAMES = 8;
        const int NUM_CLIPS = 4;
        const int pixels = 224 * 224;
        const int frame_size = 3 * pixels;

        vector<string> labels = {"12.5%", "37.5%", "62.5%", "87.5%"};
        vector<VideoResult> all_results;
        string csv_file = "stablevqa_results.csv";
        string calib_file = "stablevqa_calibration.csv";
        int vid_count = 0;

        // Warm-up: pay ORT graph init / MLAS weight-packing once, so the first
        cout << "Warm-up" << flush;
        {
            vector<float> zb(NUM_FRAMES * frame_size, 0.f);
            backbone_model.forward(zb, {NUM_FRAMES, 3, 224, 224});
            if (USE_BLUR) {
                vector<float> zd(BLUR_FRAMES * frame_size, 0.f);
                deblur_model.forward(zd, {BLUR_FRAMES, 3, 224, 224});
            }
            vector<float> zf(FLOW_PAIRS * 6 * pixels, 0.f);
            flow_model.forward(zf, {FLOW_PAIRS, 6, 224, 224});
            vector<float> zm(2 * NUM_FRAMES * pixels, 0.f);
            motion_analyzer.forward(zm, {1, 2, NUM_FRAMES, 224, 224});
            vector<float> zh(27648, 0.f);
            quality_head.forward(zh, {1, (int64_t)zh.size()});
        }
        cout << " done.\n" << endl;

        auto t_batch = chrono::high_resolution_clock::now();

        for (const string &vid_path : video_paths)
        {
            string vname = fs::path(vid_path).filename().string();
            cout << "[" << ++vid_count << "/" << video_paths.size() << "] " << vname << endl;

            VideoCapture cap(vid_path);
            if (!cap.isOpened())
            {
                cerr << "  Cannot open, skipping.\n";
                continue;
            }

            int total_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);
            int max_start = max(0, total_frames - 64);

            VideoResult vres;
            vres.video_name = vname;
            vector<float> all_scores;

            for (int clip_idx = 0; clip_idx < NUM_CLIPS; clip_idx++)
            {
                float pos = (clip_idx + 0.5f) / NUM_CLIPS;
                int start_frame = (int)(pos * max_start);
                cap.set(CAP_PROP_POS_FRAMES, start_frame);

                vector<float> norm_frames, raw_frames;
                norm_frames.reserve(NUM_FRAMES * frame_size);
                raw_frames.reserve(NUM_FRAMES * frame_size);

                int collected = 0;
                double ms_dec = 0.0, ms_prep = 0.0;

                for (int i = 0; i < 64 && collected < NUM_FRAMES; i++)
                {
                    if (i % 2 == 0)
                    {
                        Mat frame;
                        auto t0 = chrono::high_resolution_clock::now();
                        bool ok = cap.read(frame);
                        ms_dec += chrono::duration_cast<chrono::milliseconds>(
                                      chrono::high_resolution_clock::now() - t0)
                                      .count();
                        if (ok && !frame.empty())
                        {
                            auto t1 = chrono::high_resolution_clock::now();
                            preprocess_dual(frame, norm_frames, raw_frames);
                            ms_prep += chrono::duration_cast<chrono::milliseconds>(
                                           chrono::high_resolution_clock::now() - t1)
                                           .count();
                            collected++;
                        }
                    }
                    else
                    {
                        cap.grab();
                    }
                }

                if (collected == 0)
                {
                    cerr << "  Clip " << (clip_idx + 1) << ": no frames, skipping.\n";
                    continue;
                }

                while ((int)norm_frames.size() < NUM_FRAMES * frame_size)
                {
                    norm_frames.insert(norm_frames.end(),
                                       norm_frames.end() - frame_size, norm_frames.end());
                    raw_frames.insert(raw_frames.end(),
                                      raw_frames.end() - frame_size, raw_frames.end());
                }

                auto t_clip = chrono::high_resolution_clock::now();
                size_t rss0 = get_rss_kb();

                auto run_backbone = [&]() -> pair<vector<float>, double> {
                    vector<float> in = make_backbone_input_like_evaluator(
                        norm_frames, NUM_FRAMES, 3, pixels);
                    auto ta = chrono::high_resolution_clock::now();
                    auto out = backbone_model.forward(in, {NUM_FRAMES, 3, 224, 224}).first;
                    auto tb = chrono::high_resolution_clock::now();
                    return {std::move(out),
                            (double)chrono::duration_cast<chrono::milliseconds>(tb - ta).count()};
                };
                auto run_deblur = [&]() -> pair<vector<float>, double> {
                    if (!USE_BLUR)                          // blur disabled -> 2560 zeros
                        return {vector<float>(BLUR_FRAMES * 320, 0.0f), 0.0};
                    vector<float> batch;
                    batch.reserve(BLUR_FRAMES * frame_size);
                    for (int i = 0; i < NUM_FRAMES; i += BLUR_STRIDE) {
                        auto it = norm_frames.begin() + (i * frame_size);
                        batch.insert(batch.end(), it, it + frame_size);
                    }
                    auto ta = chrono::high_resolution_clock::now();
                    auto res = deblur_model.forward(batch, {BLUR_FRAMES, 3, 224, 224});
                    auto pooled = global_avg_pool(res.first, res.second);
                    auto tb = chrono::high_resolution_clock::now();
                    return {std::move(pooled),
                            (double)chrono::duration_cast<chrono::milliseconds>(tb - ta).count()};
                };
                auto run_flow = [&]() -> pair<vector<float>, double> {
                    vector<float> batch = make_flow_pairs_like_evaluator(
                        norm_frames, NUM_FRAMES, 3, pixels);
                    auto ta = chrono::high_resolution_clock::now();
                    auto out = flow_model.forward(batch, {FLOW_PAIRS, 6, 224, 224}).first;
                    auto tb = chrono::high_resolution_clock::now();
                    return {std::move(out),
                            (double)chrono::duration_cast<chrono::milliseconds>(tb - ta).count()};
                };

                pair<vector<float>, double> rb, rd, rf;
                if (USE_CONCURRENCY) {
                    auto fb = async(launch::async, run_backbone);
                    auto fd = async(launch::async, run_deblur);
                    auto ff = async(launch::async, run_flow);
                    rb = fb.get();
                    rd = fd.get();
                    rf = ff.get();
                } else {
                    rb = run_backbone();
                    rd = run_deblur();
                    rf = run_flow();
                }

                vector<float> b_feat   = std::move(rb.first);
                double        ms_back  = rb.second;
                size_t        rss1     = get_rss_kb();
                vector<float> d_feat   = std::move(rd.first);
                double        ms_debl  = rd.second;
                size_t        rss2     = get_rss_kb();
                vector<float> flow_res = std::move(rf.first);
                double        ms_flow  = rf.second;
                size_t        rss3     = get_rss_kb();

                if (flow_res.size() != (size_t)NUM_FRAMES * 2 * pixels)
                    throw runtime_error(
                        "flow_model output is not [32,2,224,224]; "
                        "re-export NeuFlow with FLOW_PAIRS=NUM_FRAMES");

                vector<float> m_input(2 * NUM_FRAMES * pixels);
                for (int k = 0; k < NUM_FRAMES; ++k)
                {
                    const float *src = flow_res.data() + (k * 2 * pixels);
                    memcpy(m_input.data() + k * pixels,
                           src, pixels * sizeof(float));
                    memcpy(m_input.data() + NUM_FRAMES * pixels + k * pixels,
                           src + pixels, pixels * sizeof(float));
                }
                auto t4s = chrono::high_resolution_clock::now();
                auto m_feat = motion_analyzer.forward(
                                                 m_input, {1, 2, NUM_FRAMES, 224, 224})
                                  .first;
                auto t4e = chrono::high_resolution_clock::now();
                double ms_motion = chrono::duration_cast<chrono::milliseconds>(t4e - t4s).count();
                size_t rss4 = get_rss_kb();

                float spatial_raw = mean_abs(b_feat);
                float blur_raw = mean_abs(d_feat);
                float motion_raw = mean_abs(m_feat);

                float s_spatial, s_blur, s_motion;
                compute_branch_indicators(spatial_raw, blur_raw, motion_raw,
                                          s_spatial, s_blur, s_motion);

                vector<float> total_feat;
                total_feat.reserve(d_feat.size() + b_feat.size() + m_feat.size());
                total_feat.insert(total_feat.end(), d_feat.begin(), d_feat.end());
                total_feat.insert(total_feat.end(), b_feat.begin(), b_feat.end());
                total_feat.insert(total_feat.end(), m_feat.begin(), m_feat.end());

                auto t5s = chrono::high_resolution_clock::now();
                auto score_res = quality_head.forward(
                    total_feat, {1, (int64_t)total_feat.size()});
                auto t5e = chrono::high_resolution_clock::now();
                double ms_qual = chrono::duration_cast<chrono::milliseconds>(t5e - t5s).count();

                float raw_score = score_res.first[0];
                float score = map_to_mos(raw_score);

                all_scores.push_back(score);

                double total_s = chrono::duration_cast<chrono::milliseconds>(
                                     chrono::high_resolution_clock::now() - t_clip)
                                     .count() /
                                 1000.0;

                cout << fixed << setprecision(2)
                     << "  Clip " << (clip_idx + 1) << " [" << labels[clip_idx] << "]"
                     << "  MOS=" << score
                     << "  SpatialIndicator=" << s_spatial
                     << "  BlurIndicator=" << s_blur
                     << "  MotionIndicator=" << s_motion
                     << "  Time=" << total_s << "s" << endl;

                ClipMetrics cm;
                cm.score = score;
                cm.latency_s = total_s;
                cm.ms_decode = ms_dec;
                cm.ms_preprocess = ms_prep;
                cm.ms_backbone = ms_back;
                cm.ms_deblur = ms_debl;
                cm.ms_flow = ms_flow;
                cm.ms_motion = ms_motion;
                cm.ms_quality = ms_qual;
                cm.rss_start = rss0;
                cm.rss_after_back = rss1;
                cm.rss_after_deblur = rss2;
                cm.rss_after_flow = rss3;
                cm.rss_after_motion = rss4;
                cm.spatial_indicator = s_spatial;
                cm.blur_indicator = s_blur;
                cm.motion_indicator = s_motion;
                cm.spatial_raw = spatial_raw;
                cm.blur_raw = blur_raw;
                cm.motion_raw = motion_raw;
                vres.clips.push_back(cm);
            }

            if (!all_scores.empty())
            {
                float mean = accumulate(all_scores.begin(), all_scores.end(), 0.0f) / all_scores.size();
                float var = 0.0f;
                for (float s : all_scores)
                    var += (s - mean) * (s - mean);
                float std_dev = sqrt(var / all_scores.size());

                vres.mean_score = mean;
                vres.std_dev = std_dev;
                all_results.push_back(vres);

                cout << "\n  Mean MOS : " << fixed << setprecision(2) << mean
                     << "  (" << stability_label(mean) << ")"
                     << "  StdDev: " << std_dev << "\n"
                     << endl;

                write_csv(csv_file, all_results);
            }
        }

        double total_s = chrono::duration_cast<chrono::milliseconds>(
                             chrono::high_resolution_clock::now() - t_batch)
                             .count() /
                         1000.0;
        cout << "Done. " << vid_count << " videos in " << total_s << "s.\n";
        cout << "Results     : " << csv_file << "\n";

        write_calibration_csv(calib_file, all_results);
        cout << "Calibration : " << calib_file << "\n";
    }
    catch (const exception &e)
    {
        cerr << "Fatal: " << e.what() << endl;
        return -1;
    }
    return 0;
}
