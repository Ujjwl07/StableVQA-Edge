#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "psapi.lib")
#endif

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "cpu_tuning.h"
using namespace std;
using namespace cv;

static int    g_intra_threads = 0;
static string g_profile       = "default";

static void applyCpuTuning(const CpuInfo &ci)
{
    Tuning t = decideTuning(ci, std::getenv("SVQA_THREADS"));
    g_intra_threads = t.threads;
    g_profile       = t.profile;

#ifdef _WIN32
    if (t.affinity != 0)
    {
        DWORD_PTR sysMask = 0, procMask = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask))
            t.affinity &= (uint64_t)sysMask;         
        if (t.affinity != 0)
            SetProcessAffinityMask(GetCurrentProcess(), (DWORD_PTR)t.affinity);
    }
#endif

    cout << "CPU: " << (ci.brand.empty() ? ci.vendor : ci.brand) << "\n"
         << "  vendor=" << ci.vendor
         << "  physical=" << ci.physical
         << "  logical="  << ci.logical
         << "  hybrid="   << (ci.hybrid ? "yes" : "no");
    if (ci.hybrid) cout << "  P-cores=" << ci.pPhysical;
    cout << "\n  profile=" << g_profile
         << "  intra_op_threads=" << g_intra_threads;
#ifdef _WIN32
    {
        const char *spinEnv = std::getenv("SVQA_ALLOW_SPINNING");
        bool allowSpin = spinEnv && atoi(spinEnv) != 0;
        cout << "  spinning=" << (allowSpin ? "on (SVQA_ALLOW_SPINNING=1)" : "off");
    }
#endif
    cout << "\n" << endl;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

bool fileExists(const std::string &name)
{
    ifstream f(name.c_str());
    return f.good();
}

float map_to_mos(float raw)
{
    const double b1 = 92.91053891;
    const double b2 = 0.35856351;
    const double b3 = -2.3526431;
    const double b4 = 63.48772513;
    double z = b2 * (raw - b3);
    z = std::max(-60.0, std::min(60.0, z));
    double mos = b1 * (0.5 - 1.0 / (1.0 + std::exp(z))) + b4;
    return (float)mos;
}

vector<float> global_avg_pool(const vector<float> &data, const vector<int64_t> &shape)
{
    if (shape.size() != 4)
        return data;
    int batch = shape[0], channels = shape[1],
        spatial = shape[2] * shape[3];
    vector<float> pooled;
    pooled.reserve(batch * channels);
    for (int b = 0; b < batch; ++b)
    {
        for (int c = 0; c < channels; ++c)
        {
            const float *ptr = &data[(b * channels * spatial) + (c * spatial)];
            float sum = 0.0f;
            for (int k = 0; k < spatial; ++k)
                sum += ptr[k];
            pooled.push_back(sum / spatial);
        }
    }
    return pooled;
}

class ONNXRunner
{
    Ort::Env &env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    vector<const char *> input_names;
    vector<const char *> output_names;
    vector<string> input_names_str;
    vector<string> output_names_str;

public:
    ONNXRunner(Ort::Env &env, const string &model_path)
        : env(env), session(nullptr)
    {
        if (!fileExists(model_path))
            throw std::runtime_error("Model file not found: " + model_path);

        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        int threads = g_intra_threads > 0 ? g_intra_threads : 8;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.EnableCpuMemArena();

#ifdef _WIN32
        {
            const char *spinEnv = std::getenv("SVQA_ALLOW_SPINNING");
            bool allowSpin = spinEnv && atoi(spinEnv) != 0;
            options.AddConfigEntry("session.intra_op.allow_spinning", allowSpin ? "1" : "0");
        }
#else
        options.AddConfigEntry("session.intra_op.allow_spinning", "1");
#endif

#ifdef _WIN32
        int size_needed = MultiByteToWideChar(
            CP_UTF8, 0, &model_path[0], (int)model_path.size(), NULL, 0);
        std::wstring wstrTo(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, &model_path[0],
                            (int)model_path.size(), &wstrTo[0], size_needed);
        session = Ort::Session(env, wstrTo.c_str(), options);
#else
        session = Ort::Session(env, model_path.c_str(), options);
#endif

        for (size_t i = 0; i < session.GetInputCount(); i++)
        {
            input_names_str.push_back(session.GetInputNameAllocated(i, allocator).get());
            input_names.push_back(input_names_str.back().c_str());
        }
        for (size_t i = 0; i < session.GetOutputCount(); i++)
        {
            output_names_str.push_back(session.GetOutputNameAllocated(i, allocator).get());
            output_names.push_back(output_names_str.back().c_str());
        }
    }

    pair<vector<float>, vector<int64_t>> forward(vector<float> &input_data, const vector<int64_t> &shape)
    {
        auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(mem, input_data.data(), input_data.size(), shape.data(), shape.size());
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
        float *f = output_tensors.front().GetTensorMutableData<float>();
        auto out_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
        size_t count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        return {vector<float>(f, f + count), out_shape};
    }
};

void preprocess_dual(const Mat &frame, vector<float> &norm_buf, vector<float> &raw_buf)
{
    Mat resized, float_mat;
    resize(frame, resized, Size(224, 224));
    cvtColor(resized, float_mat, COLOR_BGR2RGB);
    float_mat.convertTo(float_mat, CV_32F);

    const float mean[] = {123.675f, 116.28f, 103.53f};
    const float std[] = {58.395f, 57.12f, 57.375f};
    int pixels = 224 * 224;

    size_t start_idx = norm_buf.size();
    norm_buf.resize(start_idx + 3 * pixels);
    raw_buf.resize(start_idx + 3 * pixels);
    float *norm_ptr = norm_buf.data() + start_idx;
    float *raw_ptr = raw_buf.data() + start_idx;
    float *src = (float *)float_mat.data;

    for (int i = 0; i < pixels; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            float val = src[i * 3 + c];
            raw_ptr[c * pixels + i] = val;
            norm_ptr[c * pixels + i] = (val - mean[c]) / std[c];
        }
    }
}

vector<float> make_backbone_input_like_evaluator(const vector<float> &clip, int frames, int channels, int pixels)
{
    vector<float> out(frames * channels * pixels);

    for (int row = 0; row < frames; ++row)
    {
        for (int out_c = 0; out_c < channels; ++out_c)
        {
            int q = row * channels + out_c;
            int src_c = q / frames;
            int src_t = q % frames;

            const float *src = clip.data() + (src_t * channels + src_c) * pixels;
            float *dst = out.data() + (row * channels + out_c) * pixels;
            memcpy(dst, src, pixels * sizeof(float));
        }
    }

    return out;
}

vector<float> make_flow_pairs_like_evaluator(const vector<float> &clip, int frames, int channels, int pixels)
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

bool extract_clip(VideoCapture &cap, int start_frame, vector<float> &norm_frames, vector<float> &raw_frames, double &frame_load_ms, double &preprocess_ms)
{
    norm_frames.clear();
    raw_frames.clear();
    frame_load_ms = 0.0;
    preprocess_ms = 0.0;
    int frame_size = 3 * 224 * 224;
    norm_frames.reserve(32 * frame_size);
    raw_frames.reserve(32 * frame_size);

    cap.set(CAP_PROP_POS_FRAMES, start_frame);
    Mat f;
    int collected = 0;

    for (int i = 0; i < 64 && collected < 32; i++)
    {
        if (i % 2 == 0)
        {
            auto t0 = chrono::high_resolution_clock::now();
            cap >> f;
            auto t1 = chrono::high_resolution_clock::now();
            if (f.empty())
                break;
            preprocess_dual(f, norm_frames, raw_frames);
            auto t2 = chrono::high_resolution_clock::now();
            frame_load_ms += chrono::duration_cast<chrono::microseconds>(t1 - t0).count() / 1000.0;
            preprocess_ms += chrono::duration_cast<chrono::microseconds>(t2 - t1).count() / 1000.0;
            collected++;
        }
        else
        {
            cap.grab();
        }
    }

    if (collected == 0)
        return false;

    while (norm_frames.size() < (size_t)(32 * frame_size) && !norm_frames.empty())
    {
        norm_frames.insert(norm_frames.end(), norm_frames.end() - frame_size, norm_frames.end());
        raw_frames.insert(raw_frames.end(), raw_frames.end() - frame_size, raw_frames.end());
    }
    return true;
}

struct InferenceMetrics
{
    float score;
    double total_ms;
    double frame_load_ms;
    double preprocess_ms;
    double backbone_ms;
    double deblur_ms;
    double flow_ms;
    double motion_ms;
    double quality_head_ms;
    size_t rss_start_kb;
    size_t rss_after_backbone_kb;
    size_t rss_after_deblur_kb;
    size_t rss_after_flow_kb;
    size_t rss_after_motion_kb;
};

#ifdef _WIN32
size_t getCurrentRSS()
{
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.WorkingSetSize / 1024;
    return 0;
}
#else
size_t getCurrentRSS() { return 0; }
#endif

InferenceMetrics run_inference_with_metrics(vector<float> &norm_frames, vector<float> &raw_frames, ONNXRunner &flow_model, ONNXRunner &backbone_model, ONNXRunner &deblur_model, ONNXRunner &motion_analyzer, ONNXRunner &quality_head)
{
    InferenceMetrics metrics = {};
    const int NUM_FRAMES = 32;
    const int CHANNELS = 3;
    const int FLOW_PAIRS = NUM_FRAMES;
    int frame_size = CHANNELS * 224 * 224;
    int pixels = 224 * 224;
    (void)raw_frames;

    auto t_total_start = chrono::high_resolution_clock::now();
    metrics.rss_start_kb = getCurrentRSS();

    auto t_backbone_start = chrono::high_resolution_clock::now();
    vector<float> backbone_input = make_backbone_input_like_evaluator(
        norm_frames, NUM_FRAMES, CHANNELS, pixels);
    vector<float> b_out = backbone_model.forward(backbone_input, {NUM_FRAMES, CHANNELS, 224, 224}).first;
    auto t_backbone_end = chrono::high_resolution_clock::now();
    metrics.backbone_ms = chrono::duration_cast<chrono::microseconds>(t_backbone_end - t_backbone_start).count() / 1000.0;
    metrics.rss_after_backbone_kb = getCurrentRSS();

    auto t_deblur_start = chrono::high_resolution_clock::now();
    vector<float> batch;
    batch.reserve(8 * frame_size);
    for (int i = 0; i < NUM_FRAMES; i += 4)
    {
        auto start = norm_frames.begin() + (i * frame_size);
        batch.insert(batch.end(), start, start + frame_size);
    }
    while (batch.size() / frame_size < 8)
        batch.insert(batch.end(), batch.end() - frame_size, batch.end());
    auto deblur_res = deblur_model.forward(batch, {8, CHANNELS, 224, 224});
    vector<float> d_out = global_avg_pool(deblur_res.first, deblur_res.second);
    auto t_deblur_end = chrono::high_resolution_clock::now();
    metrics.deblur_ms = chrono::duration_cast<chrono::microseconds>(t_deblur_end - t_deblur_start).count() / 1000.0;
    metrics.rss_after_deblur_kb = getCurrentRSS();

    auto t_flow_start = chrono::high_resolution_clock::now();
    vector<float> batch_input = make_flow_pairs_like_evaluator(
        norm_frames, NUM_FRAMES, CHANNELS, pixels);
    auto flow_res = flow_model.forward(batch_input, {FLOW_PAIRS, 6, 224, 224}).first;
    if (flow_res.size() != (size_t)NUM_FRAMES * 2 * pixels)
        throw runtime_error("flow_model output is not [32,2,224,224]; re-export NeuFlow with FLOW_PAIRS=NUM_FRAMES");
    auto t_flow_end = chrono::high_resolution_clock::now();
    metrics.flow_ms = chrono::duration_cast<chrono::microseconds>(t_flow_end - t_flow_start).count() / 1000.0;
    metrics.rss_after_flow_kb = getCurrentRSS();

    auto t_motion_start = chrono::high_resolution_clock::now();
    vector<float> m_input(2 * NUM_FRAMES * pixels);
    for (int k = 0; k < NUM_FRAMES; ++k)
    {
        const float *src = flow_res.data() + (k * 2 * pixels);
        memcpy(m_input.data() + k * pixels, src, pixels * sizeof(float));
        memcpy(m_input.data() + NUM_FRAMES * pixels + k * pixels, src + pixels, pixels * sizeof(float));
    }
    vector<float> m_out = motion_analyzer.forward(m_input, {1, 2, NUM_FRAMES, 224, 224}).first;
    auto t_motion_end = chrono::high_resolution_clock::now();
    metrics.motion_ms = chrono::duration_cast<chrono::microseconds>(t_motion_end - t_motion_start).count() / 1000.0;
    metrics.rss_after_motion_kb = getCurrentRSS();

    auto t_head_start = chrono::high_resolution_clock::now();
    vector<float> fused;
    fused.reserve(d_out.size() + b_out.size() + m_out.size());
    fused.insert(fused.end(), d_out.begin(), d_out.end());
    fused.insert(fused.end(), b_out.begin(), b_out.end());
    fused.insert(fused.end(), m_out.begin(), m_out.end());
    auto head_res = quality_head.forward(fused, {1, (int64_t)fused.size()});
    auto t_head_end = chrono::high_resolution_clock::now();
    metrics.quality_head_ms = chrono::duration_cast<chrono::microseconds>(t_head_end - t_head_start).count() / 1000.0;

    auto t_total_end = chrono::high_resolution_clock::now();
    metrics.total_ms = chrono::duration_cast<chrono::microseconds>(t_total_end - t_total_start).count() / 1000.0;
    float raw_score = head_res.first[0];
    metrics.score = map_to_mos(raw_score);

    return metrics;
}

void write_csv_header(ofstream &csv)
{
    csv << "VideoName,Clip,StartPos,Score,TotalLatency(ms),FrameLoad(ms),Preprocess(ms),"
        << "Backbone(ms),Deblur(ms),Flow(ms),Motion(ms),QualityHead(ms),"
        << "RSS_Start(KB),RSS_AfterBackbone(KB),RSS_AfterDeblur(KB),"
        << "RSS_AfterFlow(KB),RSS_AfterMotion(KB),"
        << "BackboneRAMDelta(KB),DeblurRAMDelta(KB),FlowRAMDelta(KB),MotionRAMDelta(KB),"
        << "VideoMeanScore,VideoStdDev\n";
}

void append_csv_row(ofstream &csv, const string &video_name, int clip_idx, const string &pos_label, const InferenceMetrics &m, float video_mean, float video_std)
{
    csv << video_name << ","
        << (clip_idx + 1) << ","
        << pos_label << ","
        << m.score << ","
        << m.total_ms << ","
        << m.frame_load_ms << ","
        << m.preprocess_ms << ","
        << m.backbone_ms << ","
        << m.deblur_ms << ","
        << m.flow_ms << ","
        << m.motion_ms << ","
        << m.quality_head_ms << ","
        << m.rss_start_kb << ","
        << m.rss_after_backbone_kb << ","
        << m.rss_after_deblur_kb << ","
        << m.rss_after_flow_kb << ","
        << m.rss_after_motion_kb << ","
        << ((long long)m.rss_after_backbone_kb - (long long)m.rss_start_kb) << ","
        << ((long long)m.rss_after_deblur_kb - (long long)m.rss_after_backbone_kb) << ","
        << ((long long)m.rss_after_flow_kb - (long long)m.rss_after_deblur_kb) << ","
        << ((long long)m.rss_after_motion_kb - (long long)m.rss_after_flow_kb) << ","
        << video_mean << ","
        << video_std << "\n";
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cout << "Usage: ./stablevqa <video_list.txt> [output.csv]" << endl;
        return -1;
    }

    string csv_path = (argc >= 3) ? argv[2] : "stablevqa_results.csv";

    try
    {
#ifdef _WIN32
        timeBeginPeriod(1);
#endif
        CpuInfo cpu = detectCpu();
        applyCpuTuning(cpu);

        auto t_start_total = chrono::high_resolution_clock::now();

        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "StableVQA_CPU");
        string d = "onnx_models/";

        cout << "Loading quantized CPU models" << endl;
        ONNXRunner flow_model(env, d + "flow_model_quant.onnx");
        ONNXRunner backbone_model(env, d + "backbone_quant.onnx");
        ONNXRunner deblur_model(env, d + "deblur_net_quant.onnx");
        ONNXRunner motion_analyzer(env, d + "motion_analyzer_quant.onnx");
        ONNXRunner quality_head(env, d + "quality_head_quant.onnx");

        auto t_loaded = chrono::high_resolution_clock::now();
        double load_time = chrono::duration_cast<chrono::milliseconds>(t_loaded - t_start_total).count() / 1000.0;
        cout << "Models loaded in " << load_time << "s\n" << endl;

        {
            vector<float> zb(32 * 3 * 224 * 224, 0.f);
            vector<float> zd(8 * 3 * 224 * 224, 0.f);
            vector<float> zf(32 * 6 * 224 * 224, 0.f);
            try { backbone_model.forward(zb, {32, 3, 224, 224}); } catch (...) {}
            try { deblur_model.forward(zd, {8, 3, 224, 224}); }    catch (...) {}
            try { flow_model.forward(zf, {32, 6, 224, 224}); }     catch (...) {}
        }

        ifstream infile(argv[1]);
        if (!infile.is_open())
        {
            cerr << "Cannot open video list file: " << argv[1] << endl;
            return -1;
        }

        int total_videos = 0;
        {
            string tmp;
            while (getline(infile, tmp))
                if (!tmp.empty())
                    total_videos++;
            infile.clear();
            infile.seekg(0);
        }
        cerr << "Found " << total_videos << " videos to process\n" << endl;

        ofstream csv_file(csv_path);
        if (!csv_file.is_open())
        {
            cerr << "Cannot create CSV file: " << csv_path << endl;
            return -1;
        }
        write_csv_header(csv_file);
        cout << "Writing results to: " << csv_path << "\n" << endl;

        int num_clips = 4;
        vector<string> pos_labels = {"12.5%", "37.5%", "62.5%", "87.5%"};

        string video_path;
        int video_count = 0;

        while (getline(infile, video_path))
        {
            if (video_path.empty())
                continue;
            video_count++;

            string video_name = video_path;
            size_t last_slash = video_path.find_last_of("/\\");
            if (last_slash != string::npos)
                video_name = video_path.substr(last_slash + 1);

            VideoCapture cap(video_path);
            if (!cap.isOpened())
                cap.open(video_path, CAP_MSMF);
            if (!cap.isOpened())
            {
                cerr << "Skipping (cannot open): " << video_name << endl;
                continue;
            }

            int total = (int)cap.get(CAP_PROP_FRAME_COUNT);
            int max_start = max(0, total - 64);

            vector<float> clip_scores;
            vector<InferenceMetrics> all_metrics;

            for (int ci = 0; ci < num_clips; ci++)
            {
                float pos = (ci + 0.5f) / num_clips;
                int start_frame = (int)(pos * max_start);

                vector<float> norm_frames, raw_frames;
                double clip_frame_load_ms = 0.0, clip_preprocess_ms = 0.0;

                if (!extract_clip(cap, start_frame, norm_frames, raw_frames, clip_frame_load_ms, clip_preprocess_ms))
                {
                    cerr << "Clip extraction failed: " << video_name << " clip " << (ci + 1) << endl;
                    continue;
                }

                InferenceMetrics m = run_inference_with_metrics(
                    norm_frames, raw_frames,
                    flow_model, backbone_model,
                    deblur_model, motion_analyzer,
                    quality_head);

                m.frame_load_ms = clip_frame_load_ms;
                m.preprocess_ms = clip_preprocess_ms;
                m.total_ms += clip_frame_load_ms + clip_preprocess_ms;

                clip_scores.push_back(m.score);
                all_metrics.push_back(m);
            }

            float mean = 0.0f, std_dev = 0.0f;
            if (!clip_scores.empty())
            {
                mean = accumulate(clip_scores.begin(), clip_scores.end(), 0.0f) / clip_scores.size();
                float variance = 0.0f;
                for (float s : clip_scores)
                    variance += (s - mean) * (s - mean);
                std_dev = sqrt(variance / clip_scores.size());
            }

            cerr << "[" << video_count << "/" << total_videos << "] " << video_name << ": mean score: " << mean << endl;

            for (size_t ci = 0; ci < all_metrics.size(); ci++)
            {
                append_csv_row(csv_file, video_name, (int)ci, pos_labels[ci], all_metrics[ci], mean, std_dev);
            }
            csv_file.flush();
        }

        csv_file.close();
        cout << "\nAll done. Results saved to: " << csv_path << endl;

        auto t_end_total = chrono::high_resolution_clock::now();
        double total_time = chrono::duration_cast<chrono::milliseconds>(t_end_total - t_start_total).count() / 1000.0;
        cout << "Total videos: " << video_count << "  Total time: " << total_time << "s" << endl;
    }
    catch (const std::exception &e)
    {
        cerr << "\nFatal error: " << e.what() << endl;
        return -1;
    }
    return 0;
}
