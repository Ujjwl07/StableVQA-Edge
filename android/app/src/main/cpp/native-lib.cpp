#include <jni.h>
#include <string>
#include <vector>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sched.h>
#include <stdexcept>
#include <unordered_map>
#include <future>

#include <onnxruntime_cxx_api.h>

#define LOG_TAG "StableVQA_ONNX"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static const bool USE_INT8 = true;

static const int PERF_CORE_LO = 4;
static const int PERF_CORE_HI = 9;
static const int INTRA_OP_THREADS = 6;

static const bool USE_CONCURRENCY = false;
static const int  CONCURRENT_TRIO_THREADS = 2;   
static const int  MOTION_THREADS = 6;  

static void pin_to_performance_cores()
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    for (int i = PERF_CORE_LO; i <= PERF_CORE_HI; i++) CPU_SET(i, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}

std::vector<float> global_avg_pool(const float *data,
                                   const std::vector<int64_t> &shape)
{
    if (shape.size() != 4) return {};
    int batch    = (int)shape[0];
    int channels = (int)shape[1];
    int spatial  = (int)(shape[2] * shape[3]);
    std::vector<float> pooled;
    pooled.reserve(static_cast<size_t>(batch * channels));
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < channels; ++c)
        {
            const float *ptr = &data[(b * channels + c) * spatial];
            float sum = 0.f;
            for (int k = 0; k < spatial; ++k) sum += ptr[k];
            pooled.push_back(sum / static_cast<float>(spatial));
        }
    return pooled;
}

float map_to_mos(float raw)
{
    const double b1 = 92.91053891;
    const double b2 = 0.35856351;
    const double b3 = -2.3526431;
    const double b4 = 63.48772513;
    double z = b2 * (raw - b3);
    z = std::max(-60.0, std::min(60.0, z));
    return (float)(b1 * (0.5 - 1.0 / (1.0 + std::exp(z))) + b4);
}

std::string stability_label(float mos)
{
    if (mos <= 20) return "Very Unstable";
    if (mos <= 40) return "Unstable";
    if (mos <= 60) return "Moderately Stable";
    if (mos <= 80) return "Stable";
    return "Highly Stable";
}

static float mean_abs(const std::vector<float> &v)
{
    if (v.empty()) return 0.f;
    double sum = 0.0;
    for (float x : v) sum += std::abs((double)x);
    return (float)(sum / v.size());
}

static float clamp01(float x)
{
    return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

static float normalize_score(float x, float xmin, float xmax)
{
    if (xmax <= xmin) return 50.f;
    return clamp01((x - xmin) / (xmax - xmin)) * 100.f;
}

static void compute_branch_indicators(float spatial_raw,
                                      float blur_raw,
                                      float motion_raw,
                                      float &out_spatial,
                                      float &out_blur,
                                      float &out_motion)
{
    out_spatial = 100.f - normalize_score(spatial_raw, 0.1479f, 0.4086f);
    out_blur    = 100.f - normalize_score(blur_raw,    0.5129f, 2.8472f);
    out_motion  = 100.f - normalize_score(motion_raw,  0.2580f, 3.6100f);
}

static long readNativeRssKb()
{
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
    {
        if (line.rfind("VmRSS:", 0) == 0)
        {
            const char *p = line.c_str() + 6;
            char *end = nullptr;
            long kb = std::strtol(p, &end, 10);
            return (end != p) ? kb : 0;
        }
    }
    return 0;
}

std::string extractAsset(JNIEnv *env, jobject assetManager,
                         const std::string &filename,
                         const std::string &cacheDir)
{
    AAssetManager *mgr   = AAssetManager_fromJava(env, assetManager);
    AAsset        *asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_BUFFER);
    if (!asset) return "";

    std::string outPath = cacheDir + "/" + filename;
    {
        std::ifstream test(outPath, std::ios::binary | std::ios::ate);
        off_t assetLen = AAsset_getLength(asset);
        if (test && test.tellg() == (std::streampos)assetLen)
        {
            AAsset_close(asset);
            return outPath;
        }
    }

    off_t length = AAsset_getLength(asset);
    std::vector<char> buf(length);
    AAsset_read(asset, buf.data(), length);
    AAsset_close(asset);

    std::ofstream out(outPath, std::ios::binary);
    out.write(buf.data(), length);
    LOGI("Extracted %s (%ld bytes)", filename.c_str(), (long)length);
    return outPath;
}

struct OrtContext
{
    Ort::Env         env{ORT_LOGGING_LEVEL_WARNING, "StableVQA"};
    Ort::MemoryInfo  memory_info =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::unique_ptr<Ort::Session> backbone;
    std::unique_ptr<Ort::Session> deblur;
    std::unique_ptr<Ort::Session> flow;
    std::unique_ptr<Ort::Session> motion;
    std::unique_ptr<Ort::Session> quality;

    Ort::SessionOptions makeOptions(bool use_xnnpack, int threads)
    {
        Ort::SessionOptions o;
        o.SetIntraOpNumThreads(threads);
        o.SetInterOpNumThreads(1);
        o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        o.EnableMemPattern();
        o.EnableCpuMemArena();
        o.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        o.AddConfigEntry("session.intra_op.allow_spinning", "0");
        o.AddConfigEntry("session.inter_op.allow_spinning", "0");
        if (use_xnnpack)
        {
            std::unordered_map<std::string, std::string> xnn;
            xnn["intra_op_num_threads"] = std::to_string(threads);
            o.AppendExecutionProvider("XNNPACK", xnn);
        }
        return o;
    }
};

static std::vector<float> make_backbone_input_like_evaluator(
        const std::vector<float> &clip, int frames, int channels, int pixels)
{
    std::vector<float> out(frames * channels * pixels);
    for (int row = 0; row < frames; ++row)
        for (int out_c = 0; out_c < channels; ++out_c)
        {
            int q     = row * channels + out_c;
            int src_c = q / frames;
            int src_t = q % frames;
            const float *src = clip.data() + (src_t * channels + src_c) * pixels;
            float       *dst = out.data()  + (row   * channels + out_c) * pixels;
            std::memcpy(dst, src, pixels * sizeof(float));
        }
    return out;
}

static std::vector<float> make_flow_pairs_like_evaluator(
        const std::vector<float> &clip, int frames, int channels, int pixels)
{
    const int frame_size = channels * pixels;
    std::vector<float> out;
    out.reserve(frames * 2 * frame_size);
    for (int t = 0; t < frames; ++t)
    {
        int next_t = std::min(t + 1, frames - 1);
        const float *f1 = clip.data() + t      * frame_size;
        const float *f2 = clip.data() + next_t * frame_size;
        out.insert(out.end(), f1, f1 + frame_size);
        out.insert(out.end(), f2, f2 + frame_size);
    }
    return out;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_example_stablevqa_MainActivity_initStableVQAModels(
        JNIEnv *env, jobject,
        jobject assetManager, jstring cachePathStr)
{
    const char *ch = env->GetStringUTFChars(cachePathStr, nullptr);
    std::string cacheDir(ch);
    env->ReleaseStringUTFChars(cachePathStr, ch);

    auto *ctx = new OrtContext();

    auto load = [&](const std::string &name,
                    const std::string &dataFile,
                    std::unique_ptr<Ort::Session> &sess,
                    int threads) -> bool
    {
        std::string p = extractAsset(env, assetManager, name + ".onnx", cacheDir);
        if (p.empty()) { LOGE("Asset not found: %s.onnx", name.c_str()); return false; }

        if (!dataFile.empty())
        {
            std::string dp = extractAsset(env, assetManager, dataFile, cacheDir);
            if (dp.empty()) { LOGE("External data not found: %s", dataFile.c_str()); return false; }
        }

        try
        {
            auto opts = ctx->makeOptions(/*use_xnnpack=*/false, threads);
            sess = std::make_unique<Ort::Session>(ctx->env, p.c_str(), opts);
            LOGI("Loaded: %s (XNNPACK, %d thr)", name.c_str(), threads);
            return true;
        }
        catch (const Ort::Exception &e1)
        {
            LOGE("XNNPACK could not build %s (%s); retrying on CPU EP.",
                 name.c_str(), e1.what());
            try
            {
                auto opts = ctx->makeOptions(/*use_xnnpack=*/false, threads);
                sess = std::make_unique<Ort::Session>(ctx->env, p.c_str(), opts);
                LOGI("Loaded: %s (CPU EP, %d thr)", name.c_str(), threads);
                return true;
            }
            catch (const Ort::Exception &e2)
            {
                LOGE("Failed to load %s: %s", name.c_str(), e2.what());
                return false;
            }
        }
    };

    const int trio_threads = USE_CONCURRENCY ? CONCURRENT_TRIO_THREADS : INTRA_OP_THREADS;

    bool ok = true;
    if (USE_INT8)
    {
        LOGI("Loading INT8 quantized models. Concurrency=%d trio_threads=%d",
             (int)USE_CONCURRENCY, trio_threads);
        ok &= load("backbone_quant",        "", ctx->backbone, trio_threads);
        ok &= load("deblur_net_quant",      "", ctx->deblur,   trio_threads);
        ok &= load("flow_model_quant",      "", ctx->flow,     trio_threads);
        ok &= load("quality_head_quant",    "", ctx->quality,  1);
        ok &= load("motion_analyzer_quant", "", ctx->motion,   MOTION_THREADS);
    }
    else
    {
        LOGI("Loading FP32 prep models. Concurrency=%d trio_threads=%d",
             (int)USE_CONCURRENCY, trio_threads);
        ok &= load("backbone_prep",        "d2ef64f4-4317-11f1-92a4-ba4ad40ab4c2.data", ctx->backbone, trio_threads);
        ok &= load("deblur_net_prep",      "ea79b246-4317-11f1-92a4-ba4ad40ab4c2.data", ctx->deblur,   trio_threads);
        ok &= load("flow_model_prep",      "c718d6b0-4317-11f1-92a4-ba4ad40ab4c2.data", ctx->flow,     trio_threads);
        ok &= load("quality_head_fp32",    "",                                            ctx->quality,  1);
        ok &= load("motion_analyzer_prep", "ef642232-4317-11f1-92a4-ba4ad40ab4c2.data", ctx->motion,   MOTION_THREADS);
    }

    if (!ok) { delete ctx; return 0; }

    LOGI("Running warmup pass");
    try
    {
        const char *ib[] = {"input"};
        const char *ob[] = {"output"};
        auto run = [&](std::unique_ptr<Ort::Session> &s,
                       std::vector<float> &d, std::vector<int64_t> &sh)
        {
            auto t = Ort::Value::CreateTensor<float>(
                    ctx->memory_info, d.data(), d.size(), sh.data(), sh.size());
            s->Run(Ort::RunOptions{nullptr}, ib, &t, 1, ob, 1);
        };
        { std::vector<float> d(32*3*224*224,0.f); std::vector<int64_t> s={32,3,224,224}; run(ctx->backbone,d,s); }
        { std::vector<float> d(8*3*224*224,0.f);  std::vector<int64_t> s={8,3,224,224};  run(ctx->deblur,d,s);   }
        { std::vector<float> d(32*6*224*224,0.f); std::vector<int64_t> s={32,6,224,224}; run(ctx->flow,d,s);     }
        LOGI("Warmup done.");
    }
    catch (...) { LOGI("Warmup failed — non-fatal, continuing."); }

    LOGI("All 5 models loaded and ready.");
    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_example_stablevqa_MainActivity_runStableVQABenchmark(
        JNIEnv *env, jobject,
        jlong ctxPtr,
        jfloatArray normArr,
        jfloatArray rawArr)
{
    pin_to_performance_cores();

    auto *ctx = reinterpret_cast<OrtContext *>(ctxPtr);
    if (!ctx) return nullptr;

    jsize n    = env->GetArrayLength(normArr);
    auto *nd   = env->GetFloatArrayElements(normArr, nullptr);
    std::vector<float> norm_frames(nd, nd + n);
    env->ReleaseFloatArrayElements(normArr, nd, JNI_ABORT);

    auto *rd = env->GetFloatArrayElements(rawArr, nullptr);
    std::vector<float> raw_frames(rd, rd + n);
    env->ReleaseFloatArrayElements(rawArr, rd, JNI_ABORT);

    const int NUM_FRAMES = 32;
    const int CHANNELS   = 3;
    const int FLOW_PAIRS = NUM_FRAMES;
    const int frame_size = CHANNELS * 224 * 224;
    const int pixels     = 224 * 224;
    (void)raw_frames;

    if ((int)norm_frames.size() < NUM_FRAMES * frame_size) return nullptr;

    const char *in[] = {"input"};
    const char *ob[] = {"output"};

    auto t0 = std::chrono::high_resolution_clock::now();
    auto ts = t0;
    float ms_backbone = 0, ms_deblur = 0, ms_flow = 0, ms_motion = 0, ms_quality = 0;

    auto elapsed_ms = [](auto &from) -> float
    {
        auto now = std::chrono::high_resolution_clock::now();
        float ms = (float)std::chrono::duration_cast<
                std::chrono::milliseconds>(now - from).count();
        from = now;
        return ms;
    };

    long rss_start = 0, rss_backbone = 0, rss_deblur = 0, rss_flow = 0, rss_motion = 0;

    try
    {
        rss_start = readNativeRssKb();
        std::vector<float> b_out, d_out, flow_out;

        auto run_backbone = [&]()
        {
            auto s = std::chrono::high_resolution_clock::now();
            std::vector<float> b_in = make_backbone_input_like_evaluator(
                    norm_frames, NUM_FRAMES, CHANNELS, pixels);
            std::vector<int64_t> sh = {NUM_FRAMES, CHANNELS, 224, 224};
            auto t = Ort::Value::CreateTensor<float>(
                    ctx->memory_info, b_in.data(), b_in.size(), sh.data(), sh.size());
            auto r = ctx->backbone->Run(Ort::RunOptions{nullptr}, in, &t, 1, ob, 1);
            float *p = r[0].GetTensorMutableData<float>();
            b_out.assign(p, p + r[0].GetTensorTypeAndShapeInfo().GetElementCount());
            ms_backbone = (float)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - s).count();
        };
        auto run_deblur = [&]()
        {
            auto s = std::chrono::high_resolution_clock::now();
            std::vector<float> db;
            db.reserve(8 * frame_size);
            for (int i = 0; i < NUM_FRAMES; i += 4)
                db.insert(db.end(),
                          norm_frames.begin() + i * frame_size,
                          norm_frames.begin() + (i + 1) * frame_size);
            while ((int)db.size() / frame_size < 8)
                db.insert(db.end(), db.end() - frame_size, db.end());
            std::vector<int64_t> sh = {8, CHANNELS, 224, 224};
            auto t = Ort::Value::CreateTensor<float>(
                    ctx->memory_info, db.data(), db.size(), sh.data(), sh.size());
            auto r = ctx->deblur->Run(Ort::RunOptions{nullptr}, in, &t, 1, ob, 1);
            auto osh = r[0].GetTensorTypeAndShapeInfo().GetShape();
            d_out = global_avg_pool(r[0].GetTensorMutableData<float>(), osh);
            ms_deblur = (float)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - s).count();
        };
        auto run_flow = [&]()
        {
            auto s = std::chrono::high_resolution_clock::now();
            std::vector<float> fb = make_flow_pairs_like_evaluator(
                    norm_frames, NUM_FRAMES, CHANNELS, pixels);
            std::vector<int64_t> sh = {FLOW_PAIRS, 6, 224, 224};
            auto t = Ort::Value::CreateTensor<float>(
                    ctx->memory_info, fb.data(), fb.size(), sh.data(), sh.size());
            auto r = ctx->flow->Run(Ort::RunOptions{nullptr}, in, &t, 1, ob, 1);
            float *fp    = r[0].GetTensorMutableData<float>();
            size_t count = r[0].GetTensorTypeAndShapeInfo().GetElementCount();
            flow_out.assign(fp, fp + count);
            ms_flow = (float)std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - s).count();
            if (flow_out.size() != (size_t)NUM_FRAMES * 2 * pixels)
                throw std::runtime_error(
                        "flow output is not [32,2,224,224]; re-export NeuFlow with FLOW_PAIRS=NUM_FRAMES");
        };

        if (USE_CONCURRENCY)
        {
            LOGI("Running backbone | deblur | flow concurrently");
            auto f_b = std::async(std::launch::async, [&]{ pin_to_performance_cores(); run_backbone(); });
            auto f_d = std::async(std::launch::async, [&]{ pin_to_performance_cores(); run_deblur();   });
            run_flow();      
            f_b.get();
            f_d.get();
        }
        else
        {
            LOGI("Running backbone -> deblur -> flow sequentially");
            run_backbone();
            run_deblur();
            run_flow();
        }

        { std::vector<float>().swap(norm_frames); }
        { std::vector<float>().swap(raw_frames);  }

        rss_backbone = rss_deblur = rss_flow = readNativeRssKb();
        LOGI("Backbone: %.0f ms  Deblur: %.0f ms  Flow: %.0f ms  RSS: %ld KB",
             ms_backbone, ms_deblur, ms_flow, rss_flow);

        ts = std::chrono::high_resolution_clock::now();   // reset for motion timing

        LOGI("4/5  Running motion analyzer");
        std::vector<float> m_out;
        {
            std::vector<float> mi(2 * NUM_FRAMES * pixels);
            for (int k = 0; k < NUM_FRAMES; ++k)
            {
                const float *src = flow_out.data() + k * 2 * pixels;
                std::memcpy(mi.data() + k * pixels,
                            src, pixels * sizeof(float));
                std::memcpy(mi.data() + NUM_FRAMES * pixels + k * pixels,
                            src + pixels, pixels * sizeof(float));
            }
            { std::vector<float>().swap(flow_out); }

            std::vector<int64_t> sh = {1, 2, NUM_FRAMES, 224, 224};
            auto t = Ort::Value::CreateTensor<float>(
                    ctx->memory_info, mi.data(), mi.size(), sh.data(), sh.size());
            auto r = ctx->motion->Run(Ort::RunOptions{nullptr}, in, &t, 1, ob, 1);
            float *p = r[0].GetTensorMutableData<float>();
            m_out.assign(p, p + r[0].GetTensorTypeAndShapeInfo().GetElementCount());
        }
        ms_motion  = elapsed_ms(ts);
        rss_motion = readNativeRssKb();
        LOGI("Motion: %.0f ms  RSS: %ld KB", ms_motion, rss_motion);

        float spatial_raw = mean_abs(b_out);
        float blur_raw    = mean_abs(d_out);
        float motion_raw  = mean_abs(m_out);
        float s_spatial, s_blur, s_motion;
        compute_branch_indicators(spatial_raw, blur_raw, motion_raw,
                                  s_spatial, s_blur, s_motion);
        LOGI("Indicators — Spatial=%.1f  Blur=%.1f  Motion=%.1f",
             s_spatial, s_blur, s_motion);

        LOGI("5/5  Running quality head");
        float raw_score = 0.f, mos = 0.f;
        {
            std::vector<float> fused;
            fused.reserve(d_out.size() + b_out.size() + m_out.size());
            fused.insert(fused.end(), d_out.begin(), d_out.end());
            fused.insert(fused.end(), b_out.begin(), b_out.end());
            fused.insert(fused.end(), m_out.begin(), m_out.end());
            { std::vector<float>().swap(d_out); }
            { std::vector<float>().swap(b_out); }
            { std::vector<float>().swap(m_out); }

            std::vector<int64_t> sh = {1, (int64_t)fused.size()};
            auto t = Ort::Value::CreateTensor<float>(
                    ctx->memory_info, fused.data(), fused.size(), sh.data(), sh.size());
            auto r = ctx->quality->Run(Ort::RunOptions{nullptr}, in, &t, 1, ob, 1);
            raw_score = r[0].GetTensorMutableData<float>()[0];
            mos       = map_to_mos(raw_score);
        }
        ms_quality = elapsed_ms(ts);
        LOGI("Quality head: %.0f ms", ms_quality);

        double total_dur = static_cast<double>(
                                   std::chrono::duration_cast<std::chrono::milliseconds>(
                                           std::chrono::high_resolution_clock::now() - t0).count()) / 1000.0;

        LOGI("MOS=%.2f [%s]  Spatial=%.1f  Blur=%.1f  Motion=%.1f  Total=%.2fs",
             mos, stability_label(mos).c_str(),
             s_spatial, s_blur, s_motion, total_dur);

        float out[15] = {
                mos,
                (float)total_dur,
                ms_backbone,
                ms_deblur,
                ms_flow,
                ms_motion,
                ms_quality,
                (float)rss_start,
                (float)rss_backbone,
                (float)rss_deblur,
                (float)rss_flow,
                (float)rss_motion,
                s_spatial,
                s_blur,
                s_motion
        };

        jfloatArray res = env->NewFloatArray(15);
        env->SetFloatArrayRegion(res, 0, 15, out);
        return res;
    }
    catch (const Ort::Exception &e)
    {
        LOGE("ORT exception: %s", e.what());
        return nullptr;
    }
    catch (const std::exception &e)
    {
        LOGE("Exception: %s", e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_stablevqa_MainActivity_destroyStableVQAModels(
        JNIEnv *, jobject, jlong ctxPtr)
{
    auto *ctx = reinterpret_cast<OrtContext *>(ctxPtr);
    if (ctx) { delete ctx; LOGI("ORT context freed."); }
}