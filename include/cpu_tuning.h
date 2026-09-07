#pragma once

#include <string>
#include <vector>
#include <set>
#include <thread>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define SVQA_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
static inline void svqa_cpuidex(int regs[4], int leaf, int sub) { __cpuidex(regs, leaf, sub); }
#else
#include <cpuid.h>
static inline void svqa_cpuidex(int regs[4], int leaf, int sub)
{
    unsigned a, b, c, dd;
    __cpuid_count((unsigned)leaf, (unsigned)sub, a, b, c, dd);
    regs[0] = (int)a; regs[1] = (int)b; regs[2] = (int)c; regs[3] = (int)dd;
}
#endif
#else
#define SVQA_X86 0
#endif

struct CpuInfo
{
    std::string vendor   = "Unknown";   // "Intel" | "AMD" | "Unknown"
    std::string brand    = "";
    int         logical  = 0;           // logical processors (with SMT/HT)
    int         physical = 0;           // physical cores
    bool        hybrid   = false;       // Intel P/E hybrid
    int         pPhysical = 0;          // physical P-cores (hybrid)
    uint64_t    pCoreMask = 0;          // affinity mask of P-cores (hybrid)
};

struct Tuning
{
    int         threads  = 1;
    uint64_t    affinity = 0;           // 0 = do not restrict
    std::string profile  = "Generic";
};

static inline int svqa_popcnt64(uint64_t x)
{
    int c = 0;
    while (x) { x &= (x - 1); c++; }
    return c;
}

static inline Tuning decideTuning(const CpuInfo &ci, const char *threadsEnv)
{
    Tuning t;
    if (ci.vendor == "Intel" && ci.hybrid && ci.pPhysical > 0)
    {
        t.threads  = ci.pPhysical;      // P physical cores
        t.affinity = ci.pCoreMask;      // pin to P-cores; keep work off slow E-cores
        t.profile  = "Intel-hybrid (P-cores only)";
    }
    else if (ci.vendor == "Intel")
    {
        t.threads  = ci.physical;       // homogeneous Intel to physical cores
        t.affinity = 0;
        t.profile  = "Intel";
    }
    else if (ci.vendor == "AMD")
    {
        t.threads  = ci.physical;       // Ryzen homogeneous to physical cores (avoid SMT contention)
        t.affinity = 0;
        t.profile  = "AMD-Ryzen";
    }
    else
    {
        t.threads  = ci.physical > 0 ? ci.physical : std::max(1, ci.logical);
        t.affinity = 0;
        t.profile  = "Generic";
    }

    if (threadsEnv)
    {
        int v = atoi(threadsEnv);
        if (v > 0) { t.threads = v; t.profile += " [SVQA_THREADS override]"; }
    }

    const char *noAffEnv = std::getenv("SVQA_NO_AFFINITY");
    if (noAffEnv && atoi(noAffEnv) != 0)
    {
        t.affinity = 0;
        t.profile += " [no-affinity]";
    }

    if (t.threads < 1) t.threads = 1;
    return t;
}

static inline void detectTopology(CpuInfo &ci)
{
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()), &len))
        return;

    struct Core { int eff; uint64_t mask; };
    std::vector<Core> cores;
    int maxEff = -1;

    char *p = buf.data();
    char *end = buf.data() + len;
    while (p < end)
    {
        auto *info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(p);
        if (info->Relationship == RelationProcessorCore)
        {
            uint64_t mask = 0;
            for (int g = 0; g < info->Processor.GroupCount; ++g)
                mask |= (uint64_t)info->Processor.GroupMask[g].Mask;   // assume group 0 (desktop)
            int eff = (int)info->Processor.EfficiencyClass;            // higher = performance
            cores.push_back({eff, mask});
            maxEff = std::max(maxEff, eff);
        }
        p += info->Size;
    }

    ci.physical = (int)cores.size();
    std::set<int> effClasses;
    for (auto &c : cores)
    {
        effClasses.insert(c.eff);
        if (c.eff == maxEff) { ci.pCoreMask |= c.mask; ci.pPhysical++; }
    }
    ci.hybrid = (effClasses.size() > 1);
    if (!ci.hybrid) { ci.pPhysical = ci.physical; ci.pCoreMask = 0; }

#elif defined(__linux__)
    std::set<std::pair<int,int>> uniqueCores;
    int cpu = 0;
    while (true)
    {
        char pkgPath[256], corePath[256];
        std::snprintf(pkgPath, sizeof(pkgPath),
            "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        std::snprintf(corePath, sizeof(corePath),
            "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        FILE *fp = std::fopen(pkgPath, "r");
        if (!fp) break;
        int pkg = 0, core = 0;
        if (std::fscanf(fp, "%d", &pkg) != 1) { std::fclose(fp); break; }
        std::fclose(fp);
        FILE *fc = std::fopen(corePath, "r");
        if (fc) { if (std::fscanf(fc, "%d", &core) != 1) core = cpu; std::fclose(fc); }
        uniqueCores.insert({pkg, core});
        cpu++;
        if (cpu > 4096) break;
    }
    if (!uniqueCores.empty()) ci.physical = (int)uniqueCores.size();
    else ci.physical = ci.logical > 1 ? ci.logical / 2 : std::max(1, ci.logical);
    ci.pPhysical = ci.physical;

#else
    // Other POSIX (e.g. macOS): assume SMT (physical ≈ logical/2).
    ci.physical  = ci.logical > 1 ? ci.logical / 2 : (ci.logical > 0 ? ci.logical : 1);
    ci.pPhysical = ci.physical;
#endif
}

static inline CpuInfo detectCpu()
{
    CpuInfo ci;
    ci.logical = (int)std::thread::hardware_concurrency();

#if SVQA_X86
    int r[4];
    svqa_cpuidex(r, 0, 0);
    char v[13] = {0};
    std::memcpy(v + 0, &r[1], 4);   // EBX
    std::memcpy(v + 4, &r[3], 4);   // EDX
    std::memcpy(v + 8, &r[2], 4);   // ECX
    std::string vs(v);
    if      (vs == "GenuineIntel") ci.vendor = "Intel";
    else if (vs == "AuthenticAMD") ci.vendor = "AMD";
    else if (!vs.empty())          ci.vendor = vs;

    svqa_cpuidex(r, 0x80000000, 0);
    unsigned maxExt = (unsigned)r[0];
    if (maxExt >= 0x80000004u)
    {
        char brand[49] = {0};
        for (int i = 0; i < 3; ++i)
        {
            svqa_cpuidex(r, 0x80000002 + i, 0);
            std::memcpy(brand + i * 16, r, 16);
        }
        ci.brand = brand;
        size_t s = ci.brand.find_first_not_of(' ');
        if (s != std::string::npos) ci.brand = ci.brand.substr(s);
    }
#endif

    detectTopology(ci);
    if (ci.physical  <= 0) ci.physical  = ci.logical > 1 ? std::max(1, ci.logical / 2) : std::max(1, ci.logical);
    if (ci.pPhysical <= 0) ci.pPhysical = ci.physical;
    return ci;
}
