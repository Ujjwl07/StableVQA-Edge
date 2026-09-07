// Unit test for the per-vendor tuning decision. Runs anywhere (no ORT/OpenCV/OS
// deps): it feeds SYNTHETIC CpuInfo profiles through decideTuning() and checks
// the chosen thread count / affinity / profile. Also prints what THIS machine's
// detectCpu() reports.
//
// Build:  clang++ -std=c++17 test_cpu_tuning.cpp -o test_cpu_tuning
// Run:    ./test_cpu_tuning
#include "cpu_tuning.h"
#include <cstdio>
#include <string>

static int g_pass = 0, g_fail = 0;

static void check(const std::string &name, const CpuInfo &ci, const char *env,
                  int wantThreads, bool wantAffinity, const std::string &wantProfileContains)
{
    Tuning t = decideTuning(ci, env);
    bool okThreads = (t.threads == wantThreads);
    bool okAff     = ((t.affinity != 0) == wantAffinity);
    bool okProf    = (t.profile.find(wantProfileContains) != std::string::npos);
    bool ok = okThreads && okAff && okProf;
    (ok ? g_pass : g_fail)++;
    std::printf("%-38s -> threads=%-2d affinity=%-3s profile=\"%s\"  [%s]\n",
                name.c_str(), t.threads, (t.affinity ? "yes" : "no"),
                t.profile.c_str(), ok ? "PASS" : "FAIL");
    if (!ok)
        std::printf("    expected threads=%d affinity=%s profile~\"%s\"\n",
                    wantThreads, wantAffinity ? "yes" : "no", wantProfileContains.c_str());
}

int main()
{
    std::printf("=== decideTuning() on synthetic CPU profiles ===\n\n");

    // Ryzen 7 5800X: 8 physical, 16 logical (SMT), homogeneous.
    { CpuInfo ci; ci.vendor="AMD"; ci.physical=8; ci.logical=16; ci.hybrid=false; ci.pPhysical=8;
      check("Ryzen 7 5800X (8c/16t)", ci, nullptr, 8, false, "AMD-Ryzen"); }

    // Ryzen 9 7950X: 16 physical, 32 logical.
    { CpuInfo ci; ci.vendor="AMD"; ci.physical=16; ci.logical=32; ci.hybrid=false; ci.pPhysical=16;
      check("Ryzen 9 7950X (16c/32t)", ci, nullptr, 16, false, "AMD-Ryzen"); }

    // Intel i7-9700 (pre-hybrid): 8 physical, 8 logical, homogeneous.
    { CpuInfo ci; ci.vendor="Intel"; ci.physical=8; ci.logical=8; ci.hybrid=false; ci.pPhysical=8;
      check("Intel i7-9700 (8c/8t, non-hybrid)", ci, nullptr, 8, false, "Intel"); }

    // Intel i7-13700 hybrid: 8 P-cores (16 threads) + 8 E-cores = 16 physical, 24 logical.
    { CpuInfo ci; ci.vendor="Intel"; ci.physical=16; ci.logical=24; ci.hybrid=true;
      ci.pPhysical=8; ci.pCoreMask=0x0000FFFFull;  // 16 P-core logical procs
      check("Intel i7-13700 (hybrid, 8P+8E)", ci, nullptr, 8, true, "Intel-hybrid"); }

    // Intel i9-12900K hybrid: 8P+8E = 16 physical, 24 logical.
    { CpuInfo ci; ci.vendor="Intel"; ci.physical=16; ci.logical=24; ci.hybrid=true;
      ci.pPhysical=8; ci.pCoreMask=0x0000FFFFull;
      check("Intel i9-12900K (hybrid, 8P+8E)", ci, nullptr, 8, true, "Intel-hybrid"); }

    // Unknown vendor fallback.
    { CpuInfo ci; ci.vendor="Unknown"; ci.physical=6; ci.logical=12; ci.hybrid=false; ci.pPhysical=6;
      check("Unknown vendor (6c/12t)", ci, nullptr, 6, false, "Generic"); }

    // SVQA_THREADS override wins over everything.
    { CpuInfo ci; ci.vendor="AMD"; ci.physical=8; ci.logical=16; ci.pPhysical=8;
      check("Ryzen + SVQA_THREADS=12 override", ci, "12", 12, false, "override"); }

    std::printf("\n=== detectCpu() on THIS machine ===\n");
    CpuInfo here = detectCpu();
    Tuning  t    = decideTuning(here, std::getenv("SVQA_THREADS"));
    std::printf("vendor=%s  brand=\"%s\"  physical=%d  logical=%d  hybrid=%s  P-cores=%d\n",
                here.vendor.c_str(), here.brand.c_str(), here.physical, here.logical,
                here.hybrid ? "yes" : "no", here.pPhysical);
    std::printf("would choose: threads=%d  profile=%s\n",
                t.threads, t.profile.c_str());

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
