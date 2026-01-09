#include "uvm_prefetcher.h"
#include <chrono>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <unistd.h>
#include <stdint.h>
#include <atomic>
#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>
#include <iomanip>
#include <unordered_map>
// Debug logging control: set PREFETCH_VERBOSE to 1 to enable verbose stderr logs
#ifndef PREFETCH_VERBOSE
#define PREFETCH_VERBOSE 0
#endif

#if PREFETCH_VERBOSE
#define LOGF(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#else
#define LOGF(fmt, ...) do { } while (0)
#endif
static std::vector<uint64_t> s_addrs;
static std::mutex s_mtx;
static std::condition_variable s_cv;
static std::thread s_thread;
static std::atomic<bool> s_running(false);
static int s_interval_seconds = 5;
static int s_device_id = 0;
static size_t get_page_size() {
    static size_t ps = 0;
    if (ps == 0) ps = (size_t)sysconf(_SC_PAGESIZE);
    return ps;
}

// Simple file logger to record which pages/ranges are being prefetched.
static void prefetcher_log(const std::string &msg) {
    const char *logpath = "/root/AI4Memv3/prefetcher.log";
    std::ofstream lf(logpath, std::ios::app);
    if (!lf.is_open()) return;
    std::time_t t = std::time(nullptr);
    lf << std::put_time(std::localtime(&t), "%F %T") << ": " << msg << std::endl;
}

// When the shared object is loaded, write a tiny file so we can detect which SO was loaded.
__attribute__((constructor)) static void prefetcher_on_load() {
    const char *loaded_path = "/root/AI4Memv3/prefetcher_loaded.log";
    FILE *f = fopen(loaded_path, "a");
    if (f) {
        char tb[64];
        time_t t = time(NULL);
        strftime(tb, sizeof(tb), "%F %T", localtime(&t));
        fprintf(f, "%s: libuvm_prefetcher.so loaded (pid=%d)\n", tb, getpid());
        fclose(f);
    }
    // Also try a quick write to the main prefetcher.log so we don't depend on C++ streams here.
    const char *logpath = "/root/AI4Memv3/prefetcher.log";
    FILE *lf = fopen(logpath, "a");
    if (lf) {
        char tb2[64];
        time_t t2 = time(NULL);
        strftime(tb2, sizeof(tb2), "%F %T", localtime(&t2));
        fprintf(lf, "%s: lib loaded constructor write (pid=%d)\n", tb2, getpid());
        fclose(lf);
    }
}

// forward declaration for chunked prefetcher helper
struct AddrMeta;
static void prefetch_addresses(const std::vector<uint64_t> &addrs, int device_id, bool sync, const std::vector<AddrMeta> *metas = nullptr);
static void do_prefetch_once_internal() {
    std::vector<uint64_t> copy;
    {
        std::lock_guard<std::mutex> lg(s_mtx);
        if (s_addrs.empty()) return;
        copy.swap(s_addrs);
    }

    // normalize, sort and unique (page-aligned addresses)
    size_t page = get_page_size();
    for (auto &a : copy) {
        a = (a / page) * page;
    }
    std::sort(copy.begin(), copy.end());
    copy.erase(std::unique(copy.begin(), copy.end()), copy.end());

    // Delegate actual prefetching to the chunked implementation
    prefetch_addresses(copy, s_device_id, true);
}
static void prefetch_thread_main() {
    std::unique_lock<std::mutex> lk(s_mtx);
    while (s_running.load()) {
        // wait for interval or notify
        s_cv.wait_for(lk, std::chrono::seconds(s_interval_seconds));
        if (!s_running.load()) break;
        // unlock while doing heavy work
        lk.unlock();
        do_prefetch_once_internal();
        lk.lock();
    }
}
extern "C" void uvm_prefetcher_start(int interval_seconds, int device_id) {
    if (s_running.load()) return; // already running
    s_interval_seconds = (interval_seconds > 0) ? interval_seconds : 5;
    s_device_id = device_id;
    s_running.store(true);
    int devCount = 0;
    cudaError_t _cerr = cudaGetDeviceCount(&devCount);
    if (_cerr != cudaSuccess) {
        fprintf(stderr, "uvm_prefetcher: cudaGetDeviceCount() -> %s\n", cudaGetErrorString(_cerr));
    } else {
        LOGF("uvm_prefetcher: starting (interval=%d, device=%d), deviceCount=%d\n", s_interval_seconds, s_device_id, devCount);
    }
    s_thread = std::thread(prefetch_thread_main);
}
extern "C" void uvm_prefetcher_stop() {
    if (!s_running.load()) return;
    s_running.store(false);
    s_cv.notify_all();
    if (s_thread.joinable()) s_thread.join();
    // clear any pending addresses
    std::lock_guard<std::mutex> lg(s_mtx);
    s_addrs.clear();
}
extern "C" void uvm_prefetcher_trigger_once() {
// Wake thread and let it perform a prefetch immediately
s_cv.notify_all();
}
extern "C" void uvm_prefetcher_record_page(uint64_t addr) {
    std::lock_guard<std::mutex> lg(s_mtx);
    s_addrs.push_back(addr);
}

// Helper: perform prefetch for a vector of page-aligned addresses
struct AddrMeta {
    uint64_t page_addr;
    uint64_t orig_addr;
    std::string timestamp_ns;
    std::string confidence;
};

static void prefetch_addresses(const std::vector<uint64_t> &addrs, int device_id, bool sync, const std::vector<AddrMeta> *metas) {
    if (addrs.empty()) return;

    // merge into ranges (input assumed page-aligned)
    struct Range { uint64_t start; uint64_t end; };
    std::vector<Range> ranges;
    size_t page = get_page_size();

    std::vector<uint64_t> tmp = addrs;
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());

    // Build a lookup from page -> list of metadata entries (if provided)
    std::unordered_map<uint64_t, std::vector<AddrMeta>> meta_map;
    if (metas) {
        for (const auto &m : *metas) {
            meta_map[m.page_addr].push_back(m);
        }
    }

    for (uint64_t a : tmp) {
        if (ranges.empty()) ranges.push_back({a, a + page});
        else {
            Range &last = ranges.back();
            if (a <= last.end) {
                if (a + page > last.end) last.end = a + page;
            } else {
                ranges.push_back({a, a + page});
            }
        }
    }

    // Log the computed ranges for debugging
    {
        std::ostringstream oss;
        oss << "prefetch_addresses: computed " << ranges.size() << " ranges. ";
        for (size_t i = 0; i < ranges.size() && i < 20; ++i) {
            oss << "[" << std::hex << "0x" << ranges[i].start << "-0x" << ranges[i].end << "]";
            if (i + 1 < ranges.size()) oss << ",";
        }
        prefetcher_log(oss.str());
    }

    cudaError_t cerr;
    int devCount = 0;
    cerr = cudaGetDeviceCount(&devCount);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "uvm_prefetcher: cudaGetDeviceCount() -> %s\n", cudaGetErrorString(cerr));
    } else {
        LOGF("uvm_prefetcher: prefetch_addresses: deviceCount=%d, requestedDevice=%d, ranges=%zu\n", devCount, device_id, ranges.size());
    }

    cerr = cudaSetDevice(device_id);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "uvm_prefetcher: cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(cerr));
    } else {
        int cur = -1; cudaGetDevice(&cur);
        LOGF("uvm_prefetcher: cudaSetDevice done, current device=%d\n", cur);
        // Force context creation on this thread before prefetching
        cudaError_t sync_err = cudaDeviceSynchronize();
        LOGF("uvm_prefetcher: cudaDeviceSynchronize (after set device) -> %s\n", cudaGetErrorString(sync_err));
    }

    // Chunked prefetch: avoid issuing a single huge prefetch call
    const size_t CHUNK_BYTES = 1 << 20; // 1MB
    for (auto &r : ranges) {
        uint64_t cur = r.start;
        uint64_t end = r.end;
        while (cur < end) {
            size_t len = (size_t)std::min<uint64_t>((uint64_t)CHUNK_BYTES, end - cur);
            void *ptr = (void*)(uintptr_t)cur;
            LOGF("uvm_prefetcher: attempting chunked cudaMemPrefetchAsync(%p, %zu) -> device %d\n", ptr, len, device_id);
            cerr = cudaMemPrefetchAsync(ptr, len, device_id, 0);
            if (cerr != cudaSuccess) {
                std::ostringstream _oss;
                _oss << "cudaMemPrefetchAsync(" << std::hex << ptr << "," << std::dec << len << ") -> " << cudaGetErrorString(cerr);
                prefetcher_log(_oss.str());
                fprintf(stderr, "uvm_prefetcher: cudaMemPrefetchAsync(%p, %zu) -> %s\n", ptr, len, cudaGetErrorString(cerr));
                // Log per-page failure if we have metadata
                if (!meta_map.empty()) {
                    uint64_t p = cur;
                    for (; p < cur + len; p += page) {
                        auto it = meta_map.find(p);
                        if (it != meta_map.end()) {
                            for (const auto &mm : it->second) {
                                std::ostringstream eoss;
                                eoss << "Prefetch FAILED page 0x" << std::hex << p << " (orig=0x" << mm.orig_addr << ") device=" << device_id << " err=" << cudaGetErrorString(cerr) << " ts=" << mm.timestamp_ns << " conf=" << mm.confidence;
                                prefetcher_log(eoss.str());
                            }
                        }
                    }
                }
                // continue trying other chunks
            } else {
                std::ostringstream _oss2;
                _oss2 << "Prefetch chunk " << std::hex << ptr << " - 0x" << (cur + len) << " to device " << device_id;
                prefetcher_log(_oss2.str());
                LOGF("uvm_prefetcher: Prefetch chunk %p - %p to device %d\n", ptr, (void*)(uintptr_t)(cur + len), device_id);
                // Log per-page successes if we have metadata
                if (!meta_map.empty()) {
                    uint64_t p = cur;
                    for (; p < cur + len; p += page) {
                        auto it = meta_map.find(p);
                        if (it != meta_map.end()) {
                            for (const auto &mm : it->second) {
                                std::ostringstream soss;
                                soss << "Prefetch SUCCESS page 0x" << std::hex << p << " (orig=0x" << mm.orig_addr << ") device=" << device_id << " ts=" << mm.timestamp_ns << " conf=" << mm.confidence;
                                prefetcher_log(soss.str());
                            }
                        }
                    }
                }
            }
            cur += len;
        }
    }

    if (sync) {
        cerr = cudaDeviceSynchronize();
        if (cerr != cudaSuccess) {
            fprintf(stderr, "uvm_prefetcher: cudaDeviceSynchronize() -> %s\n", cudaGetErrorString(cerr));
        }
    }
}

// Parse CSV output produced by data_collector and prefetch listed pages.
extern "C" void uvm_prefetcher_prefetch_from_file(const char* filename, int device_id, int sync) {
    if (!filename) return;
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        fprintf(stderr, "uvm_prefetcher: cannot open file %s\n", filename);
        return;
    }
    std::vector<uint64_t> addrs;
    std::vector<AddrMeta> metas;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        // Try expected predictions.csv format: Timestamp_ns,Prefetch_Address,Confidence
        std::istringstream ss(line);
        std::string ts_col, addr_col, conf_col;
        if (!std::getline(ss, ts_col, ',')) continue;
        if (!std::getline(ss, addr_col, ',')) continue;
        if (!std::getline(ss, conf_col, ',')) {
            // fallback: older UVM format: Category,EventName,Address,...
            ss.clear(); ss.str(line);
            std::string col0, col1, col2;
            if (!std::getline(ss, col0, ',')) continue;
            if (col0 != "UVM") continue;
            if (!std::getline(ss, col1, ',')) continue;
            if (!std::getline(ss, col2, ',')) continue;
            ts_col = "";
            addr_col = col2;
            conf_col = "";
        }
        auto trim = [](std::string &s) {
            while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
            while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
        };
        trim(ts_col); trim(addr_col); trim(conf_col);
        if (addr_col.empty()) continue;
        uint64_t orig_addr = 0;
        try {
            size_t idx = 0;
            if (addr_col.find("0x") == 0 || addr_col.find("0X") == 0) orig_addr = std::stoull(addr_col, &idx, 16);
            else orig_addr = std::stoull(addr_col, &idx, 10);
        } catch (...) { continue; }
        size_t page = get_page_size();
        uint64_t page_addr = (orig_addr / page) * page;
        addrs.push_back(page_addr);
        AddrMeta m; m.page_addr = page_addr; m.orig_addr = orig_addr; m.timestamp_ns = ts_col; m.confidence = conf_col;
        metas.push_back(m);
    }

    ifs.close();
    if (addrs.empty()) return;

    // Log the addresses we parsed (count and sample) for debugging
    {
        std::ostringstream oss;
        oss << "uvm_prefetcher_prefetch_from_file: parsed " << addrs.size() << " addresses. sample=";
        size_t show = std::min<size_t>(20, addrs.size());
        for (size_t i = 0; i < show; ++i) {
            oss << std::hex << "0x" << addrs[i];
            if (i + 1 < show) oss << ",";
        }
        prefetcher_log(oss.str());
    }
    prefetch_addresses(addrs, device_id, sync != 0, &metas);
}
