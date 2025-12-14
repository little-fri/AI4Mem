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
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cuda_runtime.h>
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

// Check if [addr, addr+len) is covered by any mapping in /proc/self/maps.
static bool is_addr_in_maps(uint64_t addr, size_t len) {
    std::ifstream maps("/proc/self/maps");
    if (!maps.is_open()) return false;
    std::string line;
    uint64_t a = addr;
    uint64_t b = addr + len;
    while (std::getline(maps, line)) {
        // format: start-end perms ...
        size_t dash = line.find('-');
        if (dash == std::string::npos) continue;
        std::string s1 = line.substr(0, dash);
        size_t sp = line.find(' ', dash + 1);
        if (sp == std::string::npos) continue;
        std::string s2 = line.substr(dash + 1, sp - dash - 1);
        uint64_t start = 0, end = 0;
        try {
            start = std::stoull(s1, nullptr, 16);
            end = std::stoull(s2, nullptr, 16);
        } catch (...) {
            continue;
        }
        if (a >= start && b <= end) return true;
    }
    return false;
}

// forward declaration for chunked prefetcher helper
static void prefetch_addresses(const std::vector<uint64_t> &addrs, int device_id, bool sync);
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
static void prefetch_addresses(const std::vector<uint64_t> &addrs, int device_id, bool sync) {
    if (addrs.empty()) return;

    // merge into ranges (input assumed page-aligned)
    struct Range { uint64_t start; uint64_t end; };
    std::vector<Range> ranges;
    size_t page = get_page_size();

    std::vector<uint64_t> tmp = addrs;
    std::sort(tmp.begin(), tmp.end());
    tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());

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
    const int MAX_RETRY = 3;
    const int RETRY_SLEEP_US = 10000; // 10ms

    for (auto &r : ranges) {
        uint64_t cur = r.start;
        uint64_t end = r.end;
        while (cur < end) {
            size_t len = (size_t)std::min<uint64_t>((uint64_t)CHUNK_BYTES, end - cur);
            void *ptr = (void*)(uintptr_t)cur;
            LOGF("uvm_prefetcher: attempting chunked cudaMemPrefetchAsync(%p, %zu) -> device %d\n", ptr, len, device_id);
            // log attempt to file
            {
                FILE *lf = fopen("/root/AI4Memv2/prefetch_file.log", "a");
                if (lf) {
                    char tb[64];
                    time_t t = time(NULL);
                    strftime(tb, sizeof(tb), "%F %T", localtime(&t));
                    fprintf(lf, "%s: attempting cudaMemPrefetchAsync(%p, %zu) device=%d\n", tb, ptr, len, device_id);
                    fclose(lf);
                }
            }
            int attempt = 0;
            bool done = false;
            while (attempt < MAX_RETRY && !done) {
                if (!is_addr_in_maps((uint64_t)(uintptr_t)ptr, len)) {
                    FILE *lf = fopen("/root/AI4Memv2/prefetch_file.log", "a");
                    if (lf) {
                        char tb[64];
                        time_t t = time(NULL);
                        strftime(tb, sizeof(tb), "%F %T", localtime(&t));
                        fprintf(lf, "%s: addr %p len=%zu not in /proc/self/maps (attempt %d/%d)\n", tb, ptr, len, attempt+1, MAX_RETRY);
                        fclose(lf);
                    }
                    ++attempt;
                    if (attempt < MAX_RETRY) usleep(RETRY_SLEEP_US);
                    continue;
                }

                cerr = cudaMemPrefetchAsync(ptr, len, device_id, 0);
                if (cerr != cudaSuccess) {
                    fprintf(stderr, "uvm_prefetcher: cudaMemPrefetchAsync(%p, %zu) -> %s\n", ptr, len, cudaGetErrorString(cerr));
                    FILE *lf = fopen("/root/AI4Memv2/prefetch_file.log", "a");
                    if (lf) {
                        char tb[64];
                        time_t t = time(NULL);
                        strftime(tb, sizeof(tb), "%F %T", localtime(&t));
                        fprintf(lf, "%s: cudaMemPrefetchAsync(%p, %zu) -> ERROR: %s (attempt %d/%d)\n", tb, ptr, len, cudaGetErrorString(cerr), attempt+1, MAX_RETRY);
                        fclose(lf);
                    }
                    ++attempt;
                    if (attempt < MAX_RETRY) usleep(RETRY_SLEEP_US);
                } else {
                    LOGF("uvm_prefetcher: Prefetch chunk %p - %p to device %d\n", ptr, (void*)(uintptr_t)(cur + len), device_id);
                    FILE *lf = fopen("/root/AI4Memv2/prefetch_file.log", "a");
                    if (lf) {
                        char tb[64];
                        time_t t = time(NULL);
                        strftime(tb, sizeof(tb), "%F %T", localtime(&t));
                        fprintf(lf, "%s: cudaMemPrefetchAsync(%p, %zu) -> OK (attempt %d/%d)\n", tb, ptr, len, attempt+1, MAX_RETRY);
                        fclose(lf);
                    }
                    done = true;
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
        // also log to file
        FILE *lf = fopen("/root/AI4Memv2/prefetch_file.log", "a");
        if (lf) {
            char tb[64];
            time_t t = time(NULL);
            strftime(tb, sizeof(tb), "%F %T", localtime(&t));
            fprintf(lf, "%s: cannot open predictions file %s\n", tb, filename);
            fclose(lf);
        }
        return;
    }

    std::vector<uint64_t> addrs;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        // simple CSV split: Category,EventName,Address,... OR model: trigger_time_ns,page_addr,page_idx,score
        std::istringstream ss(line);
        std::string col;
        // first column
        if (!std::getline(ss, col, ',')) continue;
        // If the model CSV has header like "trigger_time_ns", skip header
        if (col.size() > 0 && (col.find("trigger_time") != std::string::npos || col.find("trigger_time_ns") != std::string::npos)) continue;
        // Determine whether line is UVM trace (starts with UVM) or model output (starts with timestamp)
        if (col == "UVM") {
            // UVM trace format: UVM,EventName,Address,...
            // second column (event name)
            if (!std::getline(ss, col, ',')) continue;
            // third column is address
            std::string addrstr;
            if (!std::getline(ss, addrstr, ',')) continue;
            // trim
            while (!addrstr.empty() && isspace((unsigned char)addrstr.front())) addrstr.erase(addrstr.begin());
            while (!addrstr.empty() && isspace((unsigned char)addrstr.back())) addrstr.pop_back();
            if (addrstr.size() == 0) continue;
            uint64_t addr = 0;
            try {
                size_t idx = 0;
                if (addrstr.find("0x") == 0 || addrstr.find("0X") == 0) {
                    addr = std::stoull(addrstr, &idx, 16);
                } else {
                    addr = std::stoull(addrstr, &idx, 10);
                }
            } catch (...) { continue; }
            size_t page = get_page_size();
            addr = (addr / page) * page;
            addrs.push_back(addr);
        } else {
            // Model output expected: trigger_time_ns,page_addr,page_idx,score
            // col is trigger_time_ns
            std::string addrstr;
            if (!std::getline(ss, addrstr, ',')) continue;
            // trim
            while (!addrstr.empty() && isspace((unsigned char)addrstr.front())) addrstr.erase(addrstr.begin());
            while (!addrstr.empty() && isspace((unsigned char)addrstr.back())) addrstr.pop_back();
            if (addrstr.size() == 0) continue;
            uint64_t addr = 0;
            try {
                size_t idx = 0;
                if (addrstr.find("0x") == 0 || addrstr.find("0X") == 0) {
                    addr = std::stoull(addrstr, &idx, 16);
                } else {
                    addr = std::stoull(addrstr, &idx, 10);
                }
            } catch (...) { continue; }
            size_t page = get_page_size();
            addr = (addr / page) * page;
            addrs.push_back(addr);
        }
    }

    ifs.close();

    // log how many addresses parsed
    FILE *lf = fopen("/root/AI4Memv2/prefetch_file.log", "a");
    if (lf) {
        char tb[64];
        time_t t = time(NULL);
        strftime(tb, sizeof(tb), "%F %T", localtime(&t));
        fprintf(lf, "%s: parsed %zu addresses from %s\n", tb, addrs.size(), filename);
        fclose(lf);
    }

    if (addrs.empty()) return;
    prefetch_addresses(addrs, device_id, sync != 0);
}
