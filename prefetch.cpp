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
static void do_prefetch_once_internal() {
    std::vector<uint64_t> copy;
    {
        std::lock_guard<std::mutex> lg(s_mtx);
        if (s_addrs.empty()) return;
        copy.swap(s_addrs);
    }

    // normalize, sort and unique
    size_t page = get_page_size();
    for (auto &a : copy) {
        a = (a / page) * page;
    }
std::sort(copy.begin(), copy.end());
copy.erase(std::unique(copy.begin(), copy.end()), copy.end());

// merge into ranges
struct Range { uint64_t start; uint64_t end; };
std::vector<Range> ranges;
for (uint64_t a : copy) {
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

// perform prefetch
cudaError_t cerr;
cerr = cudaSetDevice(s_device_id);
if (cerr != cudaSuccess) {
    fprintf(stderr, "uvm_prefetcher: cudaSetDevice(%d) failed: %s\n", s_device_id, cudaGetErrorString(cerr));
}

for (auto &r : ranges) {
    void *ptr = (void*)(uintptr_t)r.start;
    size_t len = (size_t)(r.end - r.start);
    cerr = cudaMemPrefetchAsync(ptr, len, s_device_id, 0);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "uvm_prefetcher: cudaMemPrefetchAsync(%p, %zu) -> %s\n", ptr, len, cudaGetErrorString(cerr));
    } else {
        fprintf(stderr, "uvm_prefetcher: Prefetch %p - %p to device %d\n", ptr, (void*)(uintptr_t)r.end, s_device_id);
    }
}
// optionally synchronize to ensure migration completes before next step
cerr = cudaDeviceSynchronize();
if (cerr != cudaSuccess) {
    fprintf(stderr, "uvm_prefetcher: cudaDeviceSynchronize() -> %s\n", cudaGetErrorString(cerr));
}
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
s_thread = std::thread(prefetch_thread_main);
}
extern "C" void uvm_prefetcher_stop() {
if (!s_running.load()) return;
s_running.store(false);
s_cv.notify_all();
if (s_thread.joinable()) s_thread.join();
// clear any pending addresses
std::lock_guardstd::mutex lg(s_mtx);
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
    cerr = cudaSetDevice(device_id);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "uvm_prefetcher: cudaSetDevice(%d) failed: %s\n", device_id, cudaGetErrorString(cerr));
    }

    for (auto &r : ranges) {
        void *ptr = (void*)(uintptr_t)r.start;
        size_t len = (size_t)(r.end - r.start);
        cerr = cudaMemPrefetchAsync(ptr, len, device_id, 0);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "uvm_prefetcher: cudaMemPrefetchAsync(%p, %zu) -> %s\n", ptr, len, cudaGetErrorString(cerr));
        } else {
            fprintf(stderr, "uvm_prefetcher: Prefetch %p - %p to device %d\n", ptr, (void*)(uintptr_t)r.end, device_id);
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
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        // simple CSV split: Category,EventName,Address,...
        std::istringstream ss(line);
        std::string col;
        // first column
        if (!std::getline(ss, col, ',')) continue;
        if (col != "UVM") continue; // only UVM lines contain addresses
        // second column (event name)
        if (!std::getline(ss, col, ',')) continue;
        // third column is address
        std::string addrstr;
        if (!std::getline(ss, addrstr, ',')) continue;
        // trim spaces
        while (!addrstr.empty() && isspace((unsigned char)addrstr.front())) addrstr.erase(addrstr.begin());
        while (!addrstr.empty() && isspace((unsigned char)addrstr.back())) addrstr.pop_back();
        if (addrstr.size() == 0) continue;
        // accept hex like 0x... or decimal
        uint64_t addr = 0;
        try {
            size_t idx = 0;
            if (addrstr.find("0x") == 0 || addrstr.find("0X") == 0) {
                addr = std::stoull(addrstr, &idx, 16);
            } else {
                addr = std::stoull(addrstr, &idx, 10);
            }
        } catch (...) {
            continue;
        }
        // page-align
        size_t page = get_page_size();
        addr = (addr / page) * page;
        addrs.push_back(addr);
    }

    ifs.close();
    if (addrs.empty()) return;
    prefetch_addresses(addrs, device_id, sync != 0);
}
