#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <cuda_runtime.h>
#include "uvm_prefetcher.h"

// Extended test: compare kernel runtime with and without prefetch
// Usage: ./test_prefetch [size_bytes]
// Default size: 8 MB

__global__ void touch_kernel(char *data, size_t page_size, size_t npages) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = gridDim.x * blockDim.x;
    for (size_t p = tid; p < npages; p += total) {
        size_t offset = p * page_size;
        // touch one byte per page
        char v = data[offset];
        data[offset] = v + 1;
    }
}

static float run_kernel_and_time(char *ptr, size_t size) {
    size_t page = sysconf(_SC_PAGESIZE);
    size_t npages = (size + page - 1) / page;

    // choose number of threads: one thread per page up to a limit
    size_t threads = npages;
    const size_t maxThreads = 1 << 20; // 1M
    if (threads > maxThreads) threads = maxThreads;
    int block = 256;
    int grid = (threads + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    touch_kernel<<<grid, block>>>(ptr, page, npages);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(int argc, char **argv) {
    size_t SIZE = 8 * 1024 * 1024; // default 8 MB
    std::string mode = "both"; // both, noprefetch, prefetch
    if (argc >= 2) SIZE = strtoull(argv[1], NULL, 10);
    if (argc >= 3) mode = argv[2];

    // Explicitly select device 0 to ensure the context exists on the main thread
    cudaError_t devErr = cudaSetDevice(0);
    if (devErr != cudaSuccess) {
        fprintf(stderr, "test_prefetch: cudaSetDevice(0) -> %s\n", cudaGetErrorString(devErr));
    } else {
        int cur = -1; cudaGetDevice(&cur);
        fprintf(stderr, "test_prefetch: cudaSetDevice(0) done, current device=%d\n", cur);
    }

    void *ptr = nullptr;
    if (cudaMallocManaged(&ptr, SIZE) != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed\n");
        return 1;
    }

    size_t page = sysconf(_SC_PAGESIZE);
    size_t npages = (SIZE + page - 1) / page;

    printf("Test size: %zu bytes, pages: %zu (page=%zu), mode=%s\n", SIZE, npages, page, mode.c_str());

    // Print base pointer and size for verification
    fprintf(stderr, "test_prefetch: managed base ptr=%p, size=%zu\n", ptr, SIZE);

    if (mode == "noprefetch" || mode == "both") {
        // Ensure pages are resident on host
        memset(ptr, 0x7f, SIZE);
        // give OS a moment
        sleep(1);

        // No-prefetch run
        printf("Running without prefetch...\n");
        // make sure pages are on host before kernel
        memset(ptr, 0x7f, SIZE);
        float t_nopref = run_kernel_and_time((char*)ptr, SIZE);
        printf("No-prefetch kernel time: %.3f ms\n", t_nopref);

        if (mode == "noprefetch") {
            cudaFree(ptr);
            return 0;
        }

        // For 'both' case, bring pages back to host for fair comparison
        memset(ptr, 0x7f, SIZE);
        sleep(1);
    }

    if (mode == "prefetch" || mode == "both") {
        printf("Running with prefetch...\n");

        // First, try a main-thread prefetch (sanity check for device/context)
        cudaError_t cerr = cudaMemPrefetchAsync(ptr, SIZE, 0, 0);
        if (cerr != cudaSuccess) {
            fprintf(stderr, "test_prefetch: main-thread cudaMemPrefetchAsync(ptr,%zu,0) -> %s\n", SIZE, cudaGetErrorString(cerr));
        } else {
            fprintf(stderr, "test_prefetch: main-thread cudaMemPrefetchAsync succeeded for %zu bytes\n", SIZE);
        }
        cerr = cudaDeviceSynchronize();
        fprintf(stderr, "test_prefetch: main-thread cudaDeviceSynchronize() -> %s\n", cudaGetErrorString(cerr));

        // start prefetcher
        uvm_prefetcher_start(1, 0);
        // ensure pages are on host before recording
        memset(ptr, 0x7f, SIZE);
        // Print first few page addresses for verification
        const size_t SHOW = 4;
        for (size_t i = 0; i < npages; ++i) {
            uint64_t addr = (uint64_t)((uintptr_t)ptr + i * page);
            if (i < SHOW) fprintf(stderr, "test_prefetch: record page[%zu]=%p\n", i, (void*)(uintptr_t)addr);
            uvm_prefetcher_record_page(addr);
        }
        uvm_prefetcher_trigger_once();
        // wait for prefetch to (likely) complete
        sleep(2);

        float t_pref = run_kernel_and_time((char*)ptr, SIZE);
        printf("With-prefetch kernel time: %.3f ms\n", t_pref);

        uvm_prefetcher_stop();

        if (mode == "prefetch") {
            cudaFree(ptr);
            return 0;
        }

        // If both, we need t_nopref which was printed earlier; for 'both' we printed both times
        // (The 'both' path already printed No-prefetch time earlier.)
    }

    cudaFree(ptr);
    return 0;
}
