#ifndef UVM_PREFETCHER_H
#define UVM_PREFETCHER_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// === 错误检查宏 ===
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// === 1. 将数据预取到 GPU ===
// 在 Kernel 启动前调用，可以消除 GPU Page Fault
static inline void uvm_prefetch_to_device(void *ptr, size_t size, cudaStream_t stream = 0) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device)); // 获取当前 GPU ID
    
    // 异步预取到显存
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, size, device, stream));
}

// === 2. 将数据预取回 CPU (Host) ===
// 在 Kernel 结束、CPU 读取结果前调用，消除 CPU Page Fault
static inline void uvm_prefetch_to_host(void *ptr, size_t size, cudaStream_t stream = 0) {
    // cudaCpuDeviceId 是 CUDA 定义的一个特殊 ID，代表主机内存
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, stream));
}

// === 3. 设置首选位置 (Preferred Location) ===
// 告诉驱动：即便数据被换出，也尽量让它留在 GPU 上
// 适合：GPU 频繁访问，CPU 偶尔访问的数据
static inline void uvm_set_preferred_gpu(void *ptr, size_t size) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device));
}

// === 4. 设置只读数据 (Read Mostly) ===
// 告诉驱动：这块数据很少修改，应该在 GPU 和 CPU 各自复制一份副本
// 适合：查找表、权重参数、只读输入数据
static inline void uvm_set_read_mostly(void *ptr, size_t size) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device));
}

// === 5. 简单的初始化/重置辅助函数 ===
// 确保数据强制在 CPU 上初始化（为了演示缺页效果）
static inline void uvm_init_on_cpu(void *ptr, size_t size, int value) {
    memset(ptr, value, size); 
    // memset 是主机函数，会强制把物理页拉回 CPU
}

#endif // UVM_PREFETCHER_H