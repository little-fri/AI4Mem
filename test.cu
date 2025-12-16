#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h>
#include <sys/time.h>
#include <vector>

// 定义总数据大小 (例如 4GB float)
// A800 显存很大，我们可以分配大一点以避免 Cache 命中
#define N (1024 * 1024 * 1024L) // 1 Billion floats = 4GB

// Kernel: 简单的向量加法，甚至更简单，只是为了触发缺页
__global__ void uvm_touch_kernel(float* data, size_t offset, size_t size, float val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 读取并写入，触发缺页
        data[offset + idx] += val;
    }
}

// 获取当前时间戳 (秒)
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    float *uvm_data;
    size_t size = N * sizeof(float);

    // 1. 分配 Unified Memory
    std::cout << "Allocating " << size / (1024*1024) << " MB Unified Memory..." << std::endl;
    cudaError_t err = cudaMallocManaged(&uvm_data, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMallocManaged failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 2. 初始化 (在 CPU 上做，确保页面在 Host 端)
    std::cout << "Initializing data on CPU..." << std::endl;
    for (size_t i = 0; i < N; ++i) {
        uvm_data[i] = 0.0f;
    }

    // 3. 循环运行 20 秒
    double start_time = get_time();
    double duration = 20.0; // 目标运行 20 秒
    
    // 将内存分成很多小块，慢慢吃
    // 假设我们希望每 100ms (0.1s) 触发一次
    int steps = 200; 
    size_t chunk_size = N / steps; 
    
    std::cout << "Starting 20s workload on GPU..." << std::endl;
    std::cout << "Events will be spread over time..." << std::endl;

    int iter = 0;
    while (true) {
        double current_time = get_time();
        if (current_time - start_time >= duration) break;

        // 计算当前要访问的内存偏移量 (循环访问，防止越界)
        // 这种顺序访问模式对于你的 LSTM 预测步长非常有帮助
        size_t offset = (iter * chunk_size) % N;
        
        // --- 关键点 A: CPU 先摸一下 ---
        // 这确保如果这块内存之前在 GPU，现在会被拉回 CPU，
        // 或者确保它处于 System Memory 中。
        // 我们只摸 chunk 的第一个元素及少量元素以节省 CPU 时间
        uvm_data[offset] += 1.0f; 

        // --- 关键点 B: GPU 访问 ---
        // 启动 Kernel 访问这块内存 -> 触发 HtoD 迁移
        int threads = 256;
        int blocks = (chunk_size + threads - 1) / threads;
        
        uvm_touch_kernel<<<blocks, threads>>>(uvm_data, offset, chunk_size, 1.0f);
        
        // 同步，确保 Fault 发生
        cudaDeviceSynchronize();

        // --- 关键点 C: 控制时间 ---
        // 每次迭代大概睡 100ms (100,000 微秒)
        // A800很快，Kernel瞬间执行完，所以大部分时间在 sleep，
        // 但 UVM 事件会记录在 Kernel 启动的那一瞬间。
        usleep(100000); 
/*
        // 打印进度条
        if (iter % 10 == 0) {
            printf("Time: %.1fs / %.1fs | Accessed Offset: 0x%lx\n", 
                   current_time - start_time, duration, offset * sizeof(float));
        }
        iter++;*/
    }

    std::cout << "Workload finished." << std::endl;
    cudaFree(uvm_data);
    return 0;
}