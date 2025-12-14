#include <stdio.h>
#include <cuda_runtime.h>
#include<unistd.h>
// 数组大小：约 20M 个 float，约 80MB
// 我们把 N 调小到 ~20M，使 test 程序总运行时间大约在 20s 左右（每批 sleep 1s）
#define N (20 * 1024 * 1024)
#define BLOCK_SIZE 1024

__global__ void vector_add_uvm(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // GPU 尝试读取 a[idx] 和 b[idx]
        // 如果这些数据在 CPU 内存中，这里会触发 GPU_PAGE_FAULT
        c[idx] = a[idx] + b[idx];
    }
    for (volatile int i = 0; i < 100000; i++); // 空循环延时
}

int main() {
    float *a, *b, *c;
    size_t bytes = N * sizeof(float);

    printf("Allocating %zu bytes of Unified Memory...\n", bytes * 3);

    // 1. 分配 UVM 内存
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // 2. [关键步骤] 在 CPU 上初始化数据
    // 这会将物理页“钉”在 CPU 内存（Host Memory）中
    printf("Initializing data on CPU (forcing pages to Host RAM)...\n");
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // 3. 启动 GPU Kernel
    // GPU 开始执行，发现数据不在显存，必须触发缺页中断从 CPU 拉取数据
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Launching Kernel (Expect Page Faults now!)...\n");
    
    int batch = 1024*1024; // 每批 1M 元素
for (int start = 0; start < N; start += batch) {
    int len = min(batch, N - start);
    int gridSize = (len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add_uvm<<<gridSize, BLOCK_SIZE>>>(a+start, b+start, c+start, len);
    cudaDeviceSynchronize();
   // sleep(1); // 每批迁移之间睡 1 秒
}

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    printf("Kernel finished.\n");

    // 验证一下结果（可选，防止编译器优化掉 kernel）
    // 再次在 CPU 访问 c，可能再次触发缺页（从 GPU 迁回 CPU）
    if (c[0] == 3.0f) {
        printf("Verification Success!\n");
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}