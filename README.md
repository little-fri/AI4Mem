AI4Mem — prefetcher usage & build
=================================

这个目录包含一个简单的 UVM 预取器（`prefetch.cpp`）和一个 CUPTI-based 数据采集器（`data_collector.cpp`）。当前你要求不要修改 `data_collector.cpp`，因此下面只说明如何构建并测试 `prefetch` 部分。

构建（假设 CUDA 安装在 `/usr/local/cuda`，并且系统可见 cupti 库）：

```bash
cd /home/whj/AI4Mem
g++ -std=c++11 -fPIC -shared \
    data_collector.cpp prefetch.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart -lcuda \
    -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti \
    -pthread -o libuvm_monitor.so
```

要运行示例测试程序（直接链接使用 prefetch API）：

```bash
# 编译 test 程序（静态链接到 cuda runtime）
g++ -std=c++11 test_prefetch.cpp prefetch.cpp -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart -pthread -o test_prefetch

# 运行（确保有 CUDA 驱动和 GPU）
./test_prefetch
```

或者也可以把 `libuvm_monitor.so` 通过 LD_PRELOAD 注入到目标程序：

```bash
export UVM_LOG_FILE=/tmp/uvm_monitor_result.csv
LD_PRELOAD=./libuvm_monitor.so ./your_cuda_program
```

API 摘要（在 `uvm_prefetcher.h` 中声明）：
- `uvm_prefetcher_start(int interval_seconds, int device_id)` — 启动后台线程，每隔 interval_seconds 聚合并预取记录的页面。
- `uvm_prefetcher_stop()` — 停止后台线程。
- `uvm_prefetcher_record_page(uint64_t addr)` — 将虚地址（页内任意位置）记录到队列，用于后续预取。
- `uvm_prefetcher_trigger_once()` — 立即唤醒线程进行一次预取尝试。
- `uvm_prefetcher_prefetch_from_file(const char* filename, int device_id, int sync)` — 解析 `data_collector` 的 CSV 输出并对其中列出的地址进行预取（sync=1 则调用 cudaDeviceSynchronize）。

注意事项：
- `cudaMemPrefetchAsync` 仅对 Unified Memory（`cudaMallocManaged`）有效。
- `/proc/self/pagemap` 读取可能受内核安全配置限制；若读取失败需要以 root 或调整策略。
- CUPTI 需要与驱动版本兼容。

Model utility
-------------
另外我添加了一个非常轻量的“模型”工具 `prefetch_model.cpp`，它会读取 `data_collector` 产生的 CSV，按简单启发式（GPU page faults 权重高、HtoD 中等、DtoH 低）为每个页打分，选择分数超过阈值的页并通过 `uvm_prefetcher` 接口触发预取。

构建并运行示例：

```bash
# 编译模型（需要与 prefetch.cpp 链接，以便使用 uvm_prefetcher 接口）
g++ -std=c++11 prefetch_model.cpp prefetch.cpp -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart -pthread -o prefetch_model

# 运行：
./prefetch_model /path/to/uvm_monitor_result.csv [threshold] [device] [sync]
# 例如：
./prefetch_model /tmp/uvm_monitor_result.csv 5 0 1
```

该工具仅用于演示和快速原型。你们可以把启发式替换成更复杂的模型（LR、light-weight NN 等），或把其封装为服务供模型调用。
# AI4Mem — prefetcher usage & build

这个目录包含一个简单的 UVM 预取器（`prefetch.cpp`）和一个 CUPTI-based 数据采集器（`data_collector.cpp`）。当前你要求不要修改 `data_collector.cpp`，因此下面只说明如何构建并测试 `prefetch` 部分。

构建（假设 CUDA 安装在 `/usr/local/cuda`，并且系统可见 cupti 库）：

```bash
cd /home/whj/AI4Mem
g++ -std=c++11 -fPIC -shared \
    data_collector.cpp prefetch.cpp \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart -lcuda \
    -L/usr/local/cuda/extras/CUPTI/lib64 -lcupti \
    -pthread -o libuvm_monitor.so
```

要运行示例测试程序（直接链接使用 prefetch API）：

```bash
# 编译 test 程序（静态链接到 cuda runtime）
g++ -std=c++11 test_prefetch.cpp prefetch.cpp -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 -lcudart -pthread -o test_prefetch

# 运行（确保有 CUDA 驱动和 GPU）
./test_prefetch
```

或者也可以把 `libuvm_monitor.so` 通过 LD_PRELOAD 注入到目标程序：

```bash
export UVM_LOG_FILE=/tmp/uvm_monitor_result.csv
LD_PRELOAD=./libuvm_monitor.so ./your_cuda_program
```

API 摘要（在 `uvm_prefetcher.h` 中声明）：
- `uvm_prefetcher_start(int interval_seconds, int device_id)` — 启动后台线程，每隔 interval_seconds 聚合并预取记录的页面。
- `uvm_prefetcher_stop()` — 停止后台线程。
- `uvm_prefetcher_record_page(uint64_t addr)` — 将虚地址（页内任意位置）记录到队列，用于后续预取。
- `uvm_prefetcher_trigger_once()` — 立即唤醒线程进行一次预取尝试。
- `uvm_prefetcher_prefetch_from_file(const char* filename, int device_id, int sync)` — 解析 `data_collector` 的 CSV 输出并对其中列出的地址进行预取（sync=1 则调用 cudaDeviceSynchronize）。

注意事项：
- `cudaMemPrefetchAsync` 仅对 Unified Memory（`cudaMallocManaged`）有效。
- `/proc/self/pagemap` 读取可能受内核安全配置限制；若读取失败需要以 root 或调整策略。
- CUPTI 需要与驱动版本兼容。
