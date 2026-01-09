# AI4Memv3 — UVM/Prefetch research toolkit

这是为调试与研究基于 UVM 的内存迁移与预取(prefetch)策略而搭建的实验性工具链（收集器、预处理、模型训练、实时推理 + 注入式回放）。

该仓库聚焦于：
- 稳健采集 CUDA/UVM 事件（CUPTI/LD_PRELOAD 采集器）
- 将采集的事件解析为页级序列数据（preprocess）
- 使用时序模型（LSTM）训练一个预取/故障预测模型
- 在运行时将模型输出写为 CSV，并通过 LD_PRELOAD 注入器（preload_replayer.so）在目标进程启动/运行时触发预取

下面把主要组件、快速使用步骤、常见问题与进阶建议都整理在一起，便于复现和二次开发。

## 目录结构（高层）
- `data_collector.*` — CUPTI / 注入式运行时采集器（C/C++），会输出旋转文件和 `data.csv` / `total_data.csv`。
- `preload_replayer.c` — 一个小型 LD_PRELOAD watcher，当 `PREDICTIONS_FILE`（默认为 `./predictions.csv`）被修改时调用用户的 prefetcher 接口。
- `preprocess.py` — 将原始 UVM CSV 解析为时序样本（序列 -> 标签）供模型训练。
- `train_model.py` — LSTM 训练脚本（现在采用 PyTorch 实现，训练数据来源 `total_data.csv`）。
- `infer.py` / `run_prefetch_loop.py` — 推理与实时循环 supervisor，负责周期性调用预处理 -> 推理 -> 写出 `predictions.csv`。
- `uvm_* .csv` — collector 输出的 CSV（旋转副本 + `data.csv` + `total_data.csv` 累积）。

## 快速开始（最小复现流程）

1) 编译 `preload_replayer.so`

```bash
gcc -shared -fPIC -O2 -Wall -Wextra -o ./preload_replayer.so ./preload_replayer.c -ldl -lpthread
```

编译成功后会生成 `preload_replayer.so`，可由 `LD_PRELOAD` 加载。示例：

```bash
export PREDICTIONS_FILE=./predictions.csv
export PREFETCH_DEVICE=0
export PREFETCH_SYNC=0
export PREFETCH_POLL_MS=1000
export LD_PRELOAD="$(pwd)/preload_replayer.so${LD_PRELOAD:+:}$LD_PRELOAD"
# 启动目标程序（被注入）
./your_target_program
```

> 注意：`preload_replayer.so` 在进程构造函数中会立即进行一次预取（若 `PREDICTIONS_FILE` 存在），并启动一个检查线程以便在文件修改时再次触发预取。

2) 编译/安装采集器（可选）

如果你要在目标进程收集 UVM 事件，需要编译 `data_collector`：

- 如果使用 nvcc (CUDA)：

```bash
# 这是示例，具体命令取决于仓库中的 Makefile / CMakeLists
nvcc -shared -Xcompiler -fPIC -o data_collector.so data_collector.cpp -lcudart
```

- 或用 g++ 编译一个基于 LD_PRELOAD 的采集器（无 CUDA CUPTI 支持）：

```bash
g++ -shared -fPIC -O2 -o data_collector.so data_collector.cpp -ldl -pthread
```

采集器编译后放到 LD_PRELOAD（确保 `preload_replayer.so` 在前面），例如：

```bash
export LD_PRELOAD="$(pwd)/preload_replayer.so:$(pwd)/data_collector.so${LD_PRELOAD:+:}$LD_PRELOAD"
./test  # 运行目标 workload
```

3) 训练模型（离线）

处理好 `total_data.csv`（由 collector 累积生成）后：

```bash
# 小规模测试训练
python3 train_model.py --total-path total_data.csv --model-path model_lstm.pth --epochs 5

# 生成的文件： model_lstm.pth, model_lstm.pth.page2idx.pkl, model_lstm.pth.meta.json
```

4) 推理与实时循环（supervisor）

仓库里已有 `launch_with_replay.sh`，它通过环境变量设置 `PREDICTIONS_FILE` 并运行 `auto_collect_loop.py`（或 `run_prefetch_loop.py`）。简化使用：

```bash
./launch_with_replay.sh
```

或者手动启动 supervisor：

```bash
python3 run_prefetch_loop.py --interval 5 --model model_lstm.pth --out_dir run_out
```

该 supervisor 周期性（默认 5s）复制/读取 collector 输出，运行 `preprocess.py` -> `infer.py`，然后写出 `predictions.csv`（也会写 timestamp 备份）。`preload_replayer.so` 会监听该文件并触发预取。

## 重要环境变量
- `PREDICTIONS_FILE` 或 `PREFETCH_CSV`：预取器监听的 CSV 路径（preload_replayer 使用）。
- `PREFETCH_DEVICE`：目标 GPU 设备 id。
- `PREFETCH_SYNC`：是否同步等待预取完成（0=异步 默认，1=同步）。
- `PREFETCH_DELAY_SEC`：注入器启动后等待的秒数（可选）。
- `PREFETCH_POLL_MS`：watcher 轮询文件修改的毫秒间隔。

## 文件格式与约定
- Collector 输出 CSV 每行包含 (Category,EventName,Address,StartTime_ns,EndTime_ns,Duration_ns,Host_Phys_Info,Mapped_File)；脚本以 `UVM` 开头的行为目标事件。
- 地址应为 0x... 十六进制或能解析为整数；脚本按 PAGE_SIZE=4096 做页化映射。
- `data.csv`：每次 collector 旋转后会把最新旋转内容写为 `data.csv` 供实时消费者使用。
- `total_data.csv`：累积的历史事件，用于离线训练。

## 调试 & 常见问题
- 如果 `preload_replayer.so` 无法加载：检查 `LD_PRELOAD` 路径是否是绝对路径、库文件权限是否可读。
- 查看注入器日志：`/root/AI4Mem/preload_replayer.log`（如果脚本使用绝对路径写入），或者仓库下的 `preload_replayer.log`。
- 如果 watcher 没有响应文件改写：确认 `PREDICTIONS_FILE` 指向你要写入的文件，且写入操作是原子替换（推荐先写到临时文件然后 rename），这样 mtime 会变化。
- 如果模型预测效果不好：
	- 检查 `preprocess.py` 的序列构造（seq_len、horizon）是否与你期待的任务一致；
	- 增加样本数、使用更长的序列或加入时间差特征；
	- 使用多标签（predict future-k pages）而非当前二分类任务。

## 评估
仓库内有评估脚本（`eval.py`），可以计算 recall@K、precision@K、mAP 和 micro-F1。训练完成后运行评估：

```bash
python3 eval.py --data train_data_big.pth --model model_lstm.pth --out eval_report.json
```

## 开发建议 / 下一步
- 若目标是“预测未来 2s 内将被访问的页集合”，建议将训练标签改为 multi-hot 的 horizon 标签，模型输出扩展到 vocab_size（多标签 BCEWithLogits）。
- 考虑把推理过程放到一个小型守护进程并用轻量模型以减少在线延迟。

如果你需要，我可以：
- 帮你把 `infer.py` 和 `preload_replayer` 的接口对接（确保 CSV 格式兼容）；
- 把 `train_model.py` 改为 multi-label / horizon 预测并连同 infer/eval 一并调整；
- 或者演示一次从采集 -> 训练 -> 推理 -> 注入的端到端运行（并把我在终端运行的命令和输出贴出来）。

---

README 更新时间：2025-12-16
## AI4Mem4：基于 UVM 的页面访问采集、训练与预取一体化框架

AI4Mem4 目录集成了你们团队当前所有的核心功能：

- 使用 CUPTI 采集 CUDA Unified Memory (UVM) 页访问与迁移事件
- 持续将采集的 CSV 日志转换为训练数据序列
- 使用深度学习模型预测未来一段时间内的“热页”
- 在被测进程内部通过 LD_PRELOAD 注入预取器，依据预测结果触发 `cudaMemPrefetchAsync` 进行页面预取

整个流程围绕一个“循环采集 + 训练/推理 + 在线预取”的闭环设计，可以对任意基于 UVM 的 CUDA 程序进行实验性优化。

---

## 目录结构与主要组件

AI4Mem4 目录下的关键文件和脚本如下（只列核心）：

- `test` / `test.cu`
	- 示例 CUDA UVM 测试程序，用于验证采集、推理与预取链路是否正常工作。

- `data_collector.cpp` / `data_collector.h` / `data_collector.so`
	- 使用 CUPTI Activity 接口采集 Unified Memory 相关事件（如 GPU page fault、HtoD/DtoH 迁移）。
	- 以 CSV 形式输出到 `uvm_monitor_result*.csv`，一行一条事件，包含：类别（KERNEL/UVM）、事件名、地址、时间戳等。

- `preprocess.py`
	- 读取采集到的 UVM CSV（例如 `uvm_monitor_result.csv` 或轮转后的文件），
	- 将其切分为固定长度的时间序列，构造训练/推理所需的张量，并序列化为 `.pth`（例如 `train_data_loop.pth`）。

- `train.py` / `train_model.py`
	- 训练深度学习模型（例如 Transformer / RNN）以预测未来一段时间内会被访问的页面（热页）。
	- 输入：由 `preprocess.py` 生成的训练数据 `.pth`。
	- 输出：模型权重文件（如 `model_big.pth`），供推理阶段使用。

- `infer.py`
	- 加载训练好的模型，对最新一轮的 UVM 事件进行推理，
	- 输出预测结果为 CSV（通常命名为 `predictions.csv`），包含：触发时间戳、页地址、页索引、置信度等。

- `auto_collect_loop.py`
	- 整个“采集→预处理→推理”的自动化循环脚本。
	- 负责：
		- 以 `LD_PRELOAD=data_collector.so` 的方式启动被测程序（如 `./test`），持续采集 UVM 事件；
		- 每隔 `--interval` 秒，从最新的 `uvm_monitor_result*.csv` 中拷贝一份为 `data.csv`；
		- 调用 `preprocess.py` 生成 `train_data_loop.pth`；
		- 调用 `infer.py` 对当前数据推理，生成/覆盖 `predictions.csv`；
		- 可选地将各轮 `data.csv` 聚合到 `total_data.csv` 以便离线训练/分析。

- `prefetch.cpp` / `uvm_prefetcher`（由其编译出的 `libuvm_prefetcher.so`）
	- 实现通用 UVM 预取器：读取 CSV 中列出的页地址，进行页对齐、排序与区间合并，然后调用 `cudaMemPrefetchAsync` 迁移到指定 GPU。
	- 支持：
		- 从 UVM 采集 CSV（`UVM,...,Address,...`）中解析地址；
		- 从模型预测 CSV（`trigger_time_ns,page_addr,page_idx,score`）中解析预测的页地址；
		- 后台线程定期预取或一次性从文件预取；
		- chunked 预取（默认 1MB）以减少大量小调用。

- `preload_replayer.c` / `preload_replayer.so`
	- 通过 `LD_PRELOAD` 注入到目标进程中的“重放+在线预取”小库。
	- 在构造函数中：
		- 读取环境变量 `PREDICTIONS_FILE` / `PREFETCH_CSV`，确定预测 CSV 路径；
		- 通过 `dlopen("libuvm_prefetcher.so")` 和 `dlsym` 找到 `uvm_prefetcher_prefetch_from_file`；
		- 首次调用预取函数（可选延时）；
		- 启动一个后台 `pthread` 线程，每隔 `PREFETCH_POLL_MS` 毫秒检查 CSV mtime，若有更新则再次调用预取。

- `launch_with_replay.sh`
	- 示例启动脚本，用于把上述所有功能串起来：
		- 设置 `PREDICTIONS_FILE` / `PREFETCH_CSV` 指向模型输出的 `predictions.csv`；
		- 配置 `PREFETCH_DEVICE`、`PREFETCH_SYNC`、`PREFETCH_DELAY_SEC`、`PREFETCH_POLL_MS` 等参数；
		- 设置 `LD_PRELOAD` 让 `preload_replayer.so` 注入目标程序，然后执行被测应用或 `auto_collect_loop.py`。

- `cycle_collect_train.py` / `train_model copy.py`
	- 针对循环采集与训练流程的脚本（例如多轮采集后统一训练），细节可参考脚本内部注释。

---

## 环境与依赖

### 基本环境
- OS：Ubuntu 20.04+（示例环境）
- GPU：支持 CUDA 的 NVIDIA GPU（例如 A800 系列）
- CUDA：CUDA Toolkit 12.x（含 `nvcc` 与 `libcudart`）
- Python：Python 3.8+，已安装常用科学计算/深度学习库（如 PyTorch、numpy 等）

### 编译依赖
- C/C++ 编译器：`g++`（支持 C++11）
- CUDA Runtime：`-I/usr/local/cuda/include` 与 `-L/usr/local/cuda/lib64 -lcudart`
- CUPTI：data_collector 依赖 CUPTI 库（通常位于 `/usr/local/cuda/targets/x86_64-linux/lib/libcupti.so`）

---

## 快速开始

下面给出一个典型的端到端流程，用于：
1. 启动测试程序并采集 UVM 事件；
2. 循环预处理并训练/推理，生成预测 CSV；
3. 使用注入的预取器根据预测做在线预取实验。

> 以下命令假设当前目录为 `/root/AI4Memv4`。根据你的实际路径适当调整。

### 1. 编译组件

（如果已经在 Makefile 或脚本中完成，可以略过）

1）编译 `data_collector.so`：
```bash
cd /root/AI4Memv4
nvcc -shared -Xcompiler -fPIC data_collector.cpp -o data_collector.so \
		-I/usr/local/cuda/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcupti
```

2）编译 `libuvm_prefetcher.so`（预取器）：
```bash
cd /root/AI4Memv4
g++ -std=c++11 -shared -fPIC prefetch.cpp -o libuvm_prefetcher.so \
		-DPREFETCH_VERBOSE=0 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -pthread
```

3）编译示例测试程序（如未编译）：
```bash
cd /root/AI4Memv4
nvcc -O2 test.cu -o test
```

### 2. 启动采集+推理循环

在 `AI4Mem4` 根目录下运行：

```bash
cd /root/AI4Mem4
python3 auto_collect_loop.py \
	--interval 5 \
	--test-cmd "./test" \
	--model model_big.pth \
	--total-log --total-sep --prune-rotated
```

上面的命令会：
- 使用 `LD_PRELOAD=data_collector.so` 启动 `./test`，持续采集 UVM 日志到 `uvm_monitor_result*.csv`；
- 每 5 秒轮询一次，生成/更新 `data.csv`；
- 调用 `preprocess.py` 和 `infer.py`，将最新的采集数据转换为张量并用模型推理，生成/覆盖 `predictions.csv`；
- 可选地将所有轮次的 `data.csv` 追加到 `total_data.csv` 以便后续离线训练。

### 3. 在目标程序中注入预取器（在线预取）

当上述循环已经能稳定生成 `predictions.csv` 后，可以通过 `preload_replayer.so` 在目标程序内部执行预测驱动的 UVM 预取。

典型使用方式是通过 `launch_with_replay.sh`：

```bash
cd /root/AI4Memv4
./launch_with_replay.sh
```

该脚本大致做的事情是：
- 设置：
	- `PREDICTIONS_FILE="/root/AI4Mem/predictions.csv"`（模型输出）
	- `PREFETCH_DEVICE=0`（目标 GPU）
	- `PREFETCH_SYNC`（是否同步等待预取完成，在线模式通常用 0）
	- `PREFETCH_DELAY_SEC`（预取器启动前延时，给应用一些初始化/分配时间）
	- `PREFETCH_POLL_MS`（轮询预测文件更新的间隔，单位 ms）
- 把 `preload_replayer.so` 放入 `LD_PRELOAD` 以注入目标进程；
- 执行你的测试/生产命令，例如自动循环脚本或某个 CUDA 应用。

预取器在进程内部运行：
- 构造时调用一次 `uvm_prefetcher_prefetch_from_file(PREDICTIONS_FILE, device, sync)`；
- 后台线程检测预测 CSV 每次被重写（mtime 变化），重新解析其中的页地址并调用 `cudaMemPrefetchAsync` 进行迁移。

> 当前版本基于“绝对虚地址”预取，如果预测 CSV 中的地址来自 **其它运行** 的采集（而非同一进程同一运行），则由于地址空间随机化和分配差异，`cudaMemPrefetchAsync` 可能返回 `invalid argument`。稳健方案是引入 `alloc_id + offset` 机制，这在后续版本中可以扩展。

---

## 典型工作流

1. **离线采集与训练**
	 - 使用 `auto_collect_loop.py` 多次运行目标程序，积累 `uvm_monitor_result*.csv` 和 `total_data.csv`；
	 - 离线调用 `preprocess.py` 将汇总数据转换为训练集；
	 - 使用 `train.py` / `train_model.py` 在 GPU 上训练模型，得到新的 `model_big.pth`。

2. **在线推理与预取实验**
	 - 启动 `auto_collect_loop.py`，使其在运行程序的同时不断更新 `predictions.csv`；
	 - 使用 `launch_with_replay.sh` 或手动设置 `LD_PRELOAD=preload_replayer.so` 注入当前进程；
	 - 观察有无预取时的 page-fault 数、迁移带宽和 kernel 时间变化（可借助 data_collector 的输出或 nvidia-smi / Nsight 工具）。

---

## 注意事项与已知限制

- **地址一致性与 invalid argument**：
	- `cudaMemPrefetchAsync` 仅接受当前进程中已分配的 managed 地址。如果预测的地址来自另一轮运行或另一个进程，对当前进程来说就是无效地址，会导致 `invalid argument` 错误。
	- 为了获得稳健的跨运行预取，推荐改造为记录 `alloc_id + offset` 并在重放时根据当前运行的分配表重建地址。

- **性能与开销**：
	- 尽量避免在热路径中输出海量日志；当前代码中 `PREFETCH_VERBOSE` 默认关闭，必要时再开启调试输出。
	- chunk 大小（默认 1MB）和轮询间隔等参数需要根据实际应用调优。

- **安全与稳定性**：
	- 由于使用了 LD_PRELOAD 注入，可能与其它使用相同机制的工具（如某些 profiler）发生干扰，建议在受控实验环境中使用。

---

## 进一步扩展方向

- 引入 `cudaMallocManaged` 钩子记录 `alloc_id + base + size`，让模型预测基于 offset 而不是绝对地址，从而支持跨运行/进程的重放预取。
- 支持多 GPU、多进程环境下的推理与预取策略（例如根据 device id 和 NUMA 拓扑做决策）。
- 与 Nsight Systems / Nsight Compute 或 Prometheus 监控整合，用统一面板展示 page-fault、迁移和预取命中率等指标。

欢迎团队成员在本 README 基础上继续补充更详细的实验记录和配置说明。