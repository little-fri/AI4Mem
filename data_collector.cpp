#include <cupti.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <cxxabi.h>
#include "uvm_prefetcher.h"
// 全局变量
FILE *outputFile = NULL;
const char *DEFAULT_FILENAME = "uvm_monitor_result.csv";
int pagemap_fd = -1;
long page_size = 0;
#define CUPTI_CALL(call)                                                
do {                                                                  
CUptiResult _status = call;                                         
if (_status != CUPTI_SUCCESS) {                                     
const char *errstr;                                               
cuptiGetResultString(_status, &errstr);                           
fprintf(stderr, "[CUPTI ERROR] %s:%d: %s\n", FILE, LINE, errstr); 
}                                                                   
} while (0)
#define CUDA_CALL(call)                                                 
do {                                                                  
cudaError_t _status = call;                                         
if (_status != cudaSuccess) {                                       
fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", FILE, LINE, cudaGetErrorString(_status)); 
}                                                                   
} while (0)
#define BUF_SIZE (4 * 1024 * 1024)
#define ALIGN_SIZE (8)
// === 辅助函数 (保持不变) ===
void get_host_phys_info(uint64_t vaddr, char* buffer, size_t buf_len) {
if (pagemap_fd == -1) { snprintf(buffer, buf_len, "ERR_NO_FD"); return; }
uint64_t page_index = vaddr / page_size;
uint64_t offset = page_index * 8;
uint64_t paddr_info = 0;
if (pread(pagemap_fd, &paddr_info, 8, offset) != 8) { snprintf(buffer, buf_len, "ERR_READ"); return; }
int present = (paddr_info >> 63) & 1;
int swapped = (paddr_info >> 62) & 1;
uint64_t pfn = paddr_info & ((1ULL << 55) - 1);
if (present) snprintf(buffer, buf_len, "RAM_Phys:0x%llx", (unsigned long long)(pfn * page_size));
else if (swapped) snprintf(buffer, buf_len, "SWAP_DISK");
else snprintf(buffer, buf_len, "NOT_RESIDENT");
}
void get_mapped_file(uint64_t vaddr, char* buffer, size_t buf_len) {
FILE* maps = fopen("/proc/self/maps", "r");
if (!maps) { snprintf(buffer, buf_len, "ERR_OPEN_MAPS"); return; }
char line[512]; int found = 0;
while (fgets(line, sizeof(line), maps)) {
uint64_t start, end;
if (sscanf(line, "%lx-%lx", &start, &end) == 2) {
if (vaddr >= start && vaddr < end) {
char* p = strrchr(line, ' ');
if (p && strlen(p) > 1) {
char* nl = strchr(p, '\n'); if (nl) *nl = 0;
snprintf(buffer, buf_len, "%s", p + 1);
} else snprintf(buffer, buf_len, "ANONYMOUS");
found = 1; break;
}
}
}
if (!found) snprintf(buffer, buf_len, "UNKNOWN_REGION");
fclose(maps);
}
// === CUPTI 回调 ===
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
uint8_t *b = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
*size = BUF_SIZE;
*buffer = b + ALIGN_SIZE;
*maxNumRecords = 0;
}
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
CUpti_Activity *record = NULL;
CUptiResult status;
FILE *fp = (outputFile != NULL) ? outputFile : stderr;
char phys_info[64]; char map_info[256];
do {
status = cuptiActivityGetNextRecord(buffer, validSize, &record);
if (status == CUPTI_SUCCESS) {
if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
CUpti_ActivityKernel4 *k = (CUpti_ActivityKernel4 )record;
int status_demangle;
char real_name = abi::__cxa_demangle(k->name, 0, 0, &status_demangle);
fprintf(fp, "KERNEL,%s,,%llu,%llu,%llu,,\n", (status_demangle == 0) ? real_name : k->name,
(unsigned long long)k->start, (unsigned long long)k->end, (unsigned long long)(k->end - k->start));
if(status_demangle == 0) free(real_name);
}
else if (record->kind == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) {
CUpti_ActivityUnifiedMemoryCounter2 *uvm = (CUpti_ActivityUnifiedMemoryCounter2 )record;
const char typeStr = "UNKNOWN";
if (uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT) typeStr = "GPU_PAGE_FAULT";
else if (uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD) typeStr = "MIGRATE_HtoD";
else if (uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH) typeStr = "MIGRATE_DtoH";
code
Code
if (strcmp(typeStr, "UNKNOWN") != 0) {
        get_host_phys_info(uvm->address, phys_info, sizeof(phys_info));
        get_mapped_file(uvm->address, map_info, sizeof(map_info));
  fprintf(fp, "UVM,%s,0x%llx,%llu,,,%s,%s\n", typeStr, (unsigned long long)uvm->address, (unsigned long long)uvm->start, phys_info, map_info);
  // record page address for optional prefetcher
  uvm_prefetcher_record_page((uint64_t)uvm->address);
    }
  }
}
} while (status == CUPTI_SUCCESS);
size_t buffer_start = (size_t)buffer - ALIGN_SIZE;
free((void*)buffer_start);
}
// === 核心修改：在构造函数中强制初始化 ===
attribute((constructor))
void init_profiler() {
printf(">>> [Force Init] Attempting to wake up GPU... <<<\n");
// 1. 强制 CUDA 初始化
// cudaFree(0) 是官方推荐的“无副作用”建立 Context 的方法
CUDA_CALL(cudaFree(0));
// 2. 准备文件
const char* env_filename = getenv("UVM_LOG_FILE");
const char* filename = (env_filename != NULL) ? env_filename : DEFAULT_FILENAME;
outputFile = fopen(filename, "w");
if (!outputFile) { perror("fopen"); exit(1); }
pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
page_size = sysconf(_SC_PAGESIZE);
// 写入表头 - 只要执行到这里，文件就不应该为空
fprintf(outputFile, "Category,EventName,Address,StartTime_ns,EndTime_ns,Duration_ns,Host_Phys_Info,Mapped_File\n");
fflush(outputFile); // 立即刷新，确保即使崩溃也有表头
// 3. 注册回调
CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
// 4. 显式配置 Ampere 计数器 (Page Fault)
CUpti_ActivityUnifiedMemoryCounterConfig config[3];
config[0].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT;
config[0].enable = 1;
config[0].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
config[0].deviceId = 0;
config[1].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
config[1].enable = 1;
config[1].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
config[1].deviceId = 0;
config[2].kind = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;
config[2].enable = 1;
config[2].scope = CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_SCOPE_PROCESS_SINGLE_DEVICE;
config[2].deviceId = 0;
CUptiResult configRes = cuptiActivityConfigureUnifiedMemoryCounter(config, 3);
if (configRes != CUPTI_SUCCESS) {
printf("Warning: Config failed (Code %d). If this is K80/Pascal, it's fine. For Ampere, ensure Driver supports Profiling.\n", configRes);
} else {
printf(">>> UVM Counters Configured Successfully. <<<\n");
}
// 5. 开启监控
CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
}
attribute((destructor))
void stop_profiler() {
printf(">>> Flushing data... <<<\n");
cuptiActivityFlushAll(0);
if (outputFile) fclose(outputFile);
if (pagemap_fd != -1) close(pagemap_fd);
printf(">>> Done. <<<\n");
}

