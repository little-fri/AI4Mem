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
#include "data_collector.h"
#include <pthread.h>
#include <time.h>

// 全局变量
FILE *outputFile = NULL;
const char *DEFAULT_FILENAME = "uvm_monitor_result.csv";
int pagemap_fd = -1;
long page_size = 0;
static char current_filename[4096] = {0};
static pthread_mutex_t file_mutex = PTHREAD_MUTEX_INITIALIZER;
static volatile int rotate_thread_running = 0;
static pthread_t rotate_thread;
static int rotate_interval = 5; // seconds, can be overridden by env UVM_ROTATE_INTERVAL

#define CUPTI_CALL(call)                                                \
  do {                                                                  \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
      const char *errstr;                                               \
      cuptiGetResultString(_status, &errstr);                           \
      fprintf(stderr, "[CUPTI ERROR] %s:%d: %s\n", __FILE__, __LINE__, errstr); \
    }                                                                   \
  } while (0)

#define CUDA_CALL(call)                                                 \
  do {                                                                  \
    cudaError_t _status = call;                                         \
    if (_status != cudaSuccess) {                                       \
      fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_status)); \
    }                                                                   \
  } while (0)

#define BUF_SIZE (4 * 1024 * 1024)
#define ALIGN_SIZE (8)

// ================= 辅助函数 =================
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

// ================= CUPTI 回调 =================
void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  uint8_t *b = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  *size = BUF_SIZE;
  *buffer = b + ALIGN_SIZE;
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  CUpti_Activity *record = NULL;
  CUptiResult status;
  FILE *fp = NULL;
  char phys_info[64]; char map_info[256];

  do {
    status = cuptiActivityGetNextRecord(buffer, validSize, &record);
  if (status == CUPTI_SUCCESS) {
        // ================= Kernel =================
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)
        {
            CUpti_ActivityKernel4 *k = (CUpti_ActivityKernel4 *)record;

            int status_demangle = 0;
            char* real_name = abi::__cxa_demangle(k->name, NULL, NULL, &status_demangle);
            const char* kernel_name = (status_demangle == 0 && real_name) ? real_name : k->name;

      pthread_mutex_lock(&file_mutex);
      fp = (outputFile != NULL) ? outputFile : stderr;
      fprintf(fp,
                "KERNEL,%s,%u,0x%llx,%llu,%llu,%llu,%u,%u,%u,%u,%u,%u,%llu,%llu,%u,%d,%d\n",
                kernel_name,
                k->streamId,
                (unsigned long long)(*(uint32_t*)&k->cacheConfig),
                (unsigned long long)k->start,
                (unsigned long long)k->end,
                (unsigned long long)(k->end - k->start),
                k->gridX,
                k->gridY,
                k->gridZ,
                k->blockX,
                k->blockY,
                k->blockZ,
                (unsigned long long)k->staticSharedMemory,
                (unsigned long long)k->dynamicSharedMemory,
                k->registersPerThread,
                k->localMemoryTotal,
                0  // reserved for future
            );

      fflush(fp);
      pthread_mutex_unlock(&file_mutex);
      if (real_name) free(real_name);
        }
        // ================= UVM =================
        else if (record->kind == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER)
        {
            CUpti_ActivityUnifiedMemoryCounter2 *uvm = (CUpti_ActivityUnifiedMemoryCounter2 *)record;
            const char* typeStr = "UNKNOWN";
            if (uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT) typeStr = "GPU_PAGE_FAULT";
            else if (uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD) typeStr = "MIGRATE_HtoD";
            else if (uvm->counterKind == CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH) typeStr = "MIGRATE_DtoH";

      if (strcmp(typeStr, "UNKNOWN") != 0) {
        get_host_phys_info(uvm->address, phys_info, sizeof(phys_info));
        get_mapped_file(uvm->address, map_info, sizeof(map_info));

        pthread_mutex_lock(&file_mutex);
        fp = (outputFile != NULL) ? outputFile : stderr;
        fprintf(fp,
          "UVM,%s,0x%llx,%llu,,,%s,%s\n",
          typeStr,
          (unsigned long long)uvm->address,
          (unsigned long long)uvm->start,
          phys_info,
          map_info
        );
        fflush(fp);
        pthread_mutex_unlock(&file_mutex);

         // uvm_prefetcher_record_page((uint64_t)uvm->address);
      }
        }
    }
  } while (status == CUPTI_SUCCESS);

  size_t buffer_start = (size_t)buffer - ALIGN_SIZE;
  free((void*)buffer_start);
}

static void make_rotated_filename(const char* base, char* out, size_t out_len) {
  time_t t = time(NULL);
  struct tm tm;
  localtime_r(&t, &tm);
  char timestr[64];
  strftime(timestr, sizeof(timestr), "%Y%m%d_%H%M%S", &tm);

  // base may include path; create base_without_ext_timestr.csv
  const char* dot = strrchr(base, '.');
  if (dot && strlen(dot) <= 5) {
    size_t base_len = dot - base;
    if (base_len + 1 + strlen(timestr) + 5 < out_len) {
      snprintf(out, out_len, "%.*s_%s%s", (int)base_len, base, timestr, dot);
      return;
    }
  }
  // fallback
  snprintf(out, out_len, "%s_%s.csv", base, timestr);
}

static void* rotate_thread_fn(void* arg) {
  int interval = rotate_interval;
  while (rotate_thread_running) {
    sleep(interval);
    if (!rotate_thread_running) break;
    // flush CUPTI buffers
    cuptiActivityFlushAll(0);

    // rotate file: close old and open new
    pthread_mutex_lock(&file_mutex);
    if (outputFile) {
      fclose(outputFile);
      outputFile = NULL;
    }
    char newname[4096];
    make_rotated_filename(current_filename[0] ? current_filename : DEFAULT_FILENAME, newname, sizeof(newname));
    outputFile = fopen(newname, "w");
    if (outputFile) {
      // write header
      fprintf(outputFile, "Category,EventName,Address,StartTime_ns,EndTime_ns,Duration_ns,Host_Phys_Info,Mapped_File\n");
      fflush(outputFile);
      // update current_filename
      strncpy(current_filename, newname, sizeof(current_filename)-1);
      current_filename[sizeof(current_filename)-1] = 0;
    // update stable symlink/file for latest
    unlink("uvm_latest.csv");
    if (symlink(current_filename, "uvm_latest.csv") != 0) {
      FILE* f2 = fopen("uvm_latest.csv","w");
      if (f2) {
        fprintf(f2, "Category,EventName,Address,StartTime_ns,EndTime_ns,Duration_ns,Host_Phys_Info,Mapped_File\n");
        fclose(f2);
      }
    }
    }
    pthread_mutex_unlock(&file_mutex);
  }
  return NULL;
}

// ================= Profiler 初始化 / 析构 =================
__attribute__((constructor))
void init_profiler() {
  printf(">>> [Force Init] Attempting to wake up GPU... <<<\n");

  CUDA_CALL(cudaFree(0));

  const char* env_filename = getenv("UVM_LOG_FILE");
  const char* filename = (env_filename != NULL) ? env_filename : DEFAULT_FILENAME;
  // remember initial filename and open it
  strncpy(current_filename, filename, sizeof(current_filename)-1);
  current_filename[sizeof(current_filename)-1] = 0;
  outputFile = fopen(current_filename, "w");
  if (!outputFile) { perror("fopen"); exit(1); }

  pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
  page_size = sysconf(_SC_PAGESIZE);

  fprintf(outputFile, "Category,EventName,Address,StartTime_ns,EndTime_ns,Duration_ns,Host_Phys_Info,Mapped_File\n");
  fflush(outputFile);

  // create or update a stable symlink/file pointing to the current rotated file
  // so external scripts can always read 'uvm_latest.csv'
  unlink("uvm_latest.csv");
  if (symlink(current_filename, "uvm_latest.csv") != 0) {
    // fallback: copy header to uvm_latest.csv if symlink fails
    FILE* f2 = fopen("uvm_latest.csv","w");
    if (f2) {
      fprintf(f2, "Category,EventName,Address,StartTime_ns,EndTime_ns,Duration_ns,Host_Phys_Info,Mapped_File\n");
      fclose(f2);
    }
  }

  // check rotate interval env
  const char* env_rot = getenv("UVM_ROTATE_INTERVAL");
  if (env_rot) {
    int v = atoi(env_rot);
    if (v > 0) rotate_interval = v;
  }
  // start rotate thread
  rotate_thread_running = 1;
  if (pthread_create(&rotate_thread, NULL, rotate_thread_fn, NULL) != 0) {
    fprintf(stderr, "Warning: cannot start rotate thread\n");
    rotate_thread_running = 0;
  }

  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));

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
      printf("Warning: Config failed (Code %d). Ensure Driver supports Profiling.\n", configRes);
  } else {
      printf(">>> UVM Counters Configured Successfully. <<<\n");
  }

  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
}

__attribute__((destructor))
void stop_profiler() {
  printf(">>> Flushing data... <<<\n");
  // stop rotate thread first
  if (rotate_thread_running) {
      rotate_thread_running = 0;
      // wake up thread by flushing
      cuptiActivityFlushAll(0);
      pthread_join(rotate_thread, NULL);
  } else {
      cuptiActivityFlushAll(0);
  }
  pthread_mutex_lock(&file_mutex);
  if (outputFile) fclose(outputFile);
  pthread_mutex_unlock(&file_mutex);
  if (pagemap_fd != -1) close(pagemap_fd);
  printf(">>> Done. <<<\n");
}
