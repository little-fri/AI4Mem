#ifndef UVM_PREFETCHER_H
#define UVM_PREFETCHER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void uvm_prefetcher_start(int interval_seconds, int device_id);
void uvm_prefetcher_stop();
void uvm_prefetcher_trigger_once();
void uvm_prefetcher_record_page(uint64_t addr);
void uvm_prefetcher_prefetch_from_file(const char* filename, int device_id, int sync);

#ifdef __cplusplus
}
#endif

#endif // UVM_PREFETCHER_H
