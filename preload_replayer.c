#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

#include <stdarg.h>

typedef void (*prefetch_file_fn)(const char*, int, int);
static void call_prefetcher(const char* csv_path, int device, int sync) {
    if (!csv_path) return;
    // Try to load the prefetcher shared lib. Try absolute name first, then generic.
    const char* libnames[] = { "/root/AI4Memv2/libuvm_prefetcher.so", "libuvm_prefetcher.so", NULL };
    void* h = NULL;
    for (int i = 0; libnames[i]; ++i) {
        h = dlopen(libnames[i], RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen(libnames[i], RTLD_NOW);
        if (h) break;
    }
    if (!h) {
        fprintf(stderr, "preload_replayer: failed to dlopen libuvm_prefetcher.so: %s\n", dlerror());
        return;
    }
    prefetch_file_fn prefetch = (prefetch_file_fn)dlsym(h, "uvm_prefetcher_prefetch_from_file");
    if (!prefetch) {
        fprintf(stderr, "preload_replayer: symbol uvm_prefetcher_prefetch_from_file not found: %s\n", dlerror());
        dlclose(h);
        return;
    }
    // Call the prefetcher: csv path, device id, sync flag
    prefetch(csv_path, device, sync);
    // optionally keep library open; closing is fine here
    dlclose(h);
}

typedef struct {
    const char* csv_path;
    int device;
    int sync;
    int poll_ms;
} watcher_config_t;

static void* watch_and_prefetch(void* arg) {
    watcher_config_t* cfg = (watcher_config_t*)arg;
    struct stat st;
    time_t last_mtime = 0;
    while (1) {
        if (stat(cfg->csv_path, &st) == 0) {
            if (st.st_mtime != last_mtime) {
                last_mtime = st.st_mtime;
                // log detection
                {
                    FILE *lf = fopen("/root/AI4Memv2/preload_replayer.log", "a");
                    if (lf) {
                        char tb[64];
                        time_t t = time(NULL);
                        strftime(tb, sizeof(tb), "%F %T", localtime(&t));
                        fprintf(lf, "%s: watcher detected mtime change for %s (mtime=%ld), calling prefetcher\n", tb, cfg->csv_path, (long)st.st_mtime);
                        fclose(lf);
                    }
                }
                call_prefetcher(cfg->csv_path, cfg->device, cfg->sync);
            }
        }
        usleep(cfg->poll_ms * 1000);
    }
    return NULL;
}

// Run as soon as this shared library is loaded into the target process.
__attribute__((constructor))
static void preload_replayer_init(void) {
    const char* csv = getenv("PREDICTIONS_FILE");
    if (!csv) csv = getenv("PREFETCH_CSV");
    if (!csv) return; // nothing to watch

    const char* devstr = getenv("PREFETCH_DEVICE");
    int device = devstr ? atoi(devstr) : 0;
    const char* syncstr = getenv("PREFETCH_SYNC");
    int sync = syncstr ? atoi(syncstr) : 0; // default async so we don't block app startup
    const char* delaystr = getenv("PREFETCH_DELAY_SEC");
    if (delaystr) {
        int d = atoi(delaystr);
        if (d > 0) sleep(d);
    }

    // log constructor start
    {
        FILE *lf = fopen("/root/AI4Memv2/preload_replayer.log", "a");
        if (lf) {
            char tb[64];
            time_t t = time(NULL);
            strftime(tb, sizeof(tb), "%F %T", localtime(&t));
            fprintf(lf, "%s: preload_replayer init: csv=%s device=%d sync=%d delay=%s\n", tb, csv, device, sync, delaystr ? delaystr : "0");
            fclose(lf);
        }
    }

    // Initial prefetch once (best effort)
    {
        FILE *lf = fopen("/root/AI4Memv2/preload_replayer.log", "a");
        if (lf) {
            char tb[64];
            time_t t = time(NULL);
            strftime(tb, sizeof(tb), "%F %T", localtime(&t));
            fprintf(lf, "%s: calling initial prefetch for %s\n", tb, csv);
            fclose(lf);
        }
    }
    call_prefetcher(csv, device, sync);

    // Start watcher thread to re-run whenever the CSV is rewritten (model updates ~5s)
    watcher_config_t* cfg = (watcher_config_t*)malloc(sizeof(watcher_config_t));
    if (!cfg) return;
    cfg->csv_path = csv;
    cfg->device = device;
    cfg->sync = sync;
    const char* poll = getenv("PREFETCH_POLL_MS");
    cfg->poll_ms = poll ? atoi(poll) : 1000; // default 1s

    pthread_t th;
    if (pthread_create(&th, NULL, watch_and_prefetch, cfg) == 0) {
        pthread_detach(th);
        FILE *lf = fopen("/root/AI4Memv2/preload_replayer.log", "a");
        if (lf) {
            char tb[64];
            time_t t = time(NULL);
            strftime(tb, sizeof(tb), "%F %T", localtime(&t));
            fprintf(lf, "%s: started watcher thread (poll_ms=%d)\n", tb, cfg->poll_ms);
            fclose(lf);
        }
    } else {
        free(cfg);
    }
}