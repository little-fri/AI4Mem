#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdlib>
#include <cctype>
#include <stdint.h>
#include <unistd.h>
#include "uvm_prefetcher.h"

// A tiny heuristic model:
// - parse CSV lines from data_collector (header: Category,EventName,Address,...)
// - for each UVM line, extract EventName and Address
// - score addresses: GPU_PAGE_FAULT += 5, MIGRATE_HtoD += 2, MIGRATE_DtoH += 1
// - aggregate scores per page (page-aligned)
// - select pages with score >= threshold, record them and trigger prefetch

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <uvm_csv_file> [threshold=5] [device=0] [sync=1]" << std::endl;
        return 1;
    }

    const char *csv = argv[1];
    int threshold = 5;
    int device = 0;
    int sync = 1;
    if (argc >= 3) threshold = atoi(argv[2]);
    if (argc >= 4) device = atoi(argv[3]);
    if (argc >= 5) sync = atoi(argv[4]);

    std::ifstream ifs(csv);
    if (!ifs.is_open()) {
        std::cerr << "Cannot open CSV file: " << csv << std::endl;
        return 2;
    }

    size_t page = (size_t)sysconf(_SC_PAGESIZE);

    std::unordered_map<uint64_t, int> score;
    std::string line;
    // skip header if exists
    if (std::getline(ifs, line)) {
        // if header contains the word "Category", treat as header; else process it
        if (line.find("Category") == std::string::npos) {
            // process first line
            std::istringstream ss(line);
            std::string col;
            if (std::getline(ss, col, ',')) {
                if (col == "UVM") {
                    // second column event
                    std::string event; std::getline(ss, event, ',');
                    std::string addrstr; std::getline(ss, addrstr, ',');
                    // parse address
                    uint64_t addr = 0;
                    try {
                        if (addrstr.rfind("0x", 0) == 0 || addrstr.rfind("0X", 0) == 0)
                            addr = std::stoull(addrstr, nullptr, 16);
                        else addr = std::stoull(addrstr, nullptr, 10);
                        addr = (addr / page) * page;
                        int add = 0;
                        if (event == "GPU_PAGE_FAULT") add = 5;
                        else if (event == "MIGRATE_HtoD") add = 2;
                        else if (event == "MIGRATE_DtoH") add = 1;
                        if (add) score[addr] += add;
                    } catch (...) {}
                }
            }
        }
    }

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string col;
        if (!std::getline(ss, col, ',')) continue;
        if (col != "UVM") continue;
        std::string event; if (!std::getline(ss, event, ',')) continue;
        std::string addrstr; if (!std::getline(ss, addrstr, ',')) continue;
        // trim
        while (!addrstr.empty() && isspace((unsigned char)addrstr.front())) addrstr.erase(addrstr.begin());
        while (!addrstr.empty() && isspace((unsigned char)addrstr.back())) addrstr.pop_back();
        if (addrstr.empty()) continue;
        uint64_t addr = 0;
        try {
            if (addrstr.rfind("0x", 0) == 0 || addrstr.rfind("0X", 0) == 0)
                addr = std::stoull(addrstr, nullptr, 16);
            else addr = std::stoull(addrstr, nullptr, 10);
        } catch (...) { continue; }
        addr = (addr / page) * page;
        int add = 0;
        if (event == "GPU_PAGE_FAULT") add = 5;
        else if (event == "MIGRATE_HtoD") add = 2;
        else if (event == "MIGRATE_DtoH") add = 1;
        if (add) score[addr] += add;
    }

    ifs.close();

    // collect selected pages
    std::vector<uint64_t> selected;
    for (auto &p : score) {
        if (p.second >= threshold) selected.push_back(p.first);
    }

    if (selected.empty()) {
        std::cout << "No pages selected for prefetch (threshold=" << threshold << ")" << std::endl;
        return 0;
    }

    std::cout << "Selected " << selected.size() << " pages for prefetch. Starting prefetcher..." << std::endl;
    // start prefetcher thread (short interval)
    uvm_prefetcher_start(1, device);

    for (uint64_t a : selected) {
        uvm_prefetcher_record_page(a);
    }

    // trigger immediate prefetch
    uvm_prefetcher_trigger_once();

    if (sync) sleep(2);

    uvm_prefetcher_stop();

    std::cout << "Prefetch requested for " << selected.size() << " pages." << std::endl;
    return 0;
}
