#pragma once
// Lightweight persistent thread pool used to parallelize matmul_int8 across
// output neurons. Workers stay alive for the lifetime of the pool (spawned
// once) instead of being created/destroyed per matmul call -- on a phone-
// class CPU, thread creation overhead would otherwise dominate for the
// small-ish matmuls in this model (dim=2048), especially since matmul_int8
// is called ~7x per layer x 22 layers x every generated token.
//
// parallel_for(n, f) splits [0, n) into contiguous chunks (one per worker)
// and calls f(begin, end) on each chunk concurrently, then blocks until all
// chunks finish. Each chunk writes to a disjoint range of the caller's
// output buffer, so this is safe with no risk of data races or floating-
// point reduction reordering: results are bit-identical to a single-
// threaded loop over the same range.

#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>
#include <atomic>
#include <algorithm>
#include <utility>

namespace threadpool {

class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads) : n_threads_(n_threads) {
        if (n_threads_ < 1) n_threads_ = 1;
        workers_.reserve(n_threads_);
        for (size_t i = 0; i < n_threads_; ++i) {
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            shutdown_ = true;
            generation_++;
        }
        cv_start_.notify_all();
        for (auto& t : workers_) t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    size_t size() const { return n_threads_; }

    // Splits [0, n) into up to n_threads_ contiguous chunks and runs
    // fn(begin, end) for each chunk on a worker thread. Blocks until all
    // chunks complete. Safe to call repeatedly from the same (single)
    // calling thread; not reentrant/thread-safe for concurrent callers.
    void parallel_for(int n, const std::function<void(int, int)>& fn) {
        if (n <= 0) return;
        size_t workers_needed = std::min(n_threads_, static_cast<size_t>(n));
        if (workers_needed <= 1) {
            fn(0, n);
            return;
        }

        fn_ = &fn;
        pending_ = workers_needed;
        int chunk = (n + static_cast<int>(workers_needed) - 1) / static_cast<int>(workers_needed);

        chunks_.clear();
        for (size_t w = 0; w < workers_needed; ++w) {
            int begin = static_cast<int>(w) * chunk;
            int end = std::min(n, begin + chunk);
            if (begin < end) chunks_.push_back({begin, end});
        }
        pending_ = chunks_.size();

        {
            std::lock_guard<std::mutex> lock(mu_);
            active_chunks_ = chunks_.size();
            generation_++;
        }
        cv_start_.notify_all();

        std::unique_lock<std::mutex> lock(mu_);
        cv_done_.wait(lock, [this] { return pending_.load() == 0; });
    }

private:
    void worker_loop(size_t worker_index) {
        uint64_t seen_generation = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mu_);
            cv_start_.wait(lock, [this, &seen_generation] {
                return shutdown_ || generation_ != seen_generation;
            });
            if (shutdown_) return;
            seen_generation = generation_;
            if (worker_index >= active_chunks_) continue; // fewer chunks than workers this round
            auto [begin, end] = chunks_[worker_index];
            const std::function<void(int, int)>* fn = fn_;
            lock.unlock();

            (*fn)(begin, end);

            {
                std::lock_guard<std::mutex> lock2(mu_);
                if (--pending_ == 0) cv_done_.notify_one();
            }
        }
    }

    size_t n_threads_;
    std::vector<std::thread> workers_;

    std::mutex mu_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    uint64_t generation_ = 0;
    bool shutdown_ = false;

    std::vector<std::pair<int, int>> chunks_;
    size_t active_chunks_ = 0;
    std::atomic<size_t> pending_{0};
    const std::function<void(int, int)>* fn_ = nullptr;
};

// Global pool shared by all MixedPrecisionTransformer instances in the
// process.
//
// DISABLED (size=1) after on-device testing on the actual target hardware
// (Infinix Hot 50 / Dimensity 6300) showed multi-threading LOSING to plain
// single-thread in every real end-to-end trial, despite an isolated matmul
// microbenchmark suggesting otherwise:
//   threads=1 (no pool):        1,43 tok/s  <- winner, every time
//   threads=8 (naive equal-split): 1,31 tok/s
//   threads=6 (tuned via on-device sweep): 1,36 tok/s
// Likely cause: only the large matmuls parallelize; attention, RMSNorm,
// softmax, and RocketKV eviction bookkeeping stay single-threaded, so the
// synchronization overhead paid on every matmul call erodes more than the
// parallel portion gains. The infrastructure (ThreadPool, ranged
// matmul_int8_scalar/neon) is kept in place -- tested, correct, race-free
// (ASan/TSan clean) -- in case a future change (e.g. coarser-grained
// parallelism across whole layers, or a bigger model where matmuls
// dominate more of the wall-clock time) makes it worth revisiting. See
// notes/TINYLLAMA_EXPORT_QUANTIZATION_REPORT.md Bagian 16-19.
inline ThreadPool& global_pool() {
    static ThreadPool pool(1);
    return pool;
}

} // namespace threadpool
