#pragma once

#include <condition_variable>
#include <exception>
#include <future>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

class media_worker {
  public:
    explicit media_worker(std::string name);
    ~media_worker();

    media_worker(const media_worker &) = delete;
    media_worker & operator=(const media_worker &) = delete;

    const std::string & name() const;
    std::thread::id thread_id() const;

    template <typename Fn>
    auto invoke(const char * stage, Fn && fn) -> std::invoke_result_t<Fn> {
        using result_t = std::invoke_result_t<Fn>;

        const std::string stage_name = stage != nullptr ? stage : "unknown";
        auto promise = std::make_shared<std::promise<result_t>>();
        auto fn_ptr = std::make_shared<std::decay_t<Fn>>(std::forward<Fn>(fn));
        auto fut = promise->get_future();
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (stop_) {
                throw std::runtime_error("media backend '" + name_ + "' worker is stopped");
            }
            queue_.push([this, stage_name, promise, fn_ptr]() mutable {
                std::ostringstream tid;
                tid << std::this_thread::get_id();
                const std::string tid_str = tid.str();

                std::cerr << "[media-worker] backend '" << name_ << "' stage '" << stage_name
                          << "' begin on thread " << tid_str << "\n";
                try {
                    if constexpr (std::is_void_v<result_t>) {
                        (*fn_ptr)();
                        std::cerr << "[media-worker] backend '" << name_ << "' stage '" << stage_name
                                  << "' end on thread " << tid_str << "\n";
                        promise->set_value();
                    } else {
                        result_t result = (*fn_ptr)();
                        std::cerr << "[media-worker] backend '" << name_ << "' stage '" << stage_name
                                  << "' end on thread " << tid_str << "\n";
                        promise->set_value(std::move(result));
                    }
                } catch (const std::invalid_argument & e) {
                    const std::string message = "media backend '" + name_ + "' worker rejected " +
                                                stage_name + " on thread " + tid_str + ": " + e.what();
                    std::cerr << "[media-worker] " << message << "\n";
                    promise->set_exception(std::make_exception_ptr(std::invalid_argument(message)));
                } catch (const std::exception & e) {
                    const std::string message = "media backend '" + name_ + "' worker failed during " +
                                                stage_name + " on thread " + tid_str + ": " + e.what();
                    std::cerr << "[media-worker] " << message << "\n";
                    promise->set_exception(std::make_exception_ptr(std::runtime_error(message)));
                } catch (...) {
                    const std::string message = "media backend '" + name_ + "' worker failed during " +
                                                stage_name + " on thread " + tid_str + ": unknown exception";
                    std::cerr << "[media-worker] " << message << "\n";
                    promise->set_exception(std::make_exception_ptr(std::runtime_error(message)));
                }
            });
        }
        cv_.notify_one();
        return fut.get();
    }

  private:
    void loop();

    std::string name_;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> queue_;
    bool stop_ = false;
    std::thread thread_;
};
