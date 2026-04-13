/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <tuple>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <new>

#include <pthread.h>
#include <signal.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>

#include <pwd.h>

#include "Macros.h"

#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <ATen/xpu/XPUContext.h>

namespace symm
{

using namespace std;

inline string get_first_set_env(initializer_list<const char*> var_names, const char* default_value = "") {
  for (const char* var_name : var_names) {
    const char* value = getenv(var_name);
    if (value != nullptr && value[0] != '\0') {
      return string(value);
    }
  }
  return string(default_value);
}

inline string make_launch_unique_token() {
  string torchrun_id = get_first_set_env({"TORCHELASTIC_RUN_ID"});
  if (!torchrun_id.empty()) {
    return torchrun_id;
  }

  string torchrun_endpoint = get_first_set_env({"MASTER_ADDR"});
  string torchrun_port = get_first_set_env({"MASTER_PORT"});
  if (!torchrun_endpoint.empty() && !torchrun_port.empty()) {
    return torchrun_endpoint + "-" + torchrun_port;
  }

  string job_scoped_id = get_first_set_env({
      "I_MPI_HYDRA_UUID",
      "PMI_JOBID",
      "SLURM_JOB_ID",
      "PBS_JOBID",
      "LSB_JOBID",
      "PMIX_NAMESPACE",
      "OMPI_MCA_orte_precondition_transports",
      "OMPI_COMM_WORLD_SESSION_NUM",
      "MPIRUN_JOBID"});
  if (!job_scoped_id.empty()) {
    return job_scoped_id;
  }
  return to_string(getppid());
}

inline const string& make_ipc_resource_suffix() {
  static const string suffix = to_string(getuid()) + "-" + make_launch_unique_token();
  return suffix;
}

sycl::queue create_queue(int rank, int world_size, bool in_order=true) {
    auto platforms = sycl::platform::get_platforms();
    for (const auto &platform : platforms) {
        if (platform.get_backend() == sycl::backend::ext_oneapi_level_zero) {
          if (platform.get_devices().size() >= world_size) {
            if (in_order) {
              return sycl::queue(platform.get_devices()[rank], {sycl::property::queue::in_order{}});
            } else {
              return sycl::queue(platform.get_devices()[rank]);
            }
          } else {
            if (in_order) {
              return sycl::queue({sycl::property::queue::in_order{}});
            } else {
              return sycl::queue();
            }
          }
        }
    }
    throw runtime_error("Level-Zero platform not found.");
}

// Create Unix domain socket server for receiving IPC handles
int create_server_socket(const string sockname) {
  int status = unlink(sockname.c_str());  // Remove old socket file if exists
  if (status != 0 && errno != ENOENT) {
    perror((string("unlink failed, ") + sockname).c_str());
    return -1;
  }
  sockaddr_un addr = {.sun_family = AF_UNIX, .sun_path = ""};
  copy(sockname.begin(), sockname.end(), addr.sun_path);

  int sock = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (sock == -1) {
    perror("socket creation failed");
    return -1;
  }

  if (bind(sock, (sockaddr*)&addr, SUN_LEN(&addr)) == -1) {
    perror("bind failed");
    return -1;
  }
  return sock;
}

void close_server_socket(const string server_socket_name, int sock_fd) {
  if (sock_fd >= 0) {
    if (close(sock_fd) == -1) {
      perror("socket close failed");
    }
    if (unlink(server_socket_name.c_str()) != 0 && errno != ENOENT) {
      perror((string("socket unlink failed, ") + server_socket_name).c_str());
    }
  }
}

void send_fd_no_connection(int socket, const string remote_sockname, int fd, int rank, size_t offset) {
  sockaddr_un addr = {.sun_family = AF_UNIX, .sun_path = ""};
  copy(remote_sockname.begin(), remote_sockname.end(), addr.sun_path);

  auto rank_offset = make_pair(rank, offset);

  // Prepare data to send
  // Data being sent is "fd", the value of fd will be sent as auxiliary data
  // (control message)
  iovec io = {.iov_base = &rank_offset, .iov_len = sizeof(rank_offset)};

  // Prepare control message data buffer and zero it out
  // NOLINTNEXTLINE(*array*)
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Create message header
  msghdr msg {
    // destination socket address and size of it
    // message content in msg_iov and number of such structs (1 in our case)
    // auxiliary data with the value of fd and size of it
    .msg_name = (void*)&addr, .msg_namelen = sizeof(sockaddr_un),
    .msg_iov = &io, .msg_iovlen = 1, .msg_control = cbuf,
    .msg_controllen = sizeof(cbuf), .msg_flags = 0
  };

  // This points to the first control message header
  // With SCM_RIGHTS we let the kernel know that we are passing file
  // descriptors.
  auto cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  // Specify socket level message
  cmsg->cmsg_level = SOL_SOCKET;
  // SCM_RIGHTS is the type used to pass file descriptors
  cmsg->cmsg_type = SCM_RIGHTS;

  if (fd != -1) {
    copy(
        reinterpret_cast<const char*>(&fd),
        reinterpret_cast<const char*>(&fd) + sizeof(fd),
        reinterpret_cast<char*>(CMSG_DATA(cmsg)));
  } else {
    msg.msg_controllen = 0;
  }

  // Retry sending with exponential backoff (wait for destination socket to be
  // ready)
  const int max_retries = 100;
  int retry = 0;
  ssize_t result = -1;

  while (retry < max_retries) {
    result = sendmsg(socket, &msg, 0);
    if (result > 0) {
      return; // Success
    }

    // Check if error is because destination doesn't exist yet
    if (errno == ENOENT || errno == ECONNREFUSED) {
      // Exponential backoff: 1ms, 2ms, 4ms, ..., up to 100ms
      int sleep_ms = min(1 << retry, 100);
      usleep(sleep_ms * 1000);
      retry++;
      continue;
    }

    // Other errors should fail immediately
    break;
  }

  // Finally check if we succeeded or report error
  COND_CHECK(
      result > 0,
      "Failed to send fd after ",
      retry,
      " retries: ",
      errno);
}

std::tuple<int, int, size_t> recv_fd_no_connection(int socket, const string remote_sockname) {
  // Prepare buffer for regular message "fd"
  // NOLINTNEXTLINE(*array*)
  pair<int, size_t> rank_offset;
  struct iovec io = {.iov_base = &rank_offset, .iov_len = sizeof(rank_offset)};

  // Prepare buffer for control message and zero it out
  // NOLINTNEXTLINE(*array*)
  char cbuf[CMSG_SPACE(sizeof(int))];
  memset(cbuf, 0, sizeof(cbuf));

  // Define socket address to receive on: family AF_UNIX means unix domain
  // socket
  struct sockaddr_un addr = {.sun_family = AF_UNIX, .sun_path = ""};
  copy(remote_sockname.begin(), remote_sockname.end(), addr.sun_path);

  // Prepare message header
  struct msghdr msg = {
      .msg_name = (void*)&addr,
      .msg_namelen = sizeof(struct sockaddr_un),
      .msg_iov = &io,
      .msg_iovlen = 1,
      .msg_control = cbuf,
      .msg_controllen = sizeof(cbuf),
      .msg_flags = 0};

  // Receive message on socket_
  COND_CHECK(
      recvmsg(socket, &msg, 0) > 0,
      "Failed to receive fd: ",
      errno);

  if (msg.msg_controllen == 0) {
    return make_tuple(-1, -1, -1);
  }

  // Extract control message and validate its content
  auto cmsg = CMSG_FIRSTHDR(&msg);
  COND_CHECK(cmsg != nullptr);
  COND_CHECK(cmsg->cmsg_len == CMSG_LEN(sizeof(int)));
  COND_CHECK(cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS);
  return make_tuple(*reinterpret_cast<int*>(CMSG_DATA(cmsg)), rank_offset.first, rank_offset.second);
}

inline void close_fd_if_valid(int fd) {
  if (fd < 0) {
    return;
  }
  while (close(fd) == -1 && errno == EINTR) {
  }
}

inline void unlink_if_exists(const string& path, const string& error_context) {
  if (path.empty()) {
    return;
  }
  if (unlink(path.c_str()) != 0 && errno != ENOENT) {
    perror(error_context.c_str());
  }
}

struct CleanupRegistry {
  int socket_fd = -1;
  int barrier_fd = -1;
  char socket_path[108] = {};
  char barrier_path[256] = {};
};

inline CleanupRegistry& get_cleanup_registry() {
  static CleanupRegistry registry;
  return registry;
}

inline void copy_path_for_cleanup(char* destination, size_t destination_size, const string& source) {
  if (destination_size == 0) {
    return;
  }
  strncpy(destination, source.c_str(), destination_size - 1);
  destination[destination_size - 1] = '\0';
}

inline void clear_cleanup_registry() {
  auto& registry = get_cleanup_registry();
  registry.socket_fd = -1;
  registry.barrier_fd = -1;
  registry.socket_path[0] = '\0';
  registry.barrier_path[0] = '\0';
}

inline void update_cleanup_registry(int socket_fd,
                                    const string& socket_path,
                                    int barrier_fd,
                                    const string& barrier_path) {
  auto& registry = get_cleanup_registry();
  registry.socket_fd = socket_fd;
  registry.barrier_fd = barrier_fd;
  copy_path_for_cleanup(registry.socket_path, sizeof(registry.socket_path), socket_path);
  copy_path_for_cleanup(registry.barrier_path, sizeof(registry.barrier_path), barrier_path);
}

inline void cleanup_registered_resources() {
  auto& registry = get_cleanup_registry();
  int socket_fd = registry.socket_fd;
  int barrier_fd = registry.barrier_fd;
  registry.socket_fd = -1;
  registry.barrier_fd = -1;

  close_fd_if_valid(socket_fd);
  close_fd_if_valid(barrier_fd);

  if (registry.socket_path[0] != '\0') {
    unlink(registry.socket_path);
    registry.socket_path[0] = '\0';
  }
  if (registry.barrier_path[0] != '\0') {
    unlink(registry.barrier_path);
    registry.barrier_path[0] = '\0';
  }
}

inline void ipc_resource_signal_handler(int signum) {
  cleanup_registered_resources();
  _exit(128 + signum);
}

inline void install_cleanup_handlers_once() {
  static once_flag cleanup_once;
  call_once(cleanup_once, []() {
    atexit(cleanup_registered_resources);

    struct sigaction action {};
    action.sa_handler = ipc_resource_signal_handler;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;

    for (int signal_number : {SIGINT, SIGTERM, SIGHUP, SIGABRT, SIGQUIT}) {
      sigaction(signal_number, &action, nullptr);
    }
  });
}

class ProcessBarrier {
public:
  ProcessBarrier(int local_rank, int world_size, string barrier_file_path)
      : local_rank_(local_rank),
        world_size_(world_size),
        barrier_file_path_(std::move(barrier_file_path)) {}

  ProcessBarrier(const ProcessBarrier&) = delete;
  ProcessBarrier& operator=(const ProcessBarrier&) = delete;

  void initialize() {
    if (initialized_ || world_size_ <= 1) {
      initialized_ = true;
      return;
    }

    if (local_rank_ == 0) {
      unlink_if_exists(barrier_file_path_, "barrier unlink failed");
      barrier_fd_ = open(barrier_file_path_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
      COND_CHECK(barrier_fd_ >= 0,
                 "Failed to create barrier file ",
                 barrier_file_path_,
                 ": ",
                 errno);
      COND_CHECK(ftruncate(barrier_fd_, sizeof(SharedBarrierState)) == 0,
                 "Failed to resize barrier file ",
                 barrier_file_path_,
                 ": ",
                 errno);
      map_state();
      new (state_) SharedBarrierState{};

      pthread_barrierattr_t barrier_attr;
      COND_CHECK(pthread_barrierattr_init(&barrier_attr) == 0,
                 "Failed to initialize barrier attributes");
      COND_CHECK(pthread_barrierattr_setpshared(&barrier_attr, PTHREAD_PROCESS_SHARED) == 0,
                 "Failed to configure process-shared barrier");
      COND_CHECK(pthread_barrier_init(&state_->barrier, &barrier_attr, world_size_) == 0,
                 "Failed to initialize shared barrier");
      COND_CHECK(pthread_barrierattr_destroy(&barrier_attr) == 0,
                 "Failed to destroy barrier attributes");

      state_->world_size = world_size_;
      state_->magic.store(kBarrierMagic, std::memory_order_release);
    } else {
      open_existing_state();
      while (state_->magic.load(std::memory_order_acquire) != kBarrierMagic) {
        sched_yield();
      }
      COND_CHECK(state_->world_size == world_size_,
                 "Shared barrier world size mismatch. expected=",
                 world_size_,
                 ", actual=",
                 state_->world_size);
    }

    initialized_ = true;
  }

  void barrier_all() const {
    if (world_size_ <= 1 || !initialized_) {
      return;
    }

    int status = pthread_barrier_wait(&state_->barrier);
    COND_CHECK(status == 0 || status == PTHREAD_BARRIER_SERIAL_THREAD,
               "Shared barrier wait failed: ",
               status);
  }

  void cleanup() {
    if (!initialized_) {
      return;
    }

    if (state_ != nullptr) {
      munmap(state_, sizeof(SharedBarrierState));
      state_ = nullptr;
    }
    close_fd_if_valid(barrier_fd_);
    barrier_fd_ = -1;
    unlink_if_exists(barrier_file_path_, "barrier unlink failed");
    initialized_ = false;
  }

  int fd() const {
    return barrier_fd_;
  }

  const string& file_path() const {
    return barrier_file_path_;
  }

private:
  struct SharedBarrierState {
    std::atomic<uint32_t> magic {0};
    uint32_t world_size = 0;
    pthread_barrier_t barrier;
  };

  static constexpr uint32_t kBarrierMagic = 0x53594d4dU;

  void map_state() {
    void* mapping = mmap(nullptr,
                         sizeof(SharedBarrierState),
                         PROT_READ | PROT_WRITE,
                         MAP_SHARED,
                         barrier_fd_,
                         0);
    COND_CHECK(mapping != MAP_FAILED,
               "Failed to mmap barrier file ",
               barrier_file_path_,
               ": ",
               errno);
    state_ = static_cast<SharedBarrierState*>(mapping);
  }

  void open_existing_state() {
    while (true) {
      barrier_fd_ = open(barrier_file_path_.c_str(), O_RDWR, 0600);
      if (barrier_fd_ >= 0) {
        struct stat barrier_stat {};
        if (fstat(barrier_fd_, &barrier_stat) == 0 &&
            barrier_stat.st_size >= static_cast<off_t>(sizeof(SharedBarrierState))) {
          map_state();
          return;
        }
        close_fd_if_valid(barrier_fd_);
        barrier_fd_ = -1;
      } else {
        COND_CHECK(errno == ENOENT,
                   "Failed to open barrier file ",
                   barrier_file_path_,
                   ": ",
                   errno);
      }
      usleep(1000);
    }
  }

  int local_rank_;
  int world_size_;
  string barrier_file_path_;
  int barrier_fd_ = -1;
  SharedBarrierState* state_ = nullptr;
  bool initialized_ = false;
};

// Union that holds either a raw device pointer or a Level-Zero IPC handle.
// On Linux the first sizeof(int) bytes of ze_ipc_mem_handle_t encode a file
// descriptor, so the two representations share the same underlying storage.
union IpcHandleOrPtr {
    void*               ptr;
    ze_ipc_mem_handle_t ipc_handle;

    IpcHandleOrPtr() { memset(this, 0, sizeof(*this)); }
    explicit IpcHandleOrPtr(void* p) : ptr(p) {}
    explicit IpcHandleOrPtr(const ze_ipc_mem_handle_t& h) : ipc_handle(h) {}

    // Convenience: read/write the embedded file descriptor (Linux-specific).
    int  fd() const  { return *reinterpret_cast<const int*>(&ipc_handle); }
    void set_fd(int f) { *reinterpret_cast<int*>(&ipc_handle) = f; }
};

// single instance class to manage symmetric shared memory
class SymmetricSharedMemory {
public:
    // Public static method to access the single instance
    static SymmetricSharedMemory& get_instance() {
        static SymmetricSharedMemory instance; // Created on first use, thread-safe in C++11+
        return instance;
    }

    // Delete copy constructor and assignment operator to prevent duplication
    SymmetricSharedMemory(const SymmetricSharedMemory&) = delete;
    SymmetricSharedMemory& operator=(const SymmetricSharedMemory&) = delete;
    // Delete move constructor and move assignment operator for completeness (C++11+)
    SymmetricSharedMemory(SymmetricSharedMemory&&) = delete;
    SymmetricSharedMemory& operator=(SymmetricSharedMemory&&) = delete;

    template <typename T>
    T* allocate_memory(size_t num_elements) {
        // Allocate memory on the device using DEFAULT_QUEUE
        T* local_ptr = sycl::malloc_device<T>(num_elements, DEFAULT_QUEUE);
        assert(local_ptr != nullptr);
        return local_ptr;
    }

    void initialize() {
        if (initialized) {
            return; // Already initialized, do nothing
        }
        lock_guard<std::mutex> lock(state_mutex); // Ensure only one thread can initialize at a time
        if (initialized) {
            return; // Another thread might have initialized while we were waiting for the lock
        }
        COND_CHECK(RANK == LOCAL_RANK, "RANK != LOCAL_RANK, It's IPC symmetric memory, all ranks should be local.", RANK, LOCAL_RANK);
        // Initialize the ZE API
        TORCH_CHECK(zeInit(0)); // Initialize the l0 drivers
        install_cleanup_handlers_once();
        proc_barrier_ptr = make_unique<ProcessBarrier>(LOCAL_RANK, WORLD_SIZE, BARRIER_FILE_PATH);
        proc_barrier_ptr->initialize();
        update_cleanup_registry(-1, "", proc_barrier_ptr->fd(), proc_barrier_ptr->file_path());
        // create server socket
        server_socket_name = SERVER_SOCKET_NAME_PREFIX + to_string(LOCAL_RANK);
        socket_fd = create_server_socket(server_socket_name);

        assert(socket_fd >= 0);
        update_cleanup_registry(socket_fd, server_socket_name, proc_barrier_ptr->fd(), proc_barrier_ptr->file_path());
        initialized = true;
    }

    void rendezvous(void* local_ptr, string group_name="") {
        if (!initialized) { // check and make changes visiable to all threads
            throw runtime_error("SymmetricSharedMemory must be initialized before rendezvous");
        }
        auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                      DEFAULT_QUEUE.get_context());
        auto it = local_to_remote_memory_ptrs.find(local_ptr);
        if (it != local_to_remote_memory_ptrs.end()) {
            return; // Already rendezvoused, do nothing
        }
        lock_guard<std::mutex> lock(state_mutex); // Ensure only one thread can rendezvous at a time
        // do IPC exchange to get remote pointers for local_ptr
        void* base_addr;
        size_t base_size;
        TORCH_CHECK(zeMemGetAddressRange(l0_ctx, local_ptr, &base_addr, &base_size));
        size_t offset = (char*)local_ptr - (char*)base_addr;

        ze_ipc_mem_handle_t ipc_handle;
        TORCH_CHECK(zeMemGetIpcHandle(l0_ctx, base_addr, &ipc_handle));

        int my_fd = *reinterpret_cast<int*>(&ipc_handle);
        vector<std::tuple<IpcHandleOrPtr, void*, int>> remote_infos(WORLD_SIZE); // remote_base, remote_ptr, fd
        remote_infos[LOCAL_RANK] = make_tuple(IpcHandleOrPtr(ipc_handle), local_ptr, my_fd); // local rank's own memory info

        auto l0_device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(DEFAULT_QUEUE.get_device());

        // exchange fd, skip local rank
        for (int i = LOCAL_RANK + 1, j = LOCAL_RANK - 1; i < (WORLD_SIZE + LOCAL_RANK); i++, j--) {
            int remote_rank = i % WORLD_SIZE;
            send_fd_no_connection(socket_fd, SERVER_SOCKET_NAME_PREFIX + to_string(remote_rank), my_fd, LOCAL_RANK, offset);
            int remote_rank_to_receive =  (j + WORLD_SIZE) % WORLD_SIZE;
            auto [remote_fd, remote_rank_received, remote_offset] = recv_fd_no_connection(socket_fd, SERVER_SOCKET_NAME_PREFIX + to_string(remote_rank_to_receive));
            assert(remote_rank_received == remote_rank_to_receive);
            // open remote IPC handle to get remote base address, then calculate remote pointer with offset
            // Reconstruct IPC handle using the received file descriptor
            ze_ipc_mem_handle_t remote_ipc_handle = ipc_handle; // copy constructor to set fields correctly
            *reinterpret_cast<int*>(&remote_ipc_handle) = remote_fd;

            // Open IPC handle to get remote memory pointer
            // Use BIAS_CACHED for better performance
            void* remote_base;
            TORCH_CHECK(zeMemOpenIpcHandle(l0_ctx, l0_device, remote_ipc_handle,
                                        ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &remote_base));

            float* remote_ptr = (float*)((char*)remote_base + remote_offset);
            remote_infos[remote_rank_received] = make_tuple(IpcHandleOrPtr(remote_base), remote_ptr, remote_fd);
        }
        local_to_remote_memory_ptrs[local_ptr] = remote_infos;
    }

    void* get_remote_ptr(void* local_ptr, int remote_rank) {
        if (!initialized) {
            throw runtime_error("SymmetricSharedMemory must be initialized before get_remote_ptr");
        }
        if (remote_rank == LOCAL_RANK) {
            return local_ptr; // Return local pointer for local rank
        }
        auto it = local_to_remote_memory_ptrs.find(local_ptr);
        if (it == local_to_remote_memory_ptrs.end()) {
            throw runtime_error("Local pointer not found in rendezvous: " + to_string((uintptr_t)local_ptr));
        }
        auto& remote_infos = it->second;
        if (remote_rank < 0 || remote_rank >= WORLD_SIZE) {
            throw runtime_error("Invalid remote rank: " + to_string(remote_rank));
        }
        return get<1>(remote_infos[remote_rank]);
    }

    void free(void* local_ptr) {
        if (!initialized) {
            throw runtime_error("SymmetricSharedMemory must be initialized before free");
        }
        lock_guard<std::mutex> lock(state_mutex); // Ensure only one thread can change internal state at a time
        auto it = local_to_remote_memory_ptrs.find(local_ptr);
        if (it != local_to_remote_memory_ptrs.end()) {
            auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                      DEFAULT_QUEUE.get_context());
            for (auto& [handle_or_ptr, remote_ptr, fd] : it->second) {
                release_resources_for_ptr(l0_ctx, local_ptr, handle_or_ptr, remote_ptr, fd);
            }
        } else { // if multiple threads call free() on the same pointer, throw exception
            throw runtime_error("SymmetricSharedMemory: free() called on unregistered pointer");
        }
        local_to_remote_memory_ptrs.erase(it); // Remove the entry from the map
    }
    
    void barrier_all() {
      if (proc_barrier_ptr) {
        proc_barrier_ptr->barrier_all();
      }
    }

    void finalize() {
        if (!initialized) {
            return; // Not initialized, do nothing
        }
        lock_guard<std::mutex> lock(state_mutex); // Ensure only one thread can finalize at a time
        do_finalize();
    }

    sycl::device get_device() const {
        return DEFAULT_QUEUE.get_device();
    }

    static sycl::queue& get_queue() {
        return DEFAULT_QUEUE;
    }

    static sycl::queue& get_compute_queue() {
        return COMPUTE_QUEUE;
    }

    static sycl::queue& get_adjacent_device_queue() {
        return QUEUE_ON_ADJACENT_DEVICE;
    }

    static int get_rank() {
        return LOCAL_RANK;  
    }

    static int get_world_size() {
        return WORLD_SIZE;
    }

    static std::string get_server_socket_name_prefix() {
        return SERVER_SOCKET_NAME_PREFIX;
    }

  private:
    // Private constructor to prevent direct instantiation from outside
    SymmetricSharedMemory() {
        LOCAL_RANK = stoi(get_first_set_env({"LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "PMI_LOCAL_RANK"}, "0"));
        RANK = stoi(get_first_set_env({"RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK"}, "0"));
        WORLD_SIZE = stoi(get_first_set_env({"WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE"}, "1"));
        // TODO: confirm in-order queue
        DEFAULT_QUEUE = at::xpu::getCurrentXPUStream().queue();
        COMPUTE_QUEUE = sycl::queue(DEFAULT_QUEUE.get_device(), {sycl::property::queue::in_order{}});
        SERVER_SOCKET_NAME_PREFIX = "/tmp/sycl-ipc-symm-server-" + make_ipc_resource_suffix() + "-";
        BARRIER_FILE_PATH = "/dev/shm/sycl-ipc-symm-barrier-" + make_ipc_resource_suffix();
        assert(LOCAL_RANK >= 0 && WORLD_SIZE > 0);
    }

    // Optional: Private destructor if specific cleanup is needed,
    // though the static instance is automatically destroyed at program exit.
    ~SymmetricSharedMemory() {
        do_finalize();
    }

    inline void release_resources_for_ptr(ze_context_handle_t& l0_ctx, void* local_ptr, IpcHandleOrPtr handle_or_ptr, void* remote_ptr, int fd) {
        if (fd >= 0) {
            close(fd);
        }
        if (local_ptr != remote_ptr) { // avoid closing local memory handle as remote handle
            ZE_CHECK(zeMemCloseIpcHandle(l0_ctx, handle_or_ptr.ptr));
        } else { // local rank's own memory, put back ipc handle instead of close
            // remote_base is actually local_ipc_base
            ZE_CHECK(zeMemPutIpcHandle(l0_ctx, handle_or_ptr.ipc_handle));
        }
    }

    void do_finalize() {
        if (!initialized) {
            return;
        }
        auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                      DEFAULT_QUEUE.get_context());
        for (auto& [local_ptr, remote_infos] : local_to_remote_memory_ptrs) {
            for (auto& [handle_or_ptr, remote_ptr, fd] : remote_infos) {
                release_resources_for_ptr(l0_ctx, local_ptr, handle_or_ptr, remote_ptr, fd);
            }
            // local_ptr is to be freed by the user, so no need to free it here
        }
        local_to_remote_memory_ptrs.clear();
        close_server_socket(server_socket_name, socket_fd);
        socket_fd = -1;
        server_socket_name.clear();
        if (proc_barrier_ptr) {
          proc_barrier_ptr->cleanup();
          proc_barrier_ptr.reset();
        }
        clear_cleanup_registry();
        initialized = false; // reset for safety, though the instance will likely be destroyed after this
    }

    const int LOCAL_RANK;
    const int RANK;
    const int WORLD_SIZE;
    sycl::queue COMPUTE_QUEUE;
    sycl::queue DEFAULT_QUEUE;
    const string SERVER_SOCKET_NAME_PREFIX;
    const string BARRIER_FILE_PATH;

    // local_ptr -> [{remote_ptr1_base, remote_ptr1, fd1}, {remote_ptr2_base, remote_ptr2, fd2}, ...] (multiple ranks)
    // local_ptr -> [local_rank]=> {local_ipc_base, local_ptr, fd} (for local rank)
    unordered_map<void*, vector<std::tuple<IpcHandleOrPtr, void*, int>>> local_to_remote_memory_ptrs;
    string server_socket_name;
    int socket_fd = -1;
    unique_ptr<ProcessBarrier> proc_barrier_ptr;
    volatile bool initialized = false; // to ensure visibility of initialization across threads

    std::mutex state_mutex;
};


void ipc_symm_init() {
  SymmetricSharedMemory& sym = SymmetricSharedMemory::get_instance();
  sym.initialize();
}

void ipc_symm_finalize() {
  SymmetricSharedMemory& sym = SymmetricSharedMemory::get_instance();
  sym.finalize();
}

const int ipc_symm_n_pes() {
  return SymmetricSharedMemory::get_world_size();
}

const int ipc_symm_my_pe() {
  return SymmetricSharedMemory::get_rank();
}

void * ipc_symm_get_remote_ptr(void* local_ptr, int remote_rank) {
  SymmetricSharedMemory& sym = SymmetricSharedMemory::get_instance();
  return sym.get_remote_ptr(local_ptr, remote_rank);
}

void ipc_symm_barrier_all() {
  SymmetricSharedMemory& sym = SymmetricSharedMemory::get_instance();
  sym.barrier_all();
}

void ipc_symm_rendezvous(void* local_ptr, string group_name="") {
  SymmetricSharedMemory& sym = SymmetricSharedMemory::get_instance();
  sym.rendezvous(local_ptr, group_name);
}

void ipc_symm_free(void* local_ptr) {
  SymmetricSharedMemory& sym = SymmetricSharedMemory::get_instance();
  sym.free(local_ptr);
}

sycl::device ipc_symm_get_device() {
  SymmetricSharedMemory& sym = SymmetricSharedMemory::get_instance();
  return sym.get_device();
}

sycl::queue& ipc_symm_get_queue() {
  return SymmetricSharedMemory::get_queue();
}

sycl::queue& ipc_symm_get_compute_queue() {
  return SymmetricSharedMemory::get_compute_queue();
}

///////////
} // namespace symm