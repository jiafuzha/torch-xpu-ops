/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <vector>
#include <future>
#include <chrono>

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/initialize_block.hpp"

#include "common/sycl_cute_common.hpp"
#include "common/ipc_symm_common.hpp"


constexpr int kMaxWorldSize = 32;
constexpr int kNumSignals = 64;
constexpr int32_t kLocalCopyThreadPerBlock = 128;

constexpr int SPLIT = 1;

inline void *
ptr_offset(void *ptr, size_t offset) {
  return static_cast<char *>(ptr) + offset;
}

using namespace cute;

// ViSA no-op used as a device-side spin hint in polling loops.
CUTE_HOST_DEVICE void visa_spin_hint() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
    asm volatile(
            "{\n"
            ".decl SPIN_HINT v_type=G type=UD num_elts=1 align=4\n"
            "mov (M1_NM, 1) SPIN_HINT(0,0)<1> 0:d\n"
            "}\n"
            :
            :
            : "memory");
#endif
}

size_t get_local_copy_max_block_num(size_t num_input, int32_t pack_size = 1) {
  size_t total_blocks =
      (num_input / pack_size + kLocalCopyThreadPerBlock - 1) / kLocalCopyThreadPerBlock + 2;
  return total_blocks;
}

template <
    typename TA,
    char LayoutKindA,
    typename TensorA_t,
    typename TensorBarrierBuffer_t,
    typename TensorSyncBuffer_t,
    typename TensorSyncPtrBuffer_t>
class AllGather {

public:
    AllGather(int m, int n, int k, int rank, int world_size): m(m), n(n), k(k), rank(rank), world_size(world_size){
        initialize();
    }

    ~AllGather() {
        release();
    }

    void initialize() {
        int max_m_dim = m * world_size;
        // input buffers
        std::tie(this->input_buffers, this->input_ptrs) = create_ipc_symm_tensors<TensorA_t, TA, LayoutKindA>({max_m_dim, k});
        this->input_buffer_ = this->input_buffers[rank];

        // Device-visible barrier slots: one slot per rank in each process-local buffer.
        std::vector<void *> temp_ptrs;
        std::tie(this->sync_buffers, temp_ptrs) = create_ipc_symm_tensors<TensorSyncBuffer_t, int32_t, '-'>({world_size});

        sycl::queue& init_Q = symm::ipc_symm_get_queue();
        init_Q.memset(this->sync_buffers[rank].data().get(), 0, world_size * sizeof(int32_t)).wait();
        for (int i = 0; i < world_size; ++i) {
            this->sync_ptrs.push_back(static_cast<int32_t *>(temp_ptrs[i]));
        }
        int sync_ptrs_buffer_size = sizeof(int32_t *) * world_size;
        this->sync_ptrs_usm_buffer = make_shared_usm_tensor_init<char, '-'>(init_Q, {sync_ptrs_buffer_size}, false);
        init_Q.memcpy(this->sync_ptrs_usm_buffer.data().get(), this->sync_ptrs.data(), sync_ptrs_buffer_size).wait();
    }

    void barrier_all() {
        int32_t** sync_buffer_ptr = reinterpret_cast<int32_t**>(this->sync_ptrs_usm_buffer.data().get());
        int rank = this->rank;
        int world_size = this->world_size;
        int32_t epoch = barrier_epoch_;
        sycl::queue& Q = symm::ipc_symm_get_queue();

        Q.submit([&](sycl::handler &h) {
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(world_size), sycl::range<1>(world_size)),
                [=](sycl::nd_item<1> item) {
                    int tid = item.get_local_id(0);
                    if (tid >= world_size) {
                        return;
                    }

                    // Arrive: write this rank's epoch into each peer's inbox slot [rank].
                    int32_t* remote_inbox = sync_buffer_ptr[tid] + rank;
                    *remote_inbox = epoch;
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);

                    // Wait: spin until every peer has written epoch into my inbox slot [tid].
                    int32_t* local_slot = sync_buffer_ptr[rank] + tid;
                    while (true) {
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        if (*local_slot == epoch) {
                            break;
                        }
                        visa_spin_hint();
                    }
                });
        }).wait();
        ++barrier_epoch_;
    }

    void sync_signals(sycl::queue& Q, int remote_rank, int signal_rank_to_query) {
        int32_t** sync_buffer_ptr = reinterpret_cast<int32_t**>(this->sync_ptrs_usm_buffer.data().get());
        int rank = this->rank;
        int epoch = barrier_epoch_;
        Q.submit([&](sycl::handler& cgh) {
            cgh.single_task([=]() {
                    int32_t* remote_inbox = sync_buffer_ptr[remote_rank] + rank;
                    *remote_inbox = epoch;
                    sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                    // wait for signal from remote rank
                    int32_t* signal_rank_inbox = sync_buffer_ptr[rank] + signal_rank_to_query;
                    while (true) {
                        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
                        if (*signal_rank_inbox == epoch) {
                            break;
                        }
                        visa_spin_hint();
                    }
            });
        });
    }

    template <typename Tensor_t, typename Element_t, char LayoutKind>
    auto create_ipc_symm_tensors(const std::vector<int>& shape) {
        sycl::queue& Q = symm::ipc_symm_get_queue();
        auto [local_raw_ptr, local_layout_a] = symm::make_ipc_symm_raw_ptr_and_layout<Element_t, LayoutKind>(Q, shape);
        auto local_engine = make_gmem_ptr(static_cast<Element_t*>(local_raw_ptr));

        symm::ipc_symm_rendezvous(local_raw_ptr);

        std::vector<Tensor_t> input_buffers;
        std::vector<void *> input_ptrs;
        input_buffers.reserve(world_size);
        input_ptrs.reserve(world_size);
        for (int i = 0; i < world_size; ++i) {
            if (rank == i) {
                input_buffers.push_back(make_tensor(local_engine, local_layout_a));
                input_ptrs.push_back(local_raw_ptr);
            } else {
                void *remote_raw_ptr = symm::ipc_symm_get_remote_ptr(local_raw_ptr, i);
                input_buffers.push_back(make_tensor(make_gmem_ptr(static_cast<Element_t*>(remote_raw_ptr)), local_layout_a));
                input_ptrs.push_back(remote_raw_ptr);
            }
        }
        return std::make_tuple(input_buffers, input_ptrs);   
    }

    void local_copy_tensor_to_shmem(TensorA_t const& inputA) {
        sycl::queue& Q = symm::ipc_symm_get_queue();
        size_t chunk_size = inputA.size() * sizeof(TA);
        assert(chunk_size == sizeof(TA) * m * k);
        void *input_ptr = inputA.data().get();
        void *input_buffer_ptr = this->input_buffer_.data().get();
        Q.memcpy(ptr_offset(input_buffer_ptr, rank * chunk_size), input_ptr, chunk_size);
    }

    void copy_to_local(sycl::queue& Q, TensorA_t const& inputA, int remote_rank) {
        size_t chunk_size = inputA.size() * sizeof(TA);
        // TODO: replace with no event version of memcpy
        Q.memcpy(
            ptr_offset(this->input_ptrs[rank], remote_rank * chunk_size),
            ptr_offset(this->input_ptrs[remote_rank], remote_rank * chunk_size),
            chunk_size);
    }

    void copy_to_remote(sycl::queue& Q, TensorA_t const& inputA, int remote_rank) {
        size_t chunk_size = inputA.size() * sizeof(TA);
        // TODO: replace with no event version of memcpy
        Q.memcpy(
            ptr_offset(this->input_ptrs[remote_rank], rank * chunk_size),
            ptr_offset(this->input_ptrs[rank], rank * chunk_size),
            chunk_size);
    }

    void run_phase1(TensorA_t const& inputA) {
        // only use copy engine for now
        local_copy_tensor_to_shmem(inputA);
    }

    void run_phase2(sycl::queue& Q, TensorA_t const& inputA, int remote_rank, int signal_rank_to_query) {
        // remote copy to local buffer of my rank
        if (signal_rank_to_query == -1) {
            copy_to_local(Q, inputA, remote_rank);
        } else {
            copy_to_remote(Q, inputA, remote_rank);
            ++barrier_epoch_;
            sync_signals(Q, remote_rank, signal_rank_to_query);
        }
    }

    TensorA_t& get_local_input_buffer() {
        return this->input_buffer_;
    }

    std::vector<TensorA_t>& get_input_buffers() {
        return this->input_buffers;
    }

    void release() {
        sycl::queue& init_Q = symm::ipc_symm_get_queue();
        if (input_ptrs.size() > rank) {
            symm::sycl_free(input_ptrs[rank], init_Q);
        }
        input_ptrs.clear();
        input_buffers.clear();

        if (sync_ptrs.size() > rank) {
            symm::sycl_free(sync_ptrs[rank], init_Q);
        }
        sync_ptrs.clear();
        sync_buffers.clear();
        if (sync_ptrs_usm_buffer.size() > 0) {
            free_usm(init_Q, sync_ptrs_usm_buffer.data().get());
        }
    }

    int get_m() {
        return m;
    }

    int get_n() {
        return n;
    }

    int get_k() {
        return k;
    }

    int get_world_size() {
        return world_size;
    }

    int get_rank() {
        return rank;
    }
private:
    int m, k, n, world_size, rank;

    std::vector<TensorSyncBuffer_t> sync_buffers;
    std::vector<TensorA_t> input_buffers;
    TensorA_t input_buffer_;
    TensorSyncPtrBuffer_t sync_ptrs_usm_buffer;  // device-visible array of per-rank barrier slot pointers
    std::vector<void *> input_ptrs;
    std::vector<int32_t *> sync_ptrs;

    int32_t barrier_epoch_ = 1;
};