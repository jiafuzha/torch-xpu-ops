#pragma once

#include "allgather.hpp"

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wpass-failed"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

template <typename T, size_t = 0>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, 0 * sizeof(T)> : std::true_type {};

template <typename T>
static constexpr bool is_complete_v = is_complete<T>::value;

using namespace cute;

template <class ATensor, class BTensor, class CTensor,
          class TiledMMA>
void gemm_device(ATensor const& A, // (M,K)
                 BTensor const& B, // (N,K)
                 CTensor& C,       // (M,N)
                 TiledMMA const& mma) {
  // -----
  // Setup
  // -----

  /* Get workgroup and local IDs */
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape()); // (M,K)
  Tensor cB = make_identity_tensor(B.shape()); // (N,K)
  Tensor cC = make_identity_tensor(C.shape()); // (M,N)

  /* Split GEMM into workgroup tiles, and identify our workgroup's tile
   * (wg_coord) */
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile),
                         make_coord(wg_m, _)); // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile),
                         make_coord(wg_n, _)); // (BLK_N,BLK_K,k)
  Tensor gC =
      local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{}); // (BLK_M,BLK_N)

  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);

  /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  /* Register fragments for MMA */
  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  /* Register fragments for copies */
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  /* Partition global tensor (proxies) for copies */
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  Tensor tCrC = partition_fragment_C(mma, select<0, 1>(wg_tile));
  Tensor tCgC =
      thr_mma.partition_C(gC); /* also matches copy_c's source layout */

  /* Create prefetch TiledCopy instances */
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  /* Partition global tensor (proxies) for prefetch */
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  /* Prefetch distance, in units of k tiles */
  const int prefetch_dist = 3;

  // ------
  // Kernel
  // ------

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  /* Clear the accumulators */
  clear(tCrC);

  /* Warm up loops with prefetch to L1 */
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  /* Main loop */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    /* Split barrier keeping threads loosely together */
    // barrier_arrive(barrier_scope);

    /* Copy A/B from global memory (ideally L1 cache) to registers */
    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    /* Prefetch A/B tiles to L1 */
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

    /* Shuffle data from copy fragments to MMA fragments */
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    /* Accumulate C += A * B */
    gemm(mma, tCrA, tCrB, tCrC);

    /* Other half of split barrier */
    // barrier_wait(barrier_scope);
  }

  /* Write C to global memory */
  copy(copy_c, tCrC, tCgC);
  // TODO: prefetch or query barrier for next run
}

template <typename TA, typename TB, typename TC>
auto choose_mma_op() {
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>)
    return XE_DPAS_TT<8, TC, TA, TB>{};
  else if constexpr (is_same_v<TA, cute::bfloat16_t>)
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  else /* Use f16 by default as upconversion sequences are typically faster */
    return XE_DPAS_TT<8, float, cute::half_t>{};
}

template <class ATensor, class BTensor, class CTensor>
auto choose_tiled_mma(ATensor const& A, BTensor const& B, CTensor const&) {
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA, TB, TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool a_t = is_constant_v<1, decltype(stride<0>(A))>;
  constexpr bool b_n = is_constant_v<1, decltype(stride<0>(B))>;

  constexpr bool use_1x_dpas_per_k =
      a_t               // Use one DPAS in k dimension for A^T case
      || (byte && b_n); //  pending compiler improvements (also int8 B^N).
  constexpr bool use_4x8_sg =
      ((sizeof_bits_v<TB> <
        sizeof_bits_v<TA>) // Use smaller B loads for expensive reorders.
       &&!(is_same_v<TB, cute::float_e5m2_t>) ) ||
      (b_n && sizeof_bits_v<TB> < 8);

  using _K = conditional_t<use_1x_dpas_per_k, C<op.K>, C<op.K * 2>>;

  using WGTile = Shape<_256, _256, _K>; // 256x256 WG tile size
  using SGLayout8x4 =
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>; // 8x4 SG tiling, n-major
  using SGLayout4x8 =
      Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>; // 4x8 SG tiling, n-major
  using SGLayout = conditional_t<use_4x8_sg, SGLayout4x8, SGLayout8x4>;

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>,
                                      SGLayout>::TiledMMA;

  return MMA{};
}

template <class, class, char, char, int>
class GemmCuteName;

struct TypeAgnosticGemmStats {
    // TODO: thread local stats to collect memory usage
}

struct TupleHash {
    size_t operator()(const std::tuple<int, int, int>& t) const {
        // Simple hash combination (e.g., using XOR and bit shifts)
        return std::get<0>(t) ^ (std::get<1>(t) << 1) ^ (std::get<2>(t) << 2);
    }
};

template <class TA, class TB, class TC, char LayoutKindA, char LayoutKindB>
class AllGatherGemm {
    using TensorA_t = Tensor<ViewEngine<gmem_ptr<TA*>>, Layout<tuple<int, int>, tuple<int, C<1>>>>;
    using TensorB_t = Tensor<ViewEngine<gmem_ptr<TB*>>, Layout<tuple<int, int>, tuple<C<1>, int>>>;
    using TensorC_t = Tensor<ViewEngine<gmem_ptr<TC*>>, Layout<tuple<int, int, int>, tuple<int, int, C<1>>>>;
    using TensorSyncBuffer_t = Tensor<ViewEngine<gmem_ptr<int32_t*>>, Layout<tuple<int>, tuple<C<1>>>>;
    using TensorBarrierBuffer_t = Tensor<ViewEngine<gmem_ptr<char*>>, Layout<tuple<int>>>;
    using TensorSyncPtrBuffer_t = Tensor<ViewEngine<gmem_ptr<char*>>, Layout<tuple<int>>>;

    public:
    static std::unique_ptr<AllGatherGemm> get_instance(int m, int n, int k) {
        // make sure ipc_symm is initialized
        symm::ipc_symm_init();
        // get exising instance or create a new one
        auto key = std::make_tuple(m, n, k);
        lock_guard<std::mutex> lock(mutex); // Ensure only one thread can initialize at a time
        auto it = instance_map.find(key);
        if (it != instance_map.end()) {
            instance_map.erase(it);
            return it->second;
        }
        return std::make_unique<AllGatherGemm>(m, n, k, ipc_symm_my_pe(), ipc_symm_n_pes());
    }

    static void release_instance(int m, int n, int k) {
        auto key = std::make_tuple(m, n, k);
        lock_guard<std::mutex> lock(mutex); // Ensure only one thread can release at a time
        instance_map.erase(key);
    }

    static void release_all_instances() {
        lock_guard<std::mutex> lock(mutex); // Ensure only one thread can release at a time
        instance_map.clear();
    }

    static void putback_instance(std::unique_ptr<AllGatherGemm> instance) {
        auto key = std::make_tuple(instance->get_m(), instance->get_n(), instance->get_k());
        lock_guard<std::mutex> lock(mutex); // Ensure only one thread can put back at a time
        auto it = instance_map.find(key);
        if (it != instance_map.end()) {
            // assume there is only one distributed gemm with same shape at same time, if there is already an instance with the same key, throw exception
            throw std::runtime_error("Instance with the same key already exists in the map");
        }
        // TODO: check if memory of cached instance exceeds a certain threshold, if so, release old instances with LRU
        instance_map[key] = std::move(instance);
    }

    template <class MMAType>
    void gemm_cute(
                sycl::queue& Q,
                TensorA_t const& max_A, // (M * world_size, K)
                int target_rank,
                TensorB_t const& B, // (N,K)
                TensorC_t& max_C,       // (world_size, M, N)
                MMAType mma) {
        namespace syclex = sycl::ext::oneapi::experimental;
        namespace intelex = sycl::ext::intel::experimental;

        syclex::properties kernel_props{syclex::sub_group_size<16>,
                                        intelex::grf_size<256>};

        int m = ag.get_m();
        int k = ag.get_k();
        int world_size = ag.get_world_size();
        int rank = ag.get_rank();

        auto mk_layout = make_layout(make_shape(m, k), LayoutRight{});
        sycl::range<2> local = {size(mma), 1};
        sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(m, get<0>(mma.tile_mnk()))};
        Q.submit([&](sycl::handler &h) {
            h.parallel_for<GemmCuteName<TA, TB, LayoutKindA, LayoutKindB, 0>>(
                sycl::nd_range<2>(global, local), kernel_props,
                [=](auto) {
                    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
                    auto C = max_C(target_rank, _, _);
                    gemm_device(make_tensor(make_gmem_ptr(max_A.data().get() + target_rank * m * k), mk_layout), B, C, mma);
                });
        });
    }

    void run(
            TensorA_t const& local_A,
            TensorB_t const& B,
            TensorC_t& max_C) {
        ag.run_phase1(local_A);
        // start gemm for local rank
        auto mma = choose_tiled_mma(local_A, B, max_C(0, _, _));
        auto input_buffer = ag.get_local_input_buffer();
        // my own rank is in input_buffer (offsetted) now, so start gemm immediately
        sycl::queue& default_queue = symm::ipc_symm_get_queue();
        gemm_cute<decltype(mma)>(default_queue, input_buffer, ag.get_rank(), B, max_C, mma);
        // copy data (push first, then pull) and run gemm in different queues in a round-robin way
        int rank = ag.get_rank();
        int world_size = ag.get_world_size();
        sycl::queue& compute_queue = symm::ipc_symm_get_compute_queue();
        // push first, then pull
        for (int i = 1, j = 0; i < world_size; ++i) {
            bool is_odd = (i % 2 != 0);
            if (i != 2) { // only push first
                j++;
            }
            // push data to remote rank only at the first time
            int remote_rank = (rank + j) % world_size;
            sycl::queue& queue = is_odd ? compute_queue : default_queue;
            ag.run_phase2(queue, local_A, remote_rank, i == 1 ? (rank + world_size - j) % world_size : -1);
            gemm_cute<decltype(mma)>(queue, input_buffer, i == 1 ? (rank + world_size - j) % world_size : remote_rank, B, max_C, mma);
        }
    }

    TensorA_t& get_local_input_buffer() {
        return ag.get_local_input_buffer();
    }

    int get_m() {
        return ag.get_m();
    }

    int get_n() {
        return ag.get_n();
    }

    int get_k() {
        return ag.get_k();
    }

    size_t get_memory_size() {
        return ag.get_memory_size();
    }

private:

    AllGatherGemm(int m, int n, int k, int rank, int world_size) : ag(m, n, k, rank, world_size) {}
    ~AllGatherGemm() {}

    AllGather<TA, LayoutKindA, TensorA_t, TensorBarrierBuffer_t, TensorSyncBuffer_t, TensorSyncPtrBuffer_t> ag;
    // assume seldom concurrent distributed gemm with the same shape and same shapes.
    static std::mutex mutex;
    static std::unordered_map<std::tuple<int, int, int>, std::unique_ptr<AllGatherGemm>, TupleHash> instance_map;
};