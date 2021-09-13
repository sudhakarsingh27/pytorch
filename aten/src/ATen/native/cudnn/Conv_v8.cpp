#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/native/cudnn/Macros.h>

#if HAS_CUDNN_V8()

#include <ATen/cudnn/cudnn-wrapper.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_find_plan.h>
#include <cudnn_frontend_get_plan.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/cudnn/ConvShared.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/TensorUtils.h>

#include <c10/util/env.h>

#include <THC/THC.h>

#include <mutex>
#include <unordered_map>

namespace at { namespace native {

namespace {

int v8_flag = -1;
int debug = -1;
auto heuristic_mode = CUDNN_HEUR_MODE_INSTANT;
int printcount = 0;

bool use_v8() {
  if (v8_flag < 0) {
    auto val = c10::utils::check_env("CUDNN_V8_API_ENABLED");
    auto debug_val = c10::utils::check_env("CUDNN_V8_API_DEBUG");
    auto heuristic_val = c10::utils::check_env("USE_HEURISTIC_MODE_B");
    if (val == true) {
      v8_flag = 1;
    } else {
      v8_flag = 0;
    }
    if (debug_val == true) {
      debug = 1;
    } else {
      debug = 0;
    }
    if (heuristic_val == true) {
      heuristic_mode = CUDNN_HEUR_MODE_B;
    }
  }
  if (debug == 1 && printcount < 10) {
    TORCH_WARN("CUDNN_V8_DEBUG ON, V8_FLAG: ", v8_flag, " HEURISTIC_MODE B: ", heuristic_mode == CUDNN_HEUR_MODE_B);
    printcount++;
  }
  return v8_flag == 1;
}

// TODO: remove duplicate code in Conv_v7.cpp
constexpr size_t operator "" _TiB(unsigned long long n) {
  return size_t(n) << 40;
}

uint8_t getAlignment(const Tensor &t) {
  // alignment are in bytes
  uint8_t alignment = 1;
  uintptr_t address = reinterpret_cast<uintptr_t>(t.data_ptr());
  while (address % alignment == 0 && alignment < 64) alignment *= 2;
  return alignment;
}

cudnn_frontend::Tensor getTensorDescriptor(const Tensor &t, const int64_t id, const uint8_t alignment) {
  auto shape = t.sizes();
  auto strides = t.strides();
  auto r = cudnn_frontend::TensorBuilder()
    .setDim(shape.size(), shape.data())
    .setStrides(strides.size(), strides.data())
    .setId(id)
    .setAlignment(alignment)
    .setDataType(getCudnnDataType(t))
    .build();
  return r;
}

cudnn_frontend::ConvDesc_v8 getConvDescriptor(cudnnDataType_t dataType, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, const at::ScalarType scalar_type) {
  uint64_t convDim = stride.size();
  if (scalar_type == kBFloat16 || scalar_type == kHalf) {
    dataType = CUDNN_DATA_FLOAT;
  }
  return cudnn_frontend::ConvDescBuilder()
    .setDataType(dataType)
    .setMathMode(CUDNN_CROSS_CORRELATION)
    .setNDims(convDim)
    .setStrides(convDim, stride.data())
    .setPrePadding(convDim, padding.data())
    .setPostPadding(convDim, padding.data())
    .setDilation(convDim, dilation.data())
    .build();
}

void filterEngineConfigs(
  cudnn_frontend::EngineConfigList &from,
  cudnn_frontend::EngineConfigList &to,
  bool deterministic, bool allow_tf32, c10::ScalarType scalar_type)
{
  auto filter = [=](cudnnBackendDescriptor_t c) {
    if (deterministic) {
      if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC>(c)) {return true;}
    }
    if (cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(c)) {return true;}
    if (scalar_type == kFloat) {
      // TODO: check under which conditions this is OK
      if (!allow_tf32 && cudnn_frontend::hasNumericalNote<CUDNN_NUMERICAL_NOTE_TENSOR_CORE>(c)) {return true;}
    }
    return false;
  };
  cudnn_frontend::filter(from, to, filter);
}

struct CacheKey {
  ConvolutionParams params;
  cudnnBackendDescriptorType_t operation;
  uint8_t x_alignment;
  uint8_t w_alignment;
  uint8_t y_alignment;
};

struct CacheKeyFused {
  ConvolutionParams params;
  // No op here because it is assumed to be a forward conv op
  uint8_t x_alignment;
  uint8_t w_alignment;
  uint8_t y_alignment;
  uint8_t z_alignment;
  uint8_t b_alignment;
  // TODO: does it make sense to have this in the key? but alpha is a graph-level param...
  float alpha;
};

template <typename T, typename KeyType>
struct BenchmarkCache {
std::mutex mutex;
std::unordered_map<KeyType, cudnn_frontend::ExecutionPlan, ParamsHash<KeyType>, ParamsEqual<KeyType>> engine_cache;

// TODO: is this thread safe if cache is updated? is pointer stale?
cudnn_frontend::ExecutionPlan* find(const KeyType& key) {
  std::lock_guard<std::mutex> guard(mutex);
  auto it = engine_cache.find(key);
  if (it == engine_cache.end()) {
    return NULL;
  }
  // TODO: probably want ExecutionPlan copy constructor or better way to return
  return &(it->second);
}

void emplace(const KeyType& key, T& results) {
  std::lock_guard<std::mutex> guard(mutex);
  engine_cache.emplace(key, std::move(results));
}

};

BenchmarkCache<cudnn_frontend::ExecutionPlan, CacheKey> benchmark_cache;
BenchmarkCache<cudnn_frontend::ExecutionPlan, CacheKeyFused> benchmark_cache_fused;

} // namespace
void get_cachekey(CacheKey& key, const cudnnBackendDescriptorType_t operation, const Tensor& y, const Tensor& x, const Tensor& w, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, int64_t groups, bool deterministic, bool allow_tf32) {
  memset(&key, 0, sizeof(key));
  setConvolutionParams(&key.params, x, w, padding, stride, dilation, groups, deterministic, allow_tf32);
  key.operation = operation;
  key.x_alignment = getAlignment(x);
  key.y_alignment = getAlignment(y);
  key.w_alignment = getAlignment(w);
}

void get_cachekey_fused(CacheKeyFused& key, const Tensor& y, const Tensor& x, const Tensor& w, const Tensor& z, const Tensor& b, const float alpha, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, int64_t groups, bool deterministic, bool allow_tf32) {
  memset(&key, 0, sizeof(key));
  setConvolutionParams(&key.params, x, w, padding, stride, dilation, groups, deterministic, allow_tf32);
  key.x_alignment = getAlignment(x);
  key.y_alignment = getAlignment(y);
  key.w_alignment = getAlignment(w);
  key.z_alignment = getAlignment(z);
  key.b_alignment = getAlignment(b);
  key.alpha = alpha;
}

void run_conv_plan(cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const cudnn_frontend::ExecutionPlan& plan) {
  auto workspace_size = plan.getWorkspaceSize();
  Tensor workspace = at::empty({workspace_size}, x.options().dtype(kByte));
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w'};
  auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace.has_storage() ? workspace.data_ptr() : NULL)
      .setDataPointers(3, data_ptrs)
      .setUids(3, uids)
      .build();
  AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
}

void run_conv_plan_fused(cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b, const cudnn_frontend::ExecutionPlan& plan) {
  auto workspace_size = plan.getWorkspaceSize();
  Tensor workspace = at::empty({workspace_size}, x.options().dtype(kByte));
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr(), z.data_ptr(), b.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w', 'z', 'b'};
  auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace.has_storage() ? workspace.data_ptr() : NULL)
      .setDataPointers(5, data_ptrs)
      .setUids(5, uids)
      .build();
  AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan.get_raw_desc(), variantPack.get_raw_desc()));
}

auto build_opgraph(const cudnnHandle_t handle, const cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, const CacheKey& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation) {
  auto op = cudnn_frontend::OperationBuilder(desc)
      .setxDesc(getTensorDescriptor(x, 'x', key.x_alignment))
      .setyDesc(getTensorDescriptor(y, 'y', key.y_alignment))
      .setwDesc(getTensorDescriptor(w, 'w', key.w_alignment))
      .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation, x.scalar_type()))
      .build();
  std::array<cudnn_frontend::Operation const *, 1> ops = {&op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(1, ops.data())
      .build();
  return opGraph;
}

auto build_opgraph_fused(const cudnnHandle_t handle, const Tensor & x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b, const float alpha, const CacheKeyFused& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation) {
  auto addDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_ADD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
  auto addBiasDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_ADD)
                            .setMathPrecision(CUDNN_DATA_FLOAT)
                            .build();
  auto actDesc = cudnn_frontend::PointWiseDescBuilder()
                           .setMode(CUDNN_POINTWISE_RELU_FWD)
                           .setMathPrecision(CUDNN_DATA_FLOAT)
                           .build();
  auto convDesc = getConvDescriptor(key.params.dataType, padding, stride, dilation, x.scalar_type());
  const float alpha1 = 1.0;
  const float alpha2 = alpha;
  auto conv_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                   .setxDesc(getTensorDescriptor(x, 'x', key.x_alignment))
                   .setyDesc(getTensorDescriptor(y, 'C', key.y_alignment))
                   .setwDesc(getTensorDescriptor(w, 'w', key.w_alignment))
                   .setAlpha(alpha1)
                   .setcDesc(getConvDescriptor(key.params.dataType, padding, stride, dilation, x.scalar_type()))
                   .build();
  auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(conv_op.getOutputTensor())
                           .setbDesc(getTensorDescriptor(z, 'z', key.z_alignment))
                           // TODO: is reusing y's alignment good here?
                           .setyDesc(getTensorDescriptor(y, 'A', key.y_alignment))
                           .setpwDesc(addDesc)
                           .setAlpha(alpha1)
                           .setAlpha2(alpha2)
                           .build();
  // auto bias_ = b.view({1, b.numel(), 1, 1});
  auto add_bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                           .setxDesc(add_op.getOutputTensor())
                           .setbDesc(getTensorDescriptor(b, 'b', key.b_alignment))
                           // TODO: is reusing y's alignment good here?
                           .setyDesc(getTensorDescriptor(y, 'B', key.y_alignment))
                           .setpwDesc(addBiasDesc)
                           .build();
  auto act_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                          .setxDesc(add_bias_op.getOutputTensor())
                          .setyDesc(getTensorDescriptor(y, 'y', key.y_alignment))
                          .setpwDesc(actDesc)
                          .build();
  std::array<cudnn_frontend::Operation const*, 4> ops = {&conv_op, &add_op, &add_bias_op, &act_op};
  auto opGraph = cudnn_frontend::OperationGraphBuilder()
                     .setHandle(handle)
                     .setOperationGraph(ops.size(), ops.data())
                     .build();
  return opGraph;
}

const auto get_generator_sources(const cudnnBackendDescriptorType_t& desc, const Tensor& x, const bool deterministic, const bool allow_tf32, const cudnnBackendHeurMode_t heur_mode) {
   // Method for engine config generator based on heuristics
  auto heurgen_method = [/*&desc,*/ &x, deterministic, allow_tf32, heur_mode](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
      auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                            .setOperationGraph(opGraph)
                            .setHeurMode(heur_mode)
                            .build();
      auto &engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
      cudnn_frontend::EngineConfigList filtered_configs;
      filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, x.scalar_type());
      return filtered_configs;
  };
  // Method for engine config generator based on fallback list
  auto fallback_method = [&desc, &x, deterministic, allow_tf32](cudnn_frontend::OperationGraph &opGraph) -> cudnn_frontend::EngineConfigList {
    auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                        .setOperationGraph(opGraph)
                        .setOperation(desc)
                        //.setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                        .build();
    auto &fallback_list = fallback.getFallbackList();
    cudnn_frontend::EngineConfigList filtered_configs;
    filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, x.scalar_type());
    return filtered_configs;
  };
  std::array<cudnn_frontend::GeneratorSource const, 2> sources = {heurgen_method, fallback_method};
  return sources;
}

size_t get_available_workspace() {
  int device;
  THCudaCheck(cudaGetDevice(&device));
  size_t max_block_size = 0;
  size_t tmp_bytes = 0;  // Only used for filling pointer parameters that aren't used later
  c10::cuda::CUDACachingAllocator::cacheInfo(device, &tmp_bytes, &max_block_size);
  return max_block_size;
}

void generate_and_filter_plans(const cudnnHandle_t handle, cudnn_frontend::OperationGraph& opGraph, cudnn_frontend::EngineConfigGenerator& generator, const Tensor& x, cudnn_frontend::executionPlans_t& valid_plans, Tensor& workspace) {
  auto initial_predicate_function = [&](cudnn_frontend::ExecutionPlan const& plan) -> bool {
    return false;
  };
  auto plans = generator.cudnnGetPlan(handle, opGraph, initial_predicate_function);
  size_t max_block_size = get_available_workspace();
  size_t max_workspace_size = 0u;
  std::for_each(plans.begin(), plans.end(), [&] (cudnn_frontend::ExecutionPlan& plan) {
    size_t curr_workspace_size = plan.getWorkspaceSize();
    if (curr_workspace_size <= max_block_size) {
      if (curr_workspace_size > max_workspace_size) {
        max_workspace_size = plan.getWorkspaceSize();
      }
      valid_plans.emplace_back(std::move(plan));
    }
  });
  TORCH_CHECK_WITH(CUDAOutOfMemoryError, max_workspace_size < 1_TiB, "Not enough memory for workspace!");
  bool remove_invalid = false;
  while (max_workspace_size) {
    try {
      workspace = at::empty({max_workspace_size}, x.options().dtype(kByte));
      break;
    } catch (c10::CUDAOutOfMemoryError &e) {
      max_workspace_size /= 2;
      cudaGetLastError(); // clear CUDA error
      remove_invalid = true;
    }
  }
  if (remove_invalid) {
    cudnn_frontend::executionPlans_t new_valid_plans;
    for (auto &plan : valid_plans) {
      if (plan.getWorkspaceSize() <= max_workspace_size) {
        new_valid_plans.emplace_back(std::move(plan));
      }
    }
    valid_plans = std::move(new_valid_plans);
  }
}

auto get_plans_from_find(const cudnnHandle_t handle, const cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, const CacheKey& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const bool deterministic, const bool allow_tf32) {
  auto opGraph = build_opgraph(handle, desc, x, y, w, key, padding, stride, dilation);
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w'};
  // We don't care about getting the best ordering of algos if we're roing to run all of them
  auto sources = get_generator_sources(desc, x, deterministic, allow_tf32, CUDNN_HEUR_MODE_INSTANT);
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  cudnn_frontend::executionPlans_t valid_plans;
  Tensor workspace;
  generate_and_filter_plans(handle, opGraph, generator, x, valid_plans, workspace);
  auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setDataPointers(3, data_ptrs)
      .setUids(3, uids)
      .setWorkspacePointer(workspace.has_storage() ? workspace.data_ptr() : NULL)
      .build();

  auto plans = cudnn_frontend::time_sorted_plan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(handle, std::move(valid_plans), variantPack);

  cudnn_frontend::executionPlans_t sorted_plans;
  for (auto& plan : plans) {
    sorted_plans.emplace_back(std::move(plan));
  }
  return sorted_plans;
}

auto get_plans_from_find_fused(const cudnnHandle_t handle,
                               const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b,
                               const float alpha, const CacheKeyFused& key,
                               const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation,
                               const bool deterministic, const bool allow_tf32) {
  auto opGraph = build_opgraph_fused(handle, x, y, w, z, b, alpha, key, padding, stride, dilation);
  void *data_ptrs[] = {x.data_ptr(), y.data_ptr(), w.data_ptr(), z.data_ptr(), b.data_ptr()};
  int64_t uids[] = {'x', 'y', 'w', 'z', 'b'};

  auto sources = get_generator_sources(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, x, deterministic, allow_tf32, CUDNN_HEUR_MODE_INSTANT);
  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  cudnn_frontend::executionPlans_t valid_plans;
  Tensor workspace;
  generate_and_filter_plans(handle, opGraph, generator, x, valid_plans, workspace);
  auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setDataPointers(5, data_ptrs)
      .setUids(5, uids)
      .setWorkspacePointer(workspace.has_storage() ? workspace.data_ptr() : NULL)
      .build();

  auto plans = cudnn_frontend::time_sorted_plan<cudnn_frontend::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(handle, std::move(valid_plans), variantPack);

  cudnn_frontend::executionPlans_t sorted_plans;
  for (auto& plan : plans) {
    sorted_plans.emplace_back(std::move(plan));
  }
  return sorted_plans;
}


// We only get configs from this stage to avoid building unnecessary plans that are never executed
auto get_configs_from_heuristics(const cudnnHandle_t handle, const cudnnBackendDescriptorType_t desc, const Tensor& x, const Tensor& y, const Tensor& w, const CacheKey& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const bool deterministic, const bool allow_tf32) {
  auto opGraph = build_opgraph(handle, desc, x, y, w, key, padding, stride, dilation);
  auto sources = get_generator_sources(desc, x, deterministic, allow_tf32, heuristic_mode);

  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  auto configs = generator.generate_engine_config(opGraph);
  return configs;
}

auto get_configs_from_heuristics_fused(const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b, const float alpha, const CacheKeyFused& key, const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const bool deterministic, const bool allow_tf32) {
  auto opGraph = build_opgraph_fused(handle, x, y, w, z, b, alpha, key, padding, stride, dilation);
  auto sources = get_generator_sources(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, x, deterministic, allow_tf32, heuristic_mode);

  cudnn_frontend::EngineConfigGenerator generator(sources.size(), sources.data());
  auto configs = generator.generate_engine_config(opGraph);
  return configs;
}

void try_plans(cudnn_frontend::executionPlans_t& plans, const CacheKey& key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w) {
  for (auto & plan : plans) {
    try {
      run_conv_plan(handle, x, y, w, plan);
      benchmark_cache.emplace(key, plan);
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch (CuDNNError &e) {}
      catch (c10::CUDAOutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
    }
  }
  TORCH_CHECK(false, "FIND was unable to find an engine to execute this computation");
}

void try_plans_fused(cudnn_frontend::executionPlans_t& plans, const CacheKeyFused& key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b) {
  for (auto & plan : plans) {
    try {
      run_conv_plan_fused(handle, x, y, w, z, b, plan);
      benchmark_cache_fused.emplace(key, plan);
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch (CuDNNError &e) {}
      catch (c10::CUDAOutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
    }
  }
  TORCH_CHECK(false, "FIND was unable to find an engine to execute this computation");
}

void try_configs(cudnn_frontend::EngineConfigList& configs, const CacheKey& key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w) {
  for (auto & config : configs) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(config)
                    .build();
      run_conv_plan(handle, x, y, w, plan);
      benchmark_cache.emplace(key, plan);
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch(CuDNNError &e) {}
      catch (c10::CUDAOutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
    }
  }
  TORCH_CHECK(false, "GET was unable to find an engine to execute this computation");
}

void try_configs_fused(cudnn_frontend::EngineConfigList& configs, const CacheKeyFused& key, const cudnnHandle_t handle, const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b) {
  for (auto & config : configs) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(handle)
                    .setEngineConfig(config)
                    .build();
      run_conv_plan_fused(handle, x, y, w, z, b, plan);
      benchmark_cache_fused.emplace(key, plan);
      return;
    } catch (cudnn_frontend::cudnnException &e) {} catch(CuDNNError &e) {}
      catch (c10::CUDAOutOfMemoryError &e) {
        cudaGetLastError(); // clear CUDA error
    }
  }
  TORCH_CHECK(false, "GET was unable to find an engine to execute this computation");
}

void run_single_conv(const cudnnBackendDescriptorType_t operation,
  const Tensor& x, const Tensor& y, const Tensor& w,
  const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
  const bool benchmark, const bool deterministic, const bool allow_tf32) {
  cudnnHandle_t handle = getCudnnHandle();

  CacheKey key;
  get_cachekey(key, operation, y, x, w, padding, stride, dilation, groups, deterministic, allow_tf32);
  // TODO: is this thread safe if cache is updated? is pointer stale?
  auto search = benchmark_cache.find(key);
  if (search) {
    try {
      run_conv_plan(handle, x, y, w, *search);
      return;
    } catch(c10::CUDAOutOfMemoryError &e) {
      cudaGetLastError(); // clear CUDA error
    }
  }

  if (!benchmark) {
    cudnn_frontend::EngineConfigList configs = get_configs_from_heuristics(handle, operation,
                                                                           x, y, w, key,
                                                                           padding, stride, dilation,
                                                                           deterministic, allow_tf32);
    try_configs(configs, key, handle, x, y, w);
  } else {
    cudnn_frontend::executionPlans_t plans = get_plans_from_find(handle, operation,
                                                                 x, y, w, key,
                                                                 padding, stride, dilation,
                                                                 deterministic, allow_tf32);
    // Replicate v7 behavior: clear cached blocks as benchmark incurs
    // significant memory consumptiont that is not needed after this step
    c10::cuda::CUDACachingAllocator::emptyCache();
    try_plans(plans, key, handle, x, y, w);
  }
}

void run_fused_conv(const Tensor& x, const Tensor& y, const Tensor& w, const Tensor& z, const Tensor& b,
  float alpha, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
  int64_t groups, const bool benchmark, const bool deterministic, const bool allow_tf32) {
  cudnnHandle_t handle = getCudnnHandle();

  CacheKeyFused key;
  get_cachekey_fused(key, y, x, w, z, b, alpha, padding, stride, dilation, groups, deterministic, allow_tf32);
  auto search = benchmark_cache_fused.find(key);
  if (search) {
    try {
      run_conv_plan_fused(handle, x, y, w, z, b, *search);
      return;
    } catch(c10::CUDAOutOfMemoryError &e) {
      cudaGetLastError(); // clear CUDA error
    }
  }

  if (!benchmark) {
    cudnn_frontend::EngineConfigList configs = get_configs_from_heuristics_fused(handle,
                                                                                 x, y, w, z, b, alpha, key,
                                                                                 padding, stride, dilation,
                                                                                 deterministic, allow_tf32);
    try_configs_fused(configs, key, handle, x, y, w, z, b);
  } else {
    cudnn_frontend::executionPlans_t plans = get_plans_from_find_fused(handle,
                                                                       x, y, w, z, b, alpha, key,
                                                                       padding, stride, dilation,
                                                                       deterministic, allow_tf32);
    try_plans_fused(plans, key, handle, x, y, w, z, b);
  }
}

void raw_cudnn_convolution_forward_out(
    const Tensor& output, const Tensor& input, const Tensor& weight,
    const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
    const bool benchmark, const bool deterministic, const bool allow_tf32)
{
  if (output.numel() == 0) { return; }
  if (use_v8()) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
      input, output, weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  } else {
    raw_cudnn_convolution_forward_out_v7(
      output, input, weight,
      padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

void raw_cudnn_convolution_backward_input_out(
    const at::Tensor& grad_input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
    const bool benchmark, const bool deterministic, const bool allow_tf32) {
  if (grad_input.numel() == 0) { return; }
  if (use_v8()) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
      grad_input, grad_output, weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  } else {
    raw_cudnn_convolution_backward_input_out_v7(
      grad_input,
      grad_output,
      weight,
      padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

void raw_cudnn_convolution_backward_weight_out(
    const Tensor& grad_weight, const Tensor& grad_output, const Tensor& input,
    const IntArrayRef padding, const IntArrayRef stride, const IntArrayRef dilation, const int64_t groups,
    const bool benchmark, const bool deterministic, const bool allow_tf32) {
  if (grad_weight.numel() == 0) { return; }
  if (use_v8()) {
    run_single_conv(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
      input, grad_output, grad_weight, padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  } else {
    raw_cudnn_convolution_backward_weight_out_v7(
      grad_weight, grad_output, input,
      padding, stride, dilation, groups,
      benchmark, deterministic, allow_tf32);
  }
}

void raw_cudnn_convolution_add_relu_out(
    const Tensor& output,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& z,
    float alpha,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    bool allow_tf32) {
  if (output.numel() == 0) { return; }
  if (use_v8()) {
    //auto bias_ = bias.repeat({output.size(0), 1, 1, 1});
    auto bias_ = bias.view({1, bias.numel(), 1, 1});
    run_fused_conv(input, output, weight, z, bias_,
      alpha, stride, padding, dilation,
      groups, benchmark, deterministic, allow_tf32);
  } else {
   raw_cudnn_convolution_add_relu_out_v7(output, input, weight, z,
                                         alpha, bias, stride, padding, dilation,
                                         groups, benchmark, deterministic, allow_tf32);
  }
}

}} // at::native

#endif  // HAS_CUDNN_V8
#endif  // AT_CUDNN_ENABLED
