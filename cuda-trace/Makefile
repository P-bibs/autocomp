ROOT := /home/paulbib/Development/autocomp/cuda-trace
CUDA_HOME ?= /home/paulbib/Builds/cuda-toolkit/toolkit
NVCC ?= nvcc
CXX ?= g++
PYTHON ?= python3

SRC_DIR := $(ROOT)/src
EXAMPLES_DIR := $(ROOT)/examples
SCRIPTS_DIR := $(ROOT)/scripts
BUILD_DIR := $(ROOT)/build
LOG_DIR := $(ROOT)/logs

CUDA_BUILD_DIR := $(BUILD_DIR)/cuda
PYTORCH_BUILD_DIR := $(BUILD_DIR)/pytorch_ext
PYTORCH_JIT_DIR := $(PYTORCH_BUILD_DIR)/jit
PYTORCH_VENDOR_DIR := $(PYTORCH_BUILD_DIR)/vendor
PYTORCH_LOG_DIR := $(LOG_DIR)/pytorch_ext
FORWARD_TRACE_LOG_DIR := $(LOG_DIR)/forward_trace
FORWARD_TRACE_CLI := $(SRC_DIR)/pytorch/forward_pipeline_trace.py

CUDA_INCLUDE := $(CUDA_HOME)/include
PTX_GENCODE ?= arch=compute_75,code=[sm_75,compute_75]

CPPFLAGS += -I$(CUDA_INCLUDE)
CXXFLAGS += -std=c++17 -fPIC
LDFLAGS += -ldl -pthread

CUDA_TRACER := $(CUDA_BUILD_DIR)/libcuda_trace_preload.so
PYTORCH_TRACER := $(PYTORCH_BUILD_DIR)/libcuda_trace_module_preload.so

.PHONY: all cuda-all cuda-tracer pytorch-all pytorch-tracer pytorch-vendor clean \
	trace-sample trace-mangled \
	trace-pytorch-single trace-pytorch-same-module trace-pytorch-multi-modules \
	trace-forward-3 trace-forward-single trace-forward-functional-linear trace-forward-same-module trace-forward-multi-modules

all: cuda-all pytorch-all

cuda-all: $(CUDA_TRACER) \
	$(CUDA_BUILD_DIR)/sample \
	$(CUDA_BUILD_DIR)/sample_static \
	$(CUDA_BUILD_DIR)/sample_mangled \
	$(CUDA_BUILD_DIR)/sample_with_ptx \
	$(CUDA_BUILD_DIR)/sample_no_ptx

cuda-tracer: $(CUDA_TRACER)

$(CUDA_TRACER): $(SRC_DIR)/cuda_trace_preload.cpp
	mkdir -p $(CUDA_BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -shared -o $@ $< $(LDFLAGS)

$(CUDA_BUILD_DIR)/sample: $(EXAMPLES_DIR)/cuda/sample.cu
	mkdir -p $(CUDA_BUILD_DIR)
	$(NVCC) -cudart shared -o $@ $<

$(CUDA_BUILD_DIR)/sample_static: $(EXAMPLES_DIR)/cuda/sample.cu
	mkdir -p $(CUDA_BUILD_DIR)
	$(NVCC) -cudart static -o $@ $<

$(CUDA_BUILD_DIR)/sample_mangled: $(EXAMPLES_DIR)/cuda/sample_mangled.cu
	mkdir -p $(CUDA_BUILD_DIR)
	$(NVCC) -cudart shared -o $@ $<

$(CUDA_BUILD_DIR)/sample_with_ptx: $(EXAMPLES_DIR)/cuda/sample_mangled.cu
	mkdir -p $(CUDA_BUILD_DIR)
	$(NVCC) -cudart shared -gencode $(PTX_GENCODE) -o $@ $<

$(CUDA_BUILD_DIR)/sample_no_ptx: $(EXAMPLES_DIR)/cuda/sample_mangled.cu
	mkdir -p $(CUDA_BUILD_DIR)
	$(NVCC) -cudart shared -gencode arch=compute_75,code=sm_75 -o $@ $<

trace-sample: $(CUDA_TRACER) $(CUDA_BUILD_DIR)/sample
	$(SCRIPTS_DIR)/trace_launch.sh $(CUDA_BUILD_DIR)/sample

trace-mangled: $(CUDA_TRACER) $(CUDA_BUILD_DIR)/sample_with_ptx
	$(SCRIPTS_DIR)/trace_launch.sh $(CUDA_BUILD_DIR)/sample_with_ptx

pytorch-all: $(PYTORCH_TRACER)

pytorch-tracer: $(PYTORCH_TRACER)

pytorch-vendor:
	@if [ ! -x $(PYTORCH_VENDOR_DIR)/bin/ninja ]; then \
		mkdir -p $(PYTORCH_BUILD_DIR); \
		$(PYTHON) -m pip install --target $(PYTORCH_VENDOR_DIR) ninja; \
	fi

$(PYTORCH_TRACER): $(SRC_DIR)/pytorch/cuda_trace_module_preload.cpp
	mkdir -p $(PYTORCH_BUILD_DIR) $(PYTORCH_LOG_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -shared -o $@ $< $(LDFLAGS)

trace-pytorch-single: $(PYTORCH_TRACER) pytorch-vendor
	mkdir -p $(PYTORCH_JIT_DIR) $(PYTORCH_LOG_DIR)
	: > $(PYTORCH_LOG_DIR)/single.jsonl
	env PATH=$(PYTORCH_VENDOR_DIR)/bin:$$PATH \
	    PYTHONPATH=$(PYTORCH_VENDOR_DIR) \
	    TORCH_EXTENSIONS_DIR=$(PYTORCH_JIT_DIR) \
	    TORCH_CUDA_ARCH_LIST='7.5+PTX' \
	    CUDA_TRACE_FILTER_PATH_SUBSTR='$(PYTORCH_JIT_DIR)' \
	    CUDA_TRACE_JSON_PATH='$(PYTORCH_LOG_DIR)/single.jsonl' \
	    LD_PRELOAD=$(PYTORCH_TRACER) \
	    MAX_JOBS=1 \
	    $(PYTHON) $(EXAMPLES_DIR)/pytorch/test.py \
	    > $(PYTORCH_LOG_DIR)/single.stdout.txt \
	    2> $(PYTORCH_LOG_DIR)/single.stderr.txt

trace-pytorch-same-module: $(PYTORCH_TRACER) pytorch-vendor
	mkdir -p $(PYTORCH_JIT_DIR) $(PYTORCH_LOG_DIR)
	: > $(PYTORCH_LOG_DIR)/same_module.jsonl
	env PATH=$(PYTORCH_VENDOR_DIR)/bin:$$PATH \
	    PYTHONPATH=$(PYTORCH_VENDOR_DIR) \
	    TORCH_EXTENSIONS_DIR=$(PYTORCH_JIT_DIR) \
	    TORCH_CUDA_ARCH_LIST='7.5+PTX' \
	    CUDA_TRACE_FILTER_PATH_SUBSTR='$(PYTORCH_JIT_DIR)' \
	    CUDA_TRACE_JSON_PATH='$(PYTORCH_LOG_DIR)/same_module.jsonl' \
	    LD_PRELOAD=$(PYTORCH_TRACER) \
	    MAX_JOBS=1 \
	    $(PYTHON) $(EXAMPLES_DIR)/pytorch/test_multi_same_module.py \
	    > $(PYTORCH_LOG_DIR)/same_module.stdout.txt \
	    2> $(PYTORCH_LOG_DIR)/same_module.stderr.txt

trace-pytorch-multi-modules: $(PYTORCH_TRACER) pytorch-vendor
	mkdir -p $(PYTORCH_JIT_DIR) $(PYTORCH_LOG_DIR)
	: > $(PYTORCH_LOG_DIR)/multi_modules.jsonl
	env PATH=$(PYTORCH_VENDOR_DIR)/bin:$$PATH \
	    PYTHONPATH=$(PYTORCH_VENDOR_DIR) \
	    TORCH_EXTENSIONS_DIR=$(PYTORCH_JIT_DIR) \
	    TORCH_CUDA_ARCH_LIST='7.5+PTX' \
	    CUDA_TRACE_FILTER_PATH_SUBSTR='$(PYTORCH_JIT_DIR)' \
	    CUDA_TRACE_JSON_PATH='$(PYTORCH_LOG_DIR)/multi_modules.jsonl' \
	    LD_PRELOAD=$(PYTORCH_TRACER) \
	    MAX_JOBS=1 \
	    $(PYTHON) $(EXAMPLES_DIR)/pytorch/test_multi_modules.py \
	    > $(PYTORCH_LOG_DIR)/multi_modules.stdout.txt \
	    2> $(PYTORCH_LOG_DIR)/multi_modules.stderr.txt

clean:
	rm -rf $(BUILD_DIR) $(LOG_DIR)

trace-forward-3: $(PYTORCH_TRACER) pytorch-vendor
	mkdir -p $(FORWARD_TRACE_LOG_DIR)
	env PATH=$(PYTORCH_VENDOR_DIR)/bin:$$PATH \
	    PYTHONPATH=$(PYTORCH_VENDOR_DIR) \
	    FORWARD_TRACE_TRACER_PATH=$(PYTORCH_TRACER) \
	    TORCH_EXTENSIONS_DIR=$(PYTORCH_JIT_DIR) \
	    TORCH_CUDA_ARCH_LIST='7.5+PTX' \
	    $(PYTHON) $(FORWARD_TRACE_CLI) \
	    $(EXAMPLES_DIR)/pytorch/3.py \
	    $(FORWARD_TRACE_LOG_DIR)/3.json
trace-forward-single: $(PYTORCH_TRACER) pytorch-vendor
	mkdir -p $(FORWARD_TRACE_LOG_DIR)
	env PATH=$(PYTORCH_VENDOR_DIR)/bin:$$PATH \
	    PYTHONPATH=$(PYTORCH_VENDOR_DIR) \
	    FORWARD_TRACE_TRACER_PATH=$(PYTORCH_TRACER) \
	    TORCH_EXTENSIONS_DIR=$(PYTORCH_JIT_DIR) \
	    TORCH_CUDA_ARCH_LIST='7.5+PTX' \
	    $(PYTHON) $(FORWARD_TRACE_CLI) \
	    $(EXAMPLES_DIR)/pytorch/pipeline_single.py \
	    $(FORWARD_TRACE_LOG_DIR)/single.json

trace-forward-functional-linear: $(PYTORCH_TRACER) pytorch-vendor
	mkdir -p $(FORWARD_TRACE_LOG_DIR)
	env PATH=$(PYTORCH_VENDOR_DIR)/bin:$$PATH \
	    PYTHONPATH=$(PYTORCH_VENDOR_DIR) \
	    FORWARD_TRACE_TRACER_PATH=$(PYTORCH_TRACER) \
	    TORCH_EXTENSIONS_DIR=$(PYTORCH_JIT_DIR) \
	    TORCH_CUDA_ARCH_LIST='7.5+PTX' \
	    $(PYTHON) $(FORWARD_TRACE_CLI) \
	    $(EXAMPLES_DIR)/pytorch/pipeline_functional_linear.py \
	    $(FORWARD_TRACE_LOG_DIR)/functional_linear.json

trace-forward-same-module: $(PYTORCH_TRACER) pytorch-vendor
	mkdir -p $(FORWARD_TRACE_LOG_DIR)
	env PATH=$(PYTORCH_VENDOR_DIR)/bin:$$PATH \
	    PYTHONPATH=$(PYTORCH_VENDOR_DIR) \
	    FORWARD_TRACE_TRACER_PATH=$(PYTORCH_TRACER) \
	    TORCH_EXTENSIONS_DIR=$(PYTORCH_JIT_DIR) \
	    TORCH_CUDA_ARCH_LIST='7.5+PTX' \
	    $(PYTHON) $(FORWARD_TRACE_CLI) \
	    $(EXAMPLES_DIR)/pytorch/pipeline_same_module.py \
	    $(FORWARD_TRACE_LOG_DIR)/same_module.json

trace-forward-multi-modules: $(PYTORCH_TRACER) pytorch-vendor
	mkdir -p $(FORWARD_TRACE_LOG_DIR)
	env PATH=$(PYTORCH_VENDOR_DIR)/bin:$$PATH \
	    PYTHONPATH=$(PYTORCH_VENDOR_DIR) \
	    FORWARD_TRACE_TRACER_PATH=$(PYTORCH_TRACER) \
	    TORCH_EXTENSIONS_DIR=$(PYTORCH_JIT_DIR) \
	    TORCH_CUDA_ARCH_LIST='7.5+PTX' \
	    $(PYTHON) $(FORWARD_TRACE_CLI) \
	    $(EXAMPLES_DIR)/pytorch/pipeline_multi_modules.py \
	    $(FORWARD_TRACE_LOG_DIR)/multi_modules.json
