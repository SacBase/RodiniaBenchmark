include ../../common/make.config

#CUDA compiler
NVCC := nvcc
SIZE ?= 1024
ITER ?= 5000
NVCC_FLAGS += -DUSEFILE -DSIZE=$(SIZE) -DITER=$(ITER) -arch=sm_35 -I$(CUDA_DIR)/include

# 'make dbg=1' enables NVCC debugging
ifeq ($(dbg),1)
	NVCC_FLAGS += -g -O0
else
	NVCC_FLAGS += -O3
endif

# 'make emu=1' compiles the CUDA kernels for emulation
ifeq ($(emu),1)
	NVCC_FLAGS += -deviceemu
endif

# 'make out=1' output detailed results
ifeq ($(out),1)
	NVCC_FLAGS += -DOUTPUT
endif

# without utilizing shared memory
noshr:  
	$(NVCC) $(NVCC_FLAGS) hotspot_cuda_noshr_float.cu -o noshr_hotspot -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

# with utilizing shared memory
shr:  
	$(NVCC) $(NVCC_FLAGS) hotspot_cuda_shr_float.cu -o shr_hotspot -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

# delete all object files
clean:
	rm *.o noshr_hotspot shr_hotspot



