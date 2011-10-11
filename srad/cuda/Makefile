
#include ../../../common/make.config

CUDA_LIB_DIR = /usr/local/cuda/lib

#CUDA compiler
NVCC := nvcc
NVCC_FLAGS = -arch=sm_20 -I$(CUTIL)

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
	$(NVCC) $(NVCC_FLAGS) -DNOSHR srad.cu -o noshr_srad -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

# with utilizing shared memory
shr:  
	$(NVCC) $(NVCC_FLAGS) srad.cu -o shr_srad -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

# delete all object files
clean:
	rm *.o noshr_srad shr_srad
