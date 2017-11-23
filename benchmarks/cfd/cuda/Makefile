#include ../../common/make.config

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

ifeq ($(reg),1)
	NVCC_FLAGS += --maxrregcount 20
endif

euler3d: euler3d_gpu_double.cu   
	$(NVCC) $(NVCC_FLAGS) euler3d_gpu_double.cu -o euler3d -L$(CUDA_LIB_DIR) -lcuda -lcudart -lm

#euler3d: euler3d.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 euler3d.cu -o euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)

#euler3d_double: euler3d_double.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 euler3d_double.cu -o euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)


#pre_euler3d: pre_euler3d.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 pre_euler3d.cu -o pre_euler3d -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)

#pre_euler3d_double: pre_euler3d_double.cu
#	nvcc -Xptxas -v -O3 --gpu-architecture=compute_13 --gpu-code=compute_13 pre_euler3d_double.cu -o pre_euler3d_double -I$(CUDA_SDK_PATH)/common/inc  -L$(CUDA_SDK_PATH)/lib $(CUTIL_LIB)

clean:
	rm *.o euler3d
