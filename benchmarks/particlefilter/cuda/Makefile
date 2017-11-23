#makefile

#include ../../common/make.config

CC := nvcc

INCLUDE := $(CUTIL)

#all: naive float double
all: double

#naive: ex_particle_CUDA_naive_seq.cu
#	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math -arch sm_20 ex_particle_CUDA_naive_seq.cu -o particlefilter_naive
	
#float: ex_particle_CUDA_float_seq.cu
#	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math -arch sm_20 ex_particle_CUDA_float_seq.cu -o particlefilter_float

double: ex_particle_CUDA_double_seq.cu
	$(CC) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -g -lm -O3 -use_fast_math -arch sm_20 ex_particle_CUDA_double_seq.cu -o particlefilter_double
clean:
	rm particlefilter_naive particlefilter_float particlefilter_double
