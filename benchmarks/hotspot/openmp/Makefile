# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O3

# 'make out=1' output detailed results
ifeq ($(out),1)
	CC_FLAGS += -DOUTPUT
endif

openmp: 
	$(CC) $(CC_FLAGS) -DOPEN hotspot_openmp_float.cpp -o openmp_hotspot  
seq: 
	$(CC) $(CC_FLAGS) hotspot_openmp_float.cpp -o seq_hotspot  

clean:
	rm -f *.o *_hotspot
