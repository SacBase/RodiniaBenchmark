# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O3

# 'make out=1' output detailed results
ifeq ($(out),1)
	CC_FLAGS += -DOUTPUT
endif

srad: 
	$(CC) $(CC_FLAGS) srad.cpp -o srad 

clean:
	rm -f srad
