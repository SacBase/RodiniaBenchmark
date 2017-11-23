# C compiler
CC = g++
CC_FLAGS = -g -fopenmp -O3 

ifeq ($(out),1)
	CC_FLAGS += -DOUTPUT
endif

needle: 
	$(CC) $(CC_FLAGS) needle.cpp -o needle 

clean:
	rm -f needle 
