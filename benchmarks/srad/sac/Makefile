SAC2C := sac2c_d
SIZE  ?= 2048
ITER  ?= 1000
SACFLAGS := -v1 -O3 -noRCO -DSIZE=$(SIZE) -DITER=$(ITER)

ifeq ($(out),1)
	SACFLAGS += -DOUTPUT
endif

SAC := seq_out
CUDA := cuda_out cuda_alloc_out cuda_reg_out cuda_man_out

.PHONY: all
all: $(SAC) cuda

.PHONY: cuda
cuda: SACFLAGS += -doEXPAR -doPFD
cuda: $(CUDA)

%_out: srad.sac
	$(SAC2C) $(SACFLAGS) -t $* -o $@ $^

.PHONY: clean
clean:
	$(RM) *_out *.cu *.c *.o *.sac2c
