SAC2C := sac2c_d
SACFLAGS := -v1 -O3 -noRCO -DFLOAT
# `-DLOOP` has been omitted as in the cuda_man
# case we have undefined references popping up.

SAC := seq_out
CUDA := cuda_out cuda_alloc_out cuda_reg_out

.PHONY: all
all: $(SAC) cuda cuda_man

.PHONY: cuda
cuda: SACFLAGS += -DCUDA -doEXPAR -doPFD
cuda: $(CUDA)

.PHONY: cuda_man
cuda_man: SACFLAGS += -DCUDA -doEXPAR
cuda_man: cuda_man_out
# would normally try to use `-doPFD` as well, but this seems
# to crash the compiler when using cuda_man

%_out: backprop.sac
	$(SAC2C) $(SACFLAGS) -t $* -o $@ $^

%_nb_out: backprop_no_boundary.sac
	$(SAC2C) -t $* $(SACFLAGS) -o $@ $^

.PHONY: clean
clean:
	$(RM) *_out *.cu *.c *.o *.sac2c
