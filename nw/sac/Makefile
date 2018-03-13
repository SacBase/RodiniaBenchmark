SAC2C := sac2c_d
SIZE ?= 2048
SACFLAGS := -v1 -O3 -profile m -DSIZE=$(SIZE)

SAC := seq_out emr_seq_out
CUDA := cuda_out emr_cuda_out
# cuda_alloc_out cuda_reg_out

.PHONY: all
all: $(SAC) $(CUDA)

%_out: needle.sac
	$(SAC2C) $(SACFLAGS) -t $* -o $@ $^

emr_%_out: needle.sac
	$(SAC2C) $(SACFLAGS) -doEMR -doELAAF -doELMP -t $* -o $@ $^

.PHONY: clean
clean:
	$(RM) *_out *.cu *.c *.o *.sac2c
