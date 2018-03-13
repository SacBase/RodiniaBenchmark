SAC2C := sac2c_d
SACFLAGS := -v1 -O3 -noRCO -DFLOAT -profile m

SAC := seq_out emr_seq_out
CUDA := cuda_out emr_cuda_out
# cuda_alloc_out cuda_reg_out

.PHONY: all
all: seq cuda

.PHONY: seq
seq: SACFLAGS += -DSEQ
seq: $(SAC)

.PHONY: cuda
cuda: SACFLAGS += -DPFD -doPFD
cuda: $(CUDA)

.PHONY: cuda_man
cuda_man: SACFLAGS += -DPFD -doPFD -doEXPAR
cuda_man: cuda_man_out

%_out: kmeans.sac
	$(SAC2C) $(SACFLAGS) -t $* -o $@ $^

emr_%_out: kmeans.sac
	$(SAC2C) $(SACFLAGS) -doEMR -doELAAF -doELMP -t $* -o $@ $^

.PHONY: clean
clean:
	$(RM) *_out *.cu *.c *.o *.sac2c
