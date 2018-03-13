SAC2C := sac2c_d
SIZE ?= 512
SACFLAGS := -v1 -O3 -DFLOAT -profile m -noRCO -doPRA -doEXPAR -DSIZE=$(SIZE)
# `-noRCO` is needed for the CUDA transfer mechs to work, this
# option should be left on for all runs - for consistency.

.PHONY: all
all: seq_out cuda_out emr_seq_out emr_cuda_out
#cuda_alloc_out cuda_man_out cuda_reg_out

%_out: lud.sac
	$(SAC2C) $(SACFLAGS) -t $* -o $@ $^

emr_%_out: lud.sac
	$(SAC2C) $(SACFLAGS) -doEMR -doELAAF -doELMP -t $* -o $@ $^

.PHONY: clean
clean:
	$(RM) *_out *_out.cu *_out.c
