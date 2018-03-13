SAC2C := sac2c_d
SIZE ?= 1024
ITER ?= 5000
SACFLAGS := -v1 -O3 -noRCO -DFLOAT -DSIZE=$(SIZE) -DITER=$(ITER) -profile m
# `-noRCO` is needed for the CUDA transfer mechs to work, this
# option should be left on for all runs - for consistency.

.PHONY: all
all: seq_out emr_seq_out cuda_out emr_cuda_out
#    cuda_alloc_out cuda_man_out cuda_reg_out

%_out: hotspot.sac
	$(SAC2C) $(SACFLAGS) -t $* -o $@ $^
	@echo -e "#!/usr/bin/env sh\n./$@ < ../input/sac_temp_power_$(SIZE) > $@.log" > run_$@
	@chmod +x run_$@

emr_%_out: hotspot.sac
	$(SAC2C) $(SACFLAGS) -doEMR -doELAAF -doELMP -t $* -o $@ $^
	@echo -e "#!/usr/bin/env sh\n./$@ < ../input/sac_temp_power_$(SIZE) > $@.log" > run_$@
	@chmod +x run_$@

.PHONY: clean
clean:
	$(RM) *_out *_out.cu *_out.c run_*
