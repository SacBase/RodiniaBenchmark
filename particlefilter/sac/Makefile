SAC2C := sac2c_d
SACFLAGS := -v1 -O3 -g -DFLOAT -profile m

SAC := seq_out seq_emr_out seq_c_out seq_c_emr_out
CUDA := cuda_out cuda_emr_out cuda_c_out cuda_c_emr_out
# cuda_alloc_out cuda_reg_out

.PHONY: all
all: $(SAC) $(CUDA)

%_out: particle_filter.sac
	$(SAC2C) $(SACFLAGS) -t $* -o $@ $^

%_emr_out: particle_filter.sac
	$(SAC2C) $(SACFLAGS) -doEMR -doELAAF -doELMP -t $* -o $@ $^

%_c_out: particle_filter_c_like.sac
	$(SAC2C) $(SACFLAGS) -t $* -o $@ $^

%_c_emr_out: particle_filter_c_like.sac
	$(SAC2C) $(SACFLAGS) -doEMR -doELAAF -doELMP -t $* -o $@ $^

.PHONY: clean
clean:
	$(RM) *_out *.cu *.c *.o *.sac2c
