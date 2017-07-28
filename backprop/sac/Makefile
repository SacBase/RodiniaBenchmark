SAC2C := sac2c_d
SACFLAGS := -v1 -O3

.PHONY: all
all: cuda seq

.PHONY: cuda
cuda: cuda_out cuda_nb_out

.PHONY: seq
seq: sac_out sac_nb_out

cuda_out: backprop.sac
	$(SAC2C) -tcuda $(SACFLAGS) -DCUDA -DLOOP -DFLOAT -doexpar -dopfd -o $@ $^

sac_out: backprop.sac
	$(SAC2C) -tseq $(SACFLAGS) -DLOOP -DFLOAT -o $@ $^

cuda_nb_out: backprop_no_boundary.sac
	$(SAC2C) -tcuda $(SACFLAGS) -DCUDA -DLOOP -DFLOAT -doexpar -dopfd -o $@ $^

sac_nb_out: backprop_no_boundary.sac
	$(SAC2C) -tseq $(SACFLAGS) -DLOOP -DFLOAT -o $@ $^

.PHONY: clean
clean:
	$(RM) *_out *.cu *.c *.o *.sac2c
