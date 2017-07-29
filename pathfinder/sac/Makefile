SAC2C := sac2c_d
SIZE ?= 512000
ITER ?= 100
SACFLAGS := -v1 -O3 -noRCO -DSIZE=$(SIZE) -DITER=$(ITER)
# `-noRCO` is needed for the CUDA transfer mechs to work, this
# option should be left on for all runs - for consistency.


EXEC := seq_out cuda_out cuda_alloc_out cuda_man_out \
	cuda_reg_out

.PHONY: all
all: $(EXEC) condi

.PHONY: condi
condi: $(patsubst %_out,%_condi_out,$(EXEC))

%_out: pathfinder.sac
	$(SAC2C) $(SACFLAGS) -t $* -o $@ $^

%_condi_out: pathfinder.sac
	$(SAC2C) $(SACFLAGS) -DCONDI -t $* -o $@ $^

.PHONY: clean
clean:
	$(RM) *_out *_out.cu *_out.c
