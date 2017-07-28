SAC2C = sac2c_d
SAC2CFLAGS = -O3 -v2 -DOUTPUT

all: hotspot_sac_seq hotspot_sac_cuda

hotspot_sac_seq: hotspot.sac
	$(SAC2C) $(SAC2CFLAGS) -o $@ $<

hotspot_sac_mt: hotspot.sac
	$(SAC2C) -t mt_pth $(SAC2CFLAGS) -o $@ $<

hotspot_sac_cuda: hotspot.sac
	$(SAC2C) -t cuda $(SAC2CFLAGS) -o $@ $<

hotspot_sac_seq_out: hotspot.sac
	$(SAC2C) $(SAC2CFLAGS) -o $@ $<

hotspot_sac_mt_out: hotspot.sac
	$(SAC2C) -t mt_pth $(SAC2CFLAGS) -o $@ $<

hotspot_sac_cuda_out: hotspot.sac
	$(SAC2C) -t cuda $(SAC2CFLAGS) -o $@ $<

check: hotspot_sac_cuda_out hotspot_sac_seq_out
	./hotspot_sac_cuda_out >res_cuda
	./hotspot_sac_seq_out >res_seq
	diff res_cuda res_seq | head

run: hotspot_sac_seq hotspot_sac_cuda
	@echo "\n \n ***** SAC seq *****"
	-time ./hotspot_sac_seq
	@echo "\n \n ***** SAC cuda *****"
	-time ./hotspot_sac_cuda
