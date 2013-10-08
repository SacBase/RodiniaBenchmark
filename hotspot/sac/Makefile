OUTPUT = -DOUTPUT

SAC2CFLAGS = -O3

all: hotspot_sac_seq hotspot_sac_cuda

hotspot_sac_seq: hotspot.sac
	sac2c $(SAC2CFLAGS) hotspot.sac -o hotspot_sac_seq

hotspot_sac_mt: hotspot.sac
	sac2c -mt $(SAC2CFLAGS) hotspot.sac -o hotspot_sac_mt

hotspot_sac_cuda: hotspot.sac
	sac2c -t cuda $(SAC2CFLAGS) hotspot.sac -o hotspot_sac_cuda

hotspot_sac_seq_out: hotspot.sac
	sac2c $(OUTPUT) $(SAC2CFLAGS) hotspot.sac -o hotspot_sac_seq_out

hotspot_sac_mt_out: hotspot.sac
	sac2c -mt $(OUTPUT) $(SAC2CFLAGS) hotspot.sac -o hotspot_sac_mt_out

hotspot_sac_cuda_out: hotspot.sac
	sac2c $(OUTPUT) -t cuda $(SAC2CFLAGS) hotspot.sac -o hotspot_sac_cuda_out

check: hotspot_sac_cuda_out hotspot_sac_seq_out
	./hotspot_sac_cuda_out >res_cuda
	./hotspot_sac_seq_out >res_seq
	diff res_cuda res_seq | head

run: hotspot_sac_seq hotspot_sac_cuda
	@echo "\n \n ***** SAC seq *****"
	-time ./hotspot_sac_seq
	@echo "\n \n ***** SAC cuda *****"
	-time ./hotspot_sac_cuda
