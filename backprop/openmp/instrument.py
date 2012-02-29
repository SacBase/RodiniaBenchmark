#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;

sizes = [65536, 131072, 262144, 524288]; 
#sizes = [65536]; 

max_measure_regions = 10;
actual_measure_regions = 0;

# Total number of runs for each compilation
runs = 2;

# Variables for various file names
out_exe = "a.out"
out_sac2c = "./a.out.sac2c";

tmp_file = "tmp";
all_runtimes = "all_runtimes"

opt_flags = [];

flops_file = "./flops.csv"

# Execution starts from here
try:
    prog         = sys.argv[1];
    flops        = sys.argv[2];
    unit         = sys.argv[3];
    iterations   = sys.argv[4];
    threads      = sys.argv[5];
    
    for arg in sys.argv[6:]:
       opt_flags.append( arg); 

except: 
    print "Usage:", sys.argv[0]," [program] [compute flops?] [unit of time (0 - seconds, 1 - milliseconds, 2 - microseconds)] [# of iterations] [# of threads] [OPT flag]*";
    sys.exit(1);

# Total number of iterations
iterations = int(iterations);

# Total number of iterations
threads = int(threads);

# Do we compute flops?
flops = int(flops);

# Unit of time to measure in
unit = int(unit);
time_factors = [1.0, 1.0];
if unit == 0:
    time_factors[0] = 1.0;
    time_factors[1] = 1000000.0;
elif unit == 1:
    time_factors[0] = 1000.0;
    time_factors[1] = 1000.0;
elif unit == 2:
    time_factors[0] = 1000000.0;
    time_factors[1] = 1.0;
else:
    print "Unknow time unit!"
    exit( 1);

def compute_average( src, dst):
    infile = open( src, "r");
    outfile = open( dst, "a");

    print "runs is " + `runs`;

    i = 0;
    while i < len(sizes):
        runtimes = [];
        j = 0;
	while j < runs:
	    k = 0;
	    while k < actual_measure_regions:
		line = infile.readline(); 
		if j == 0:
		    runtimes.append(float(line)); 
		else:
		    runtimes[k] = runtimes[k] + float(line);
		k = k + 1;
	    j = j + 1;

	l = 0;
	while l < actual_measure_regions:
	    runtimes[l] = runtimes[l]/runs;
	    outfile.write( `sizes[i]` + " " + `runtimes[l]` + " region" + `l` + "\n"); 
	    l = l + 1;

	i = i + 1;

    infile.close();
    outfile.close();

#===========================================================================================================================================

standard_flags = ["-lm"];
out_srcs = ["a.out.c"];
runtime_csv = ["./runtimes/c_seq.csv"];


# Compile and meaures for standard runs, i.e. sac sequential,
# sac multi-threaded, cuda baseline and cuda with memopt.
r = 0;
while r < len(standard_flags):
    i = 0;
    while i < len(sizes):
	cmd = "gcc " + standard_flags[r]  + " -O3 -DSIZE=" + `sizes[i]` + " -DITER=" + `iterations` + " " +  prog + " -o " + out_exe;
	print cmd;
	os.system(cmd); 

        actual_measure_regions = 1;

	# Perform several runs and store the runtimes in tmp file 
	j = 0;
	while j < runs:
	    os.system( "./" + out_exe + " >> " + all_runtimes);
	    j = j + 1;

	i = i + 1;

    compute_average( all_runtimes, runtime_csv[r]);
    os.system("rm " + all_runtimes);

    r = r + 1;

