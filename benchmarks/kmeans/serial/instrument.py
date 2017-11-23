#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;

sizes = [131072, 262144, 524288, 1048576]; 
#sizes = [1024,2048]; 

# Total number of runs for each compilation
runs = 3;

# Variables for various file names
out_exe = "a.out"

all_runtimes = "all_runtimes"

# Execution starts from here
try:
    prog         = sys.argv[1];
except: 
    print "Usage:", sys.argv[0]," [program]";
    sys.exit(1);

def compute_average( src, dst):
    infile = open( src, "r");
    outfile = open( dst, "a");

    print "runs is " + `runs`;

    i = 0;
    while i < len(sizes):
        runtimes = 0.0;
        j = 0;
	while j < runs:
	    line = infile.readline(); 
            runtimes = runtimes + float(line);
	    j = j + 1;

	runtimes = runtimes/runs;
	outfile.write( `sizes[i]` + " " + `runtimes` + "\n"); 

	i = i + 1;

    infile.close();
    outfile.close();

#===========================================================================================================================================

standard_flags = [""];
runtime_csv = ["./runtimes/c_seq.csv"];

# Compile and meaures for standard runs, i.e. sac sequential,
# sac multi-threaded, cuda baseline and cuda with memopt.
r = 0;
while r < len(standard_flags):
    i = 0;
    while i < len(sizes):
	cmd = "gcc -O3 " + standard_flags[r]  + " -DSIZE=" + `sizes[i]` + " " +  prog + " -o " + out_exe;
	print cmd;
	os.system(cmd); 

	# Perform several runs and store the runtimes in tmp file 
	j = 0;
	while j < runs:
	    os.system( "./" + out_exe + " >> " + all_runtimes);
	    j = j + 1;

	i = i + 1;

    compute_average( all_runtimes, runtime_csv[r]);
    os.system("rm " + all_runtimes);

    r = r + 1;

