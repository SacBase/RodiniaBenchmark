#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;

#sizes = [1024, 2048, 3072, 4096]; 
sizes = [128, 256]; 
max_measure_regions = 10;
actual_measure_regions = 0;

# Total number of runs for each compilation
runs = 2;

# Variables for various file names
sac_out_exe = "sac_out";
sac_out_src = "sac_out.c";
sac_out_sac2c = "./sac_out.sac2c";

cuda_out_exe = "cuda_out";
cuda_out_src = "cuda_out.cu";
cuda_out_sac2c = "./cuda_out.sac2c";

tmp_file = "tmp";
all_runtimes = "all_runtimes"


# File to store final computed runtime results
sac_seq       = "./runtimes/sac_seq.csv"
sac_mt        = "./runtimes/sac_mt.csv"
cuda_baseline = "./runtimes/cuda_baseline.csv"
cuda_memopt   = "./runtimes/cuda_memopt.csv"
cuda_expar    = "./runtimes/cuda_expar.csv"
cuda_lao      = "./runtimes/cuda_lao.csv"
cuda_pfd      = "./runtimes/cuda_pfd.csv"


flops_file = "./flops.csv"

# Execution starts from here
try:
    prog         = sys.argv[1];
    flops        = sys.argv[2];
    unit         = sys.argv[3];
    expar        = sys.argv[4];
    lao          = sys.argv[5];
    pfd          = sys.argv[6];
    iterations   = sys.argv[7];
    threads      = sys.argv[8];
except: 
    print "Usage:", sys.argv[0]," [program] [compute flops?] [unit of time (0 - seconds, 1 - milliseconds, 2 - microseconds)] \n [do expar?] [do lao?] [do pfd?] [# of iterations] [# of threads]";

    sys.exit(1);

expar  = int(expar); 
lao    = int(lao); 
pfd    = int(pfd); 

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

def add_papi_library( source=""):
    infile = open( source, "r");  
    outfile = open( tmp_file, "w");
    for line in infile:
        if line.find("gcc") != -1:
            outfile.write(line.strip() + " -lpapi\n");
        else:
            outfile.write(line);

    infile.close();
    outfile.close();
 
    os.system("mv " + tmp_file + " " + source);  
    os.system("chmod 777 " + source);

def instrument_flops( source="", size=0):
    measure_region = 0;
    infile = open( source, "r");  
    outfile = open( tmp_file, "w");
    for line in infile:
        if line.find("sac.h") != -1:
            outfile.write(line);
            outfile.write("#include <stdlib.h>\n");  
            outfile.write("#include <stdio.h>\n");  
            outfile.write("#include <papi.h>\n");  
            
        elif line.find("SAC_ND_FUNAP2(SACf_C99Benchmarking__start__SACt_C99Benchmarking") != -1:
            if measure_region == 0:
		outfile.write("int retval, event_set, event_code;\n");
		outfile.write("long long values[1];\n");

		outfile.write("retval = PAPI_library_init( PAPI_VER_CURRENT);\n");
		outfile.write("if( retval != PAPI_VER_CURRENT && retval > 0) {\n");
		outfile.write("  fprintf(stderr, \"PAPI library version mismatch!\\n\");\n");
		outfile.write("  exit(1);\n");
		outfile.write("}\n");

		outfile.write("event_set = PAPI_NULL;\n");

		outfile.write("retval = PAPI_create_eventset( &event_set);\n");
		outfile.write("if( retval != PAPI_OK) {\n");
		outfile.write("  fprintf(stderr, \"Error occurred in PAPI_create_eventset!\\n\");\n");
		outfile.write("  exit(1);\n");
		outfile.write("}\n");

		#outfile.write("event_code = PAPI_FMA_INS;\n");
		#outfile.write("retval = PAPI_add_event( event_set, event_code);\n");
		#outfile.write("if( retval != PAPI_OK) {\n");
		#outfile.write("  fprintf(stderr, \"Error occurred in PAPI_add_event - PAPI_FMA_INS!\\n\");\n");
		#outfile.write("  exit(1);\n");
		#outfile.write("}\n");

		outfile.write("event_code = PAPI_FP_OPS;\n");
		outfile.write("retval = PAPI_add_event( event_set, event_code);\n");
		outfile.write("if( retval != PAPI_OK) {\n");
		outfile.write("  fprintf(stderr, \"Error occurred in PAPI_add_event - PAPI_FP_OPS!\\n\");\n");
		outfile.write("  exit(1);\n");
		outfile.write("}\n");
		  
		outfile.write("retval = PAPI_start( event_set);\n");
		outfile.write("if( retval != PAPI_OK) {\n");
		outfile.write("  fprintf(stderr, \"Error occurred in PAPI_start!\\n\");\n");
		outfile.write("  exit(1);\n");
		outfile.write("}\n");

            measure_region = measure_region + 1;

        elif line.find("SAC_ND_FUNAP2(SACf_C99Benchmarking__end__SACt_C99Benchmarking") != -1:
            measure_region = measure_region - 1;
            # If we are at the outermost measure region, print all measured flops 
            if measure_region == 0:
		outfile.write("retval = PAPI_stop( event_set, values);\n");
		outfile.write("if( retval != PAPI_OK) {\n");
		outfile.write("  fprintf(stderr, \"Error occurred in PAPI_stop!\\n\");\n");
		outfile.write("  exit(1);\n");
		outfile.write("}\n");
                outfile.write("printf(\"" + `size` + " %d\\n\", values[0]);\n"); 

        else:
            outfile.write(line);

    infile.close();
    outfile.close();
 
    os.system("mv " + tmp_file + " " + source);  


def instrument_time( source=""):
    global actual_measure_regions;
    actual_measure_regions = 0;
    measure_region = 0;
    infile = open( source, "r");  
    outfile = open( tmp_file, "w");
    for line in infile:
        if line.find("sac.h") != -1:
            outfile.write(line);
            outfile.write("#include <sys/time.h>\n");  
            if flops == 1:
                outfile.write("#include <papi.h>\n");  
            
            # Declare global timing variables for a maximum of 10 measure regions 
            i = 0;
            while i < max_measure_regions: 
                outfile.write("static double runtime" + `i` + " = 0.0;\n"); 
                outfile.write("static struct timeval start" + `i` + ", end" + `i` + ";\n"); 
                i = i + 1;
 
        elif line.find("SAC_ND_FUNAP2(SACf_C99Benchmarking__start__SACt_C99Benchmarking") != -1:
            outfile.write("gettimeofday( &start" + `measure_region` + ", NULL);\n");
            measure_region = measure_region + 1;
            actual_measure_regions = actual_measure_regions + 1;

        elif line.find("SAC_ND_FUNAP2(SACf_C99Benchmarking__end__SACt_C99Benchmarking") != -1:
            measure_region = measure_region - 1;
            outfile.write("gettimeofday( &end" + `measure_region` + ", NULL);\n");
            outfile.write("runtime" + `measure_region` + \
                          " += ((end" + `measure_region` + ".tv_sec*" + `time_factors[0]` + " + end" +`measure_region`+ ".tv_usec/" + `time_factors[1]` + ")-" + \
                          "     (start" + `measure_region` + ".tv_sec*" + `time_factors[0]` + " + start" + `measure_region` + ".tv_usec/" + `time_factors[1]` + "));\n");

            # If we are at the outermost measure region, print all measured runtime  
            if measure_region == 0:
                i = 0;
                while i < actual_measure_regions:
                     outfile.write("printf(\"%f\\n\", runtime" + `i` + ");\n"); 
                     i = i + 1;
                   
        else:
            outfile.write(line);

    infile.close();
    outfile.close();
 
    os.system("mv " + tmp_file + " " + source);   

if flops:
    i = 0;
    while i < len(sizes):
	cmd = "sac2c -v0 -O3 -d cccall -DSIZE=" + `sizes[i]` + " -DITER=" + `iterations` + " " +  prog + " -o " + sac_out_exe;
	print cmd;
	os.system(cmd); 

	#instrument compiler generated code
	instrument_flops( sac_out_src, sizes[i]);

        add_papi_library( sac_out_sac2c);

	#recompile after instrumentation
	os.system( sac_out_sac2c);

        os.system( "./" + sac_out_exe + " >> " + flops_file);

	i = i + 1;



"""
#===========================================================================================================================================
# Collect runtime for sac sequential
i = 0;
while i < len(sizes):
    cmd = "sac2c -v0 -O3 -d cccall -DSIZE=" + `sizes[i]` + " -DITER=" + `iterations` + " " +  prog + " -o " + sac_out_exe;
    print cmd;
    os.system(cmd); 

    #instrument compiler generated code
    instrument_time( sac_out_src);

    #recompile after instrumentation
    os.system( sac_out_sac2c);

    # Perform several runs and store the runtimes in tmp file 
    j = 0;
    while j < runs:
        os.system( "./" + sac_out_exe + " >> " + all_runtimes);
        j = j + 1;

    i = i + 1;

compute_average(all_runtimes, sac_seq);
os.system("rm " + all_runtimes);
"""

"""
#===========================================================================================================================================
# Collect runtime for sac multi-threaded
i = 0;
while i < len(sizes):
    cmd = "sac2c -mt -numthreads " + `threads` +  " -v0 -O3 -d cccall -DSIZE=" + `sizes[i]` + " -DITER=" + `iterations` + " " +  prog + " -o " + sac_out_exe;
    print cmd;
    os.system(cmd); 

    #instrument compiler generated code
    instrument_time( sac_out_src);

    #recompile after instrumentation
    os.system( sac_out_sac2c);

    # Perform several runs and store the runtimes in tmp file 
    j = 0;
    while j < runs:
        os.system( "./" + sac_out_exe + " >> " + all_runtimes);
        j = j + 1;

    i = i + 1;

compute_average(all_runtimes, sac_mt);
os.system("rm " + all_runtimes);
"""


"""
#===========================================================================================================================================
# Collect runtime for cuda (baseline)
i = 0;
while i < len(sizes):
    cmd = "sac2c -t cuda -v0 -O3 -nomemopt -d cccall -DSIZE=" + `sizes[i]` + " -DITER=" + `iterations` + " " +  prog + " -o " + cuda_out_exe;
    print cmd;
    os.system(cmd); 

    #instrument compiler generated code
    instrument_time( cuda_out_src);

    #recompile after instrumentation
    os.system( cuda_out_sac2c);

    # Perform several runs and store the runtimes in tmp file 
    j = 0;
    while j < runs:
        os.system( "./" + cuda_out_exe + " >> " + all_runtimes);
        j = j + 1;

    i = i + 1;

compute_average(all_runtimes, cuda_baseline);
os.system("rm " + all_runtimes);


#===========================================================================================================================================
# Collect runtime for cuda (memopt)
i = 0;
while i < len(sizes):
    cmd = "sac2c -t cuda -v0 -O3 -d cccall -DSIZE=" + `sizes[i]` + " -DITER=" + `iterations` + " " +  prog + " -o " + cuda_out_exe;
    print cmd;
    os.system(cmd); 

    #instrument compiler generated code
    instrument_time( cuda_out_src);

    #recompile after instrumentation
    os.system( cuda_out_sac2c);

    # Perform several runs and store the runtimes in tmp file 
    j = 0;
    while j < runs:
        os.system( "./" + cuda_out_exe + " >> " + all_runtimes);
        j = j + 1;

    i = i + 1;

compute_average(all_runtimes, cuda_memopt);
os.system("rm " + all_runtimes);

#===========================================================================================================================================
# Collect runtime for cuda (memopt+expar)

if expar == 1:
    i = 0;
    while i < len(sizes):
	cmd = "sac2c -t cuda -v0 -O3 -doexpar -d cccall -DSIZE=" + `sizes[i]` + " -DITER=" + `iterations` + " " +  prog + " -o " + cuda_out_exe;
	print cmd;
	os.system(cmd); 

	#instrument compiler generated code
	instrument_time( cuda_out_src);

	#recompile after instrumentation
	os.system( cuda_out_sac2c);

	# Perform several runs and store the runtimes in tmp file 
	j = 0;
	while j < runs:
	    os.system( "./" + cuda_out_exe + " >> " + all_runtimes);
	    j = j + 1;

	i = i + 1;

    compute_average(all_runtimes, cuda_expar);
    os.system("rm " + all_runtimes);

#===========================================================================================================================================
# Collect runtime for cuda (memopt+lao)

if lao == 1:
    i = 0;
    while i < len(sizes):
	cmd = "sac2c -t cuda -v0 -O3 -dolao -d cccall -DSIZE=" + `sizes[i]` + " -DITER=" + `iterations` + " " +  prog + " -o " + cuda_out_exe;
	print cmd;
	os.system(cmd); 

	#instrument compiler generated code
	instrument_time( cuda_out_src);

	#recompile after instrumentation
	os.system( cuda_out_sac2c);

	# Perform several runs and store the runtimes in tmp file 
	j = 0;
	while j < runs:
	    os.system( "./" + cuda_out_exe + " >> " + all_runtimes);
	    j = j + 1;

	i = i + 1;

    compute_average(all_runtimes, cuda_lao);
    os.system("rm " + all_runtimes);

#===========================================================================================================================================
# Collect runtime for cuda (memopt+pfd)

if pfd == 1:
    i = 0;
    while i < len(sizes):
	cmd = "sac2c -t cuda -v0 -O3 -dopfd -d cccall -DSIZE=" + `sizes[i]` + " -DITER=" + `iterations` + " " +  prog + " -o " + cuda_out_exe;
	print cmd;
	os.system(cmd); 

	#instrument compiler generated code
	instrument_time( cuda_out_src);

	#recompile after instrumentation
	os.system( cuda_out_sac2c);

	# Perform several runs and store the runtimes in tmp file 
	j = 0;
	while j < runs:
	    os.system( "./" + cuda_out_exe + " >> " + all_runtimes);
	    j = j + 1;

	i = i + 1;

    compute_average(all_runtimes, cuda_pfd);
    os.system("rm " + all_runtimes);
"""

