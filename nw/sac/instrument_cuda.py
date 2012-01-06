#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;

# Start from here
try:
    prog      = sys.argv[1];
    nth_goto  = sys.argv[2];
    nth_while = sys.argv[3];
    compute_kernel_time = sys.argv[4];
except: 
    print "Usage:", sys.argv[0]," [program] [nth goto?(starting from 1)] [nth while?(starting from 1)] [compute kernel time?(1 yes; 0 no)]";
    sys.exit(1);

nth_goto  = int(nth_goto);
nth_while = int(nth_while);
compute_kernel_time = int(compute_kernel_time);

num_of_kernels = 0;
runs = 4;
kernel_names = [];  

sac_out_exe = "sac_out";
sac_out_src = "sac_out.c";
sac_out_sac2c = "./sac_out.sac2c";

cuda_out_exe = "cuda_out";
cuda_out_src = "cuda_out.cu";
cuda_out_sac2c = "./cuda_out.sac2c";

tmp_file = "tmp";
sac_no_reuse = "./runtimes/sac_no_reuse.csv"
cuda_no_reuse_loop = "./runtimes/cuda_no_reuse_loop.csv"
cuda_no_reuse_kernel = "./runtimes/cuda_no_reuse_kernel.csv"
sac_reuse = "./runtimes/sac_reuse.csv"
cuda_reuse = "./runtimes/cuda_reuse.csv"
cuda_reuse_rnb = "./runtimes/cuda_reuse_rnb.csv"


sizes = [256,512,1024,2048,3072,4096];
#sizes = [256];

def instrument_loop( source="", time_kernel=0, kernel_count=0, prob_size=0):
    _goto = 1;
    _while = 1;
    kernel = 0;
    infile = open( source, "r");  
    outfile = open( tmp_file, "w");
    for line in infile:
        if line.find("sac.h") != -1:
            outfile.write("#include <sys/time.h>\n");  
            outfile.write(line);
        elif line.find("SAC_ND_GOTO") != -1:
            if _goto == nth_goto:
                #declare all time variables to store each kernel's total execution time
                if time_kernel == 1:
                    outfile.write("double ");
                    i = 0;
                    while i < kernel_count:
                        outfile.write("kernel_" + `i` + "_time=0.0");
                        if i != kernel_count-1:
                            outfile.write(",");
                        i = i + 1; 
                    outfile.write(";\n");
                else:
                   outfile.write("struct timeval loop_start, loop_end;\n");  
                   outfile.write("gettimeofday( &loop_start, NULL);\n");

            outfile.write(line);
            _goto = _goto + 1;
        elif line.find("while") != -1:
            outfile.write(line);
            if _while == nth_while:
                if time_kernel:
                    i = 0;
                    while i < kernel_count:
                        outfile.write("printf(\"%f\\n\", kernel_" + `i` + "_time);\n");
                        i = i + 1; 
                else:
                    outfile.write("gettimeofday( &loop_end, NULL);\n");
                    outfile.write("double runtime = ((loop_end.tv_sec*1000.0 + loop_end.tv_usec/1000.0)-(loop_start.tv_sec*1000.0 + loop_start.tv_usec/1000.0));\n");
                    outfile.write("printf(\"%f\\n\", runtime);\n");              
            _while = _while + 1;
        elif line.find("<<<") != -1:
            if time_kernel == 1:
                outfile.write("struct timeval kernel_start, kernel_end;\n");  
                outfile.write("gettimeofday( &kernel_start, NULL);\n");  
                outfile.write(line);
                #outfile.write("cudaThreadSynchronize();\n");  
                outfile.write("gettimeofday( &kernel_end, NULL);\n");  
                outfile.write("kernel_" + `kernel` + "_time += ((kernel_end.tv_sec*1000.0 + kernel_end.tv_usec/1000.0)-(kernel_start.tv_sec*1000.0 + kernel_start.tv_usec/1000.0));\n");
            else:
                outfile.write(line);
          
            kernel = kernel + 1;
        else:
            outfile.write(line);

    infile.close();
    outfile.close();
 
    os.system("mv " + tmp_file + " " + source);             

def count_kernels( source=""):
    kernel_count = 0;
    infile = open( source, "r");  
    for line in infile:
        if line.find("<<<") != -1:
            kernel_count = kernel_count + 1;
            kernel_names.append(line[line.find("SACf"):line.find("CUDA")]);
    infile.close();
    return kernel_count;

def compute_loop_average( size, src, dst):
    infile = open( src, "r");
    outfile = open( dst, "a");
    time = 0.0;
    for line in infile:
        time = time + float(line); 
    avg_time = time/runs;
    outfile.write( `size` + " " + `avg_time` + "\n"); 
    infile.close();
    outfile.close();

def compute_kernel_average( size, src, dst):
    infile = open( src, "r");
    outfile = open( dst, "a");
  
    times = [];

    i = 0;
    while i < runs:
        j = 0;
        while j < num_of_kernels:
            line = infile.readline(); 
            if i == 0:
                times.append(float(line)); 
            else:
                times[j] = times[j] + float(line);
            j = j + 1;
        i = i + 1;

    i = 0;
    while i < num_of_kernels:
        times[i] = times[i]/runs;
        outfile.write( `size` + " " + kernel_names[i] + " " +  `times[i]` + "\n"); 
        i = i + 1;

    infile.close();
    outfile.close();

"""
i = 0;
while i < len(sizes):
    cmd = "sac2c -t cuda -v0 -O3 -d cccall -DSIZE=" + `sizes[i]` + " " + prog + " -o " + cuda_out_exe;
    print cmd;
    os.system(cmd); 

    #instrument compiler generated code
    instrument_loop( cuda_out_src, 0, -1, sizes[i]);

    #recompile after instrumentation
    os.system( cuda_out_sac2c);
 
    #cleanup
    os.system("rm " + tmp_file);

    j = 0;
    while j < runs:
        os.system( "./" + cuda_out_exe + " >> " + tmp_file);
        j = j + 1;
    compute_loop_average(sizes[i], tmp_file, cuda_no_reuse_loop);

    #==================================================================

    if compute_kernel_time == 1:
	cmd = "sac2c -t cuda -v0 -O3 -d cccall -DSIZE=" + `sizes[i]` + " " + prog + " -o " + cuda_out_exe;
	print cmd;
	os.system(cmd); 

	if i == 0:
	    num_of_kernels = count_kernels( cuda_out_src);

	#instrument compiler generated code
	instrument_loop( cuda_out_src, 1, num_of_kernels, sizes[i]);

	#recompile after instrumentation
	os.system( cuda_out_sac2c);
     
	#cleanup
	os.system("rm " + tmp_file);

	j = 0;
	while j < runs:
	    os.system( "./" + cuda_out_exe + " >> " + tmp_file);
	    j = j + 1;

	compute_kernel_average(sizes[i], tmp_file, cuda_no_reuse_kernel);

    i = i + 1;

kenrel_names = [];
"""

i = 0;
while i < len(sizes):
    cmd = "sac2c -t cuda -v0 -O3 -dopra -d cccall -DSIZE=" + `sizes[i]` + " " + prog + " -o " + cuda_out_exe;
    print cmd;
    os.system(cmd); 
    instrument_loop( cuda_out_src, 0, -1, sizes[i]);
    os.system( cuda_out_sac2c);
    os.system("rm " + tmp_file);
    j = 0;
    while j < runs:
        os.system( "./" + cuda_out_exe + " >> " + tmp_file);
        j = j + 1;
    compute_loop_average(sizes[i], tmp_file, cuda_reuse);
    i = i + 1;

i = 0;
while i < len(sizes):
    cmd = "sac2c -t cuda -v0 -O3 -dopra -dornb -d cccall -DSIZE=" + `sizes[i]` + " " + prog + " -o " + cuda_out_exe;
    print cmd;
    os.system(cmd); 
    instrument_loop( cuda_out_src, 0, -1, sizes[i]);
    os.system( cuda_out_sac2c);
    os.system("rm " + tmp_file);
    j = 0;
    while j < runs:
        os.system( "./" + cuda_out_exe + " >> " + tmp_file);
        j = j + 1;
    compute_loop_average(sizes[i], tmp_file, cuda_reuse_rnb);
    i = i + 1;

