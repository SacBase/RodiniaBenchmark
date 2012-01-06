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
    compute_nest_time = sys.argv[4];
except: 
    print "Usage:", sys.argv[0]," [program] [nth goto?(starting from 1)] [nth while?(starting from 1)] [compute nest time?(1 yes; 0 no)]";
    sys.exit(1);

nth_goto  = int(nth_goto);
nth_while = int(nth_while);
compute_nest_time = int(compute_nest_time);

num_of_forloop_nests = 0;
runs = 2;
nest_names = [];  

sac_out_exe = "sac_out";
sac_out_src = "sac_out.c";
sac_out_sac2c = "./sac_out.sac2c";

cuda_out_exe = "cuda_out";
cuda_out_src = "cuda_out.cu";
cuda_out_sac2c = "./cuda_out.sac2c";

tmp_file = "tmp";
sac_no_reuse_loop = "./runtimes/sac_no_reuse_loop.csv"
sac_no_reuse_nest = "./runtimes/sac_no_reuse_nest.csv"
cuda_no_reuse_loop = "./runtimes/cuda_no_reuse_loop.csv"
cuda_no_reuse_kernel = "./runtimes/cuda_no_reuse_kernel.csv"
sac_reuse = "./runtimes/sac_reuse.csv"
cuda_reuse = "./runtimes/cuda_reuse.csv"


sizes = [1024,2048,3072,4096];
#sizes = [4096];

def instrument_loop( source="", time_nest=0, nest_count=0):
    _goto = 1;
    _while = 1;
    nest = 0;
    infile = open( source, "r");  
    outfile = open( tmp_file, "w");
    for line in infile:
        if line.find("sac.h") != -1:
            outfile.write("#include <sys/time.h>\n");  
            outfile.write(line);
        elif line.find("SAC_ND_GOTO") != -1:
            if _goto == nth_goto:
                #declare all time variables to store each kernel's total execution time
                if time_nest == 1:
                    outfile.write("double ");
                    i = 0;
                    while i < nest_count:
                        outfile.write("nest_" + `i` + "_time=0.0");
                        if i != nest_count-1:
                            outfile.write(",");
                        i = i + 1; 
                    outfile.write(";\n");
                    outfile.write("struct timeval nest_start, nest_end;\n");  
                else:
                    outfile.write("struct timeval loop_start, loop_end;\n");  
                    outfile.write("gettimeofday( &loop_start, NULL);\n");

            outfile.write(line);
            _goto = _goto + 1;
        elif line.find("while") != -1:
            outfile.write(line);
            if _while == nth_while:
                if time_nest == 1:
                    i = 0;
                    while i < nest_count:
                        outfile.write("printf(\"%f\\n\", nest_" + `i` + "_time);\n");
                        i = i + 1; 
                else:
                    outfile.write("gettimeofday( &loop_end, NULL);\n");
                    outfile.write("double runtime = ((loop_end.tv_sec*1000.0 + loop_end.tv_usec/1000.0)-(loop_start.tv_sec*1000.0 + loop_start.tv_usec/1000.0));\n");
                    outfile.write("printf(\"%f\\n\", runtime);\n");              
            _while = _while + 1;
        elif line.find("SAC_WL_STRIDE_LOOP0_BEGIN(0") != -1:
            if time_nest == 1:
                outfile.write("gettimeofday( &nest_start, NULL);\n");  
                outfile.write(line);
            else:
                outfile.write(line);
        elif line.find("SAC_WL_STRIDE_LOOP_END(0") != -1:
            if time_nest == 1:
                outfile.write(line);
                outfile.write("gettimeofday( &nest_end, NULL);\n");  
                outfile.write("nest_" + `nest` + "_time += ((nest_end.tv_sec*1000.0 + nest_end.tv_usec/1000.0)-(nest_start.tv_sec*1000.0 + nest_start.tv_usec/1000.0));\n");
            else:
                outfile.write(line);     
            nest = nest + 1;
        else:
            outfile.write(line);

    infile.close();
    outfile.close();
 
    os.system("mv " + tmp_file + " " + source);             

def count_forloop_nests( source=""):
    nest_count = 0;
    infile = open( source, "r");  
    for line in infile:
        if line.find("SAC_WL_STRIDE_LOOP0_BEGIN(0") != -1:
            nest_names.append("nest" + `nest_count`);
            nest_count = nest_count + 1;
    infile.close();
    return nest_count;

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

def compute_nest_average( size, src, dst):
    infile = open( src, "r");
    outfile = open( dst, "a");
  
    times = [];

    i = 0;
    while i < runs:
        j = 0;
        while j < num_of_forloop_nests:
            line = infile.readline(); 
            if i == 0:
                times.append(float(line)); 
            else:
                times[j] = times[j] + float(line);
            j = j + 1;
        i = i + 1;

    i = 0;
    while i < num_of_forloop_nests:
        times[i] = times[i]/runs;
        outfile.write( `size` + " " + nest_names[i] + " " +  `times[i]` + "\n"); 
        i = i + 1;

    infile.close();
    outfile.close();

i = 0;
while i < len(sizes):
    cmd = "sac2c -v0 -O3 -norip -norwo -d cccall -DSIZE=" + `sizes[i]` + " " + prog + " -o " + sac_out_exe;
    print cmd;
    os.system(cmd); 

    #instrument compiler generated code
    instrument_loop( sac_out_src, 0, -1);

    #recompile after instrumentation
    os.system( sac_out_sac2c);

    #cleanup
    os.system("rm " + tmp_file);

    j = 0;
    while j < runs:
        os.system( "./" + sac_out_exe + " ../input/sac_" + `sizes[i]` + ".dat >> " + tmp_file);
        j = j + 1;
    compute_loop_average(sizes[i], tmp_file, sac_no_reuse_loop);

    if compute_nest_time == 1:
	cmd = "sac2c -v0 -O3 -norip -norwo -d cccall -DSIZE=" + `sizes[i]` + " " + prog + " -o " + sac_out_exe;
	print cmd;
	os.system(cmd); 

	if i == 0:
	    num_of_forloop_nests = count_forloop_nests( sac_out_src);

	#instrument compiler generated code
	instrument_loop( sac_out_src, 1 , num_of_forloop_nests);

	#recompile after instrumentation
	os.system( sac_out_sac2c);

	#cleanup
	os.system("rm " + tmp_file);

	j = 0;
	while j < runs:
	    os.system( "./" + sac_out_exe + " ../input/sac_" + `sizes[i]` + ".dat >> " + tmp_file);
	    j = j + 1;
	compute_nest_average(sizes[i], tmp_file, sac_no_reuse_nest);
      
    i = i + 1;

"""
i = 0;
while i < len(sizes):
    cmd = "sac2c -v0 -O3 -dopra -d cccall -DSIZE=" + `sizes[i]` + " " + prog + " -o " + sac_out_exe;
    print cmd;
    os.system(cmd); 
    instrument_loop( sac_out_src, 0, -1);
    os.system( sac_out_sac2c);
    os.system("rm " + tmp_file);
    j = 0;
    while j < runs:
        os.system( "./" + sac_out_exe + " ../input/sac_" + `sizes[i]` + ".dat >> " + tmp_file);
        j = j + 1;
    compute_loop_average(sizes[i], tmp_file, sac_reuse);
    i = i + 1;
"""
