#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;

# Start from here
try:
    csv_file        = sys.argv[1];
    num_of_kernels  = sys.argv[2];
    num_of_sizes    = sys.argv[3];
except: 
    print "Usage:", sys.argv[0]," [input csv file] [num of kernels] [num of sizes]";
    sys.exit(1);

num_of_kernels = int(num_of_kernels);
num_of_sizes   = int(num_of_sizes);

analysis_output = "./analysis.out"
#compute_kernel_names = ["cuknl_1915", "cuknl_1920"];
compute_kernel_names = ["nest4", "nest9"];

def name_in_list( name, name_list):
    i = 0;
    while i < len(name_list):
        if name.find( name_list[i]) != -1:
            return True;
        i = i + 1;
    return False;


infile  = open( csv_file, "r");
outfile = open( analysis_output, "w");

i = 0;
while i < num_of_sizes:
    j = 0;
    copy_time = 0.0;
    compute_time = 0.0;
    while j < num_of_kernels:
        line = infile.readline();
        components = line.split(' ');
        if name_in_list(components[1], compute_kernel_names): 
            compute_time = compute_time + float(components[2]); 
        else:
            copy_time = copy_time + float(components[2]); 
        j = j + 1;
    outfile.write("compute time(" + components[0] + "): " + `compute_time` + "\n");
    outfile.write("copy time(" + components[0] + "): " + `copy_time` + "\n");
    i = i + 1;

infile.close();
outfile.close();

