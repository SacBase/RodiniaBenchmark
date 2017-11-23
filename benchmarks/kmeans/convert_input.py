#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;

# Start from here
try:
    infilename      = sys.argv[1];
    outfilename     = sys.argv[2];
except: 
    print "Usage:", sys.argv[0]," [input file] [output file]";
    sys.exit(1);

infile = open( infilename, "r");
line = infile.readline();
size = int( line); # the first line stores the number of elements 

outfile = open( outfilename, "w");

input_array = [];

j = 0;
while j < size:
    line = infile.readline();
    components = line.split();
    input_array.append([]);

    k = 1;
    while k <= 34:
        # skip the first element
        try:
            input_array[j].append(float(components[k]));
        except:
            print "component is: " + components[k];
        k = k + 1;

    j = j + 1;

#print input_array;
infile.close();

outfile.write("[1,34:\n");
i = 0;
while i < 34: 
    j = 0;
    outfile.write("[1," + `size` + ":\n");
    while j < size:
        outfile.write(`input_array[j][i]` + "\n");
        j = j + 1;
    outfile.write("]\n");

    i = i + 1;

outfile.write("]\n");
outfile.close();

