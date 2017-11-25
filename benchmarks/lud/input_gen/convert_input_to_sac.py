#!/usr/bin/env python2
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
size = int( line); # the first line stores the number of nodes

outfile = open( outfilename, "w");

outfile.write("[1," + `size` + ":\n");

j = 0;
while j < size:
    line = infile.readline();

    outfile.write("[1," + `size` + ":\n");
    outfile.write(line);
    outfile.write("]\n");
    j = j + 1;

outfile.write("]\n");

# close input and output files
infile.close();
outfile.close();

