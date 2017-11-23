#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;


# Start from here
try:
    sizestr        = sys.argv[1] 
    infilename     = sys.argv[2];
    outfilename    = sys.argv[3];
except: 
    print "Usage:", sys.argv[0]," [input size] [input file] [output file]";
    sys.exit(1);

size = int( sizestr);
infile = open( infilename, "r");
outfile = open( outfilename, "w");

outfile.write("[1," + `size` + ":\n")
j = 0;
while j < (size*size):
    i = 0;
    outfile.write("[1," + `size` + ":\n")
    while i < size:
        line = infile.readline();
        outfile.write( line);
        i = i + 1;
    outfile.write("]\n")
    j = j + size;
outfile.write("]")

infile.close();
outfile.close();


