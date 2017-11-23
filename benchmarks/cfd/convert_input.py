#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;

# Start from here
try:
    infilename     = sys.argv[1];
except: 
    print "Usage:", sys.argv[0]," [input file]";
    sys.exit(1);

infile = open( infilename, "r");
line = infile.readline();
size = int( line); # the first line stores the number of elements 

outfilenames = ["areas.input","surrounding.input","normals.input"];

outfile_areas = open( outfilenames[0], "w");
outfile_surrounding = open( outfilenames[1], "w");
outfile_normals = open( outfilenames[2], "w");

outfile_areas.write("[1," + `size` + ":\n")

surrounding_col1 = [];
surrounding_col2 = [];
surrounding_col3 = [];
surrounding_col4 = [];

normals_cols1 = [];
normals_cols2 = [];
normals_cols3 = [];


j = 0;
while j < size:

    line = infile.readline();
    components = line.split();

    # writting for area file
    outfile_areas.write( components[0] + "\n");     

    # writting for surrounding element file
    surrounding = int( components[1]);
    if surrounding < 0:
        surrounding_col1.append(-2);
    else:
        surrounding_col1.append(surrounding-1);
             
    surrounding = int( components[5]);
    if surrounding < 0:
        surrounding_col2.append(-2);
    else:
        surrounding_col2.append(surrounding-1);
             
    surrounding = int( components[9]);
    if surrounding < 0:
        surrounding_col3.append(-2);
    else:
        surrounding_col3.append(surrounding-1);

    surrounding = int( components[13]);
    if surrounding < 0:
        surrounding_col4.append(-2);
    else:
        surrounding_col4.append(surrounding-1);
  
    # writting for normals file 
    i = 0;
    while i < 4:
        normal = float( components[2+i*4]);
        normals_cols1.append( -normal);      # 4xsize elements 
        normal = float( components[3+i*4]);
        normals_cols2.append( -normal);      # 4xsize elements 
        normal = float( components[4+i*4]);
        normals_cols3.append( -normal);      # 4xsize elements 

        i = i + 1;
#        outfile_normals.write("]\n"); # for [1,3:
#        i = i + 1;

#    outfile_normals.write("]\n")
   
    j = j + 1;

#print "length of normals_cols1 " + `len(normals_cols1)`;
#print "length of normals_cols2 " + `len(normals_cols2)`;
#print "length of normals_cols3 " + `len(normals_cols3)`;

outfile_surrounding.write("[1,4:\n")
surrounding_cols = [surrounding_col1,surrounding_col2,surrounding_col3,surrounding_col4];


j = 0;
while j < 4:
    i = 0;
    outfile_surrounding.write("[1," + `size` + ":\n")
    while i < size:
        outfile_surrounding.write( `(surrounding_cols[j])[i]` + "\n");
        i = i + 1;
     
    outfile_surrounding.write( "]\n");
    j = j + 1;

outfile_surrounding.write("]\n") # for [1, 4: 

outfile_normals.write("[1,3:\n")
normals_cols = [normals_cols1, normals_cols2, normals_cols3];
j = 0;
while j < 3:
    i = 0;
    outfile_normals.write("[1,4:\n");

    count = 0;
    outfile_normals.write("[1," + `size` + ":\n");
    while count < size:
        outfile_normals.write( `(normals_cols[j])[i]` + "\n");

        i = i + 1;
        count = count + 1;
    outfile_normals.write("]\n");

    count = 0;
    outfile_normals.write("[1," + `size` + ":\n");
    while count < size:
        outfile_normals.write( `(normals_cols[j])[i]` + "\n");

        i = i + 1;
        count = count + 1;
    outfile_normals.write("]\n");

    count = 0;
    outfile_normals.write("[1," + `size` + ":\n");
    while count < size:
        outfile_normals.write( `(normals_cols[j])[i]` + "\n");

        i = i + 1;
        count = count + 1;
    outfile_normals.write("]\n");

    count = 0;
    outfile_normals.write("[1," + `size` + ":\n");
    while count < size:
        outfile_normals.write( `(normals_cols[j])[i]` + "\n");

        i = i + 1;
        count = count + 1;
    outfile_normals.write("]\n");

    outfile_normals.write("]\n");
    j = j + 1;        


outfile_normals.write("]\n") # for [1, 3: 
outfile_areas.write("]\n") # for [1, size: 

infile.close();
outfile_areas.close();
outfile_surrounding.close();
outfile_normals.close();

cmd1 = "cat " + outfilenames[1] + " >> " + outfilenames[0];  
print cmd1;
os.system(cmd1);
cmd2 = "cat " + outfilenames[2] + " >> " + outfilenames[0];  
print cmd2;
os.system(cmd2);
cmd3 = "mv " + outfilenames[0] + " sac_input_" + `size`;
print cmd3;
os.system(cmd3);
cmd4 = "rm " + outfilenames[1] + " " + outfilenames[2];
print cmd4;
os.system(cmd4);



