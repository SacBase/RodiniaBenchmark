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
number_of_nodes = int( line); # the first line stores the number of nodes 

outfile = open( outfilename, "w");

graph_nodes_starting = [];
graph_nodes_edges = [];
graph_edges = [];

j = 0;
while j < number_of_nodes:
    line = infile.readline();
    components = line.split();

    graph_nodes_starting.append(components[0]);
    graph_nodes_edges.append(components[1]);

    j = j + 1;

# read the following three dummy lines
line = infile.readline();
print "Dummy line 1:" + line;
line = infile.readline();
print "Dummy line 2:" + line;
line = infile.readline();
print "Dummy line 3:" + line;

line = infile.readline();
number_of_edges = int( line); # this line stores the number of edges 

j = 0;
while j < number_of_edges:
    line = infile.readline();
    components = line.split();

    if int(components[0]) == 0:
        print "found node 0 as a son!" + `j`;

    graph_edges.append(components[0]);

    j = j + 1;

# close input file
infile.close();

# write the graph node starting array
outfile.write("[1," + `number_of_nodes` + ":\n");
i = 0;
while i < number_of_nodes: 
    outfile.write(graph_nodes_starting[i] + "\n");
    i = i + 1;
outfile.write("]\n");

# write the graph node edge array
outfile.write("[1," + `number_of_nodes` + ":\n");
i = 0;
while i < number_of_nodes: 
    outfile.write(graph_nodes_edges[i] + "\n");
    i = i + 1;
outfile.write("]\n");

# write the graph node edge array
#stats = [];
#i = 0;
#while i < number_of_nodes: 
#    stats.append(0);
#    i = i + 1;

outfile.write("[1," + `number_of_edges` + ":\n");
i = 0;
while i < number_of_edges: 
    outfile.write(graph_edges[i] + "\n");
#    stats[int(graph_edges[i])] = stats[int(graph_edges[i])] + 1;   
    i = i + 1;
outfile.write("]\n");

#i = 0;
#while i < number_of_nodes: 
#    outfile.write(`i` + ":" + `stats[i]` + "\n");
#    i = i + 1;


outfile.close();

