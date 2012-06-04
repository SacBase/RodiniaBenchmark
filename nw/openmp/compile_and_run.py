#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;

sizes = [1024, 2048, 3072, 4096]; 
#sizes = [64, 128,256,512]; 

os.system("rm copy_compute_time.txt"); 

i = 0;
while i < 4:
    cmd = "gcc -O3 -DV=3 -DSIZE=" + `sizes[i]` + " needle.c";
    print cmd;
    os.system(cmd); 
    os.system("./a.out >> copy_compute_time.txt" ); 
    i = i + 1;


os.system("rm reuse_time.txt"); 

i = 0;
while i < 4:
    cmd = "gcc  -O3 -DSIZE=" + `sizes[i]` + " needle.c";
    print cmd;
    os.system(cmd); 
    os.system("./a.out >> reuse_time.txt" ); 
    i = i + 1;


