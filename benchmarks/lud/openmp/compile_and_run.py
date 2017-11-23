#!/usr/bin/env python
import sys;
import time;
import datetime;
import os;
import random;

sizes = [1024, 2048, 3072, 4096]; 
#sizes = [1024]; 

os.system("rm copy_compute_time.txt"); 

cmd = "gcc  -O3 -DNIPU -DSIZE=" + `sizes[0]` + " lud_one_file.c";
print cmd;
os.system(cmd); 
os.system("./a.out ../input/" + `sizes[0]` + ".dat >> copy_compute_time.txt" ); 

cmd = "gcc  -O3 -DNIPU -DSIZE=" + `sizes[1]` + " lud_one_file.c";
print cmd;
os.system(cmd); 
os.system("./a.out ../input/" + `sizes[1]` + ".dat >> copy_compute_time.txt" ); 

cmd = "gcc  -O3 -DNIPU -DSIZE=" + `sizes[2]` + " lud_one_file.c";
print cmd;
os.system(cmd); 
os.system("./a.out ../input/" + `sizes[2]` + ".dat >> copy_compute_time.txt" ); 

cmd = "gcc  -O3 -DNIPU -DSIZE=" + `sizes[3]` + " lud_one_file.c";
print cmd;
os.system(cmd); 
os.system("./a.out ../input/" + `sizes[3]` + ".dat >> copy_compute_time.txt" ); 


os.system("rm reuse_time.txt"); 

cmd = "gcc  -O3 -DIPU -DSIZE=" + `sizes[0]` + " lud_one_file.c";
print cmd;
os.system(cmd); 
os.system("./a.out ../input/" + `sizes[0]` + ".dat >> reuse_time.txt" ); 

cmd = "gcc  -O3 -DIPU -DSIZE=" + `sizes[1]` + " lud_one_file.c";
print cmd;
os.system(cmd); 
os.system("./a.out ../input/" + `sizes[1]` + ".dat >> reuse_time.txt" ); 

cmd = "gcc  -O3 -DIPU -DSIZE=" + `sizes[2]` + " lud_one_file.c";
print cmd;
os.system(cmd); 
os.system("./a.out ../input/" + `sizes[2]` + ".dat >> reuse_time.txt" ); 

cmd = "gcc  -O3 -DIPU -DSIZE=" + `sizes[3]` + " lud_one_file.c";
print cmd;
os.system(cmd); 
os.system("./a.out ../input/" + `sizes[3]` + ".dat >> reuse_time.txt" ); 


