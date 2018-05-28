import sys
import os

def getALM(fit_filename):
    fp = open(fit_filename)
    lineALM,wordALM = (7,5)      # ALM utilization is in line 8, word 6 of .fit.summary file
    for i,line in enumerate(fp):
        if (i == lineALM):
            print(line.split()[wordALM])
            return int(line.split()[wordALM])

def quartusFit(qpf_filename):
    QUARTUS_DIR = "C:/Intel-FPGA/quartus/bin64/"
    cmd = QUARTUS_DIR + "quartus_sh --flow compile " + qpf_filename
    os.system(cmd)

def writeLine(file_name, line_number, line):
    with open(file_name,'r') as f:
        lines = f.readlines()
    lines[line_number] = line
    with open(file_name, 'w') as g:
        g.writelines(lines)
