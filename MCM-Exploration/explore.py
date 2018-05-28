import fileinput
import sys
import time
import os
import numpy as np


def getALM(fitFile):
    fp = open(fitFile)
    lineALM,wordALM = (7,5)      # ALM utilization is in line 8, word 6 of .fit.summary file
    for i,line in enumerate(fp):
        if (i == lineALM):
            print(line.split()[wordALM])
            return int(line.split()[wordALM])

def quartusFit(projName):
    QUARTUS_DIR = "C:/Intel-FPGA/quartus/bin64/"
    cmd = QUARTUS_DIR + "quartus_sh --flow compile quartus/" + projName + ".qpf"
    os.system(cmd)

def writeLine(file_name, line_number, line):
    with open(file_name,'r') as f:
        lines = f.readlines()
    lines[line_number] = line
    with open(file_name, 'w') as g:
        g.writelines(lines)



def explore(projName,dataRange,step):
    previousVal = 0
    vals = np.zeros([])                                                 # Multiplicand value
    alms = np.zeros([])                                       # Corresponding number of ALMs
    vhdFile  = "quartus/"+ projName + ".vhd"                                 # VHDL Filename
    fitFile  = "quartus/output_files/" + projName + ".fit.summary" # Fitting report Filename
    resFile  = projName + ".csv"                                       # CSV Result Filename

    for val in range(-dataRange,dataRange,step):
        valStr = "KERNEL : std_logic_vector(7 downto 0) := std_logic_vector(to_signed("
        valStr += str(val)
        valStr += ",8))\n"
        writeLine(vhdFile,6,valStr)                # Replace multiplicand value in VHDL File
        previousVal = val;
        quartusFit(projName);                                       # Lunch Quartus fitting
        numALM = getALM(fitFile)                                       # Read number of ALMs
        alms = np.append(alms,numALM)
        vals = np.append(vals,val)

    data = np.zeros((2*dataRange + 1,2));               # Save into matrix and export to CSV
    data[:,0] = vals;
    data[:,1] = alms;
    np.savetxt(resFile,data,fmt='%d',delimiter=',')

if __name__ == '__main__':

    projName = "constMult"
    dataRange = 127;
    step = 1;
    explore(projName,dataRange,step)
