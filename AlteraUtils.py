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

def getALM(fit_rpt_filename,instance_name):
    num_alm_pos = 6 # Number of ALMs used is the 6th word of the line
    instance_pos = [] # name of instance in the 1st word of line (after ;)
    instance = []
    num_alm = []
    fp = open(fit_rpt_filename)
    instance_occur = 0; #have to add instance index to have proper out dictionary
    #

    for i,line in enumerate(fp):
        if instance_name in line:
            instance_occur += 1
            num_alm.append(line.split()[6])
    num_alm = list(map(float,num_alm))
    instance_number = listAsQuartus(instance_occur)
    d = (dict(zip(instance_number,num_alm)))
    return d

def dispSortedDict(dictionary):
    for key in sorted(dictionary):
        print("%s: %s" % (key, dictionary[key]))

def listAsQuartus(length):
    l = []
    for x in range(length):
        l.append(x)
    l = sorted(l,key=str)
    l.insert(11, l.pop(1))
    return l

if __name__ == '__main__':
    fit_rpt = "example/quartus/FittingReport.txt"
    # instance_name = ";          |MOA:MOA_i|"
    instance_name = ";          |MCM:MCM_i|"
    # instance_name = ";       |DotProduct:"
    dd = getALM(fit_rpt, instance_name)
    dispSortedDict(dd)
