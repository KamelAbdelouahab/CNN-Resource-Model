import sys
import os

def getSummaryALM(fit_filename):
    fp = open(fit_filename)
    lineALM,wordALM = (7,5)             # ALM utilization is in line 8, word 6
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
    num_alm_pos = 6          # Number of ALMs used is the 6th word of the line
    instance_pos = [] # name of instance in the 1st word of line (after the ;)
    instance = []
    num_alm = []
    fp = open(fit_rpt_filename)
    instance_occur = 0;
    # have to add instance index to have proper out dictionary

    for i,line in enumerate(fp):
        if instance_name in line:
            instance_occur += 1
            # ALMs Needed = 3 , ALMs Used = 6
            num_alm.append(line.split()[3])
    num_alm = list(map(float,num_alm))
    instance_number = listAsQuartus(instance_occur)
    d = (dict(zip(instance_number,num_alm)))
    return d

def getALMStop(fit_rpt_filename,instance_name, stop_inst_name):
    num_alm_pos = 6          # Number of ALMs used is the 6th word of the line
    instance_pos = [] # name of instance in the 1st word of line (after the ;)
    instance = []
    num_alm = []
    fp = open(fit_rpt_filename)
    instance_occur = 0;
    # have to add instance index to have proper out dictionary

    for i,line in enumerate(fp):
        if instance_name in line:
            instance_occur += 1
            num_alm.append(line.split()[3])
        if stop_inst_name in line:
            break
    num_alm = list(map(float,num_alm))
    instance_number = listAsQuartus(instance_occur)
    d = (dict(zip(instance_number,num_alm)))
    return d

def dispSortedDict(dictionary):
    for key in sorted(dictionary):
        print("%s: %s" % (key, dictionary[key]))

def listAsQuartus(length):
    #    0 1 10 11 12 13 14 15 16 17 18 19 2 20 21 22 23 24 25 26 27 28 29
    # -> 0 10 11 12 13 14 15 16 17 18 19 1 2 20 21 22 23 24 25 26 27 28 29
    # -> 0 10 11 12 13 14 15 16 17 18 19 1 20 21 22 23 24 25 26 27 28 29 2
    l = []
    for x in range(length):
        l.append(x)
    l = sorted(l,key=str)
    x,y = 0,0;
    while y < length: #BUG HERE: But im too tired
        x = y + 1
        y = 11 + y
        if y > length:
            break
        else:
            l.insert(y, l.pop(x))
    # print("Quartus List")
    # print(l)
    return l
