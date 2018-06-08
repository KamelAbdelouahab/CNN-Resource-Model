import sys
import os
import time
import numpy as np
import AlteraUtils

def ExploreMOA(vhdl_filename,
               fit_filename,
               qpf_filename,
               num_opd_max,
               stride,
               csv_filename):
    nopd = np.zeros([])                                       # Number of operands
    alms = np.zeros([])                             # Corresponding number of ALMs
    nb_iter = 0
    for i in range(0, num_opd_max, stride):
        nb_iter += 1
    exploration_res = np.zeros([nb_iter,2]);

    for n in range(0, num_opd_max, stride):
        num_opd_str = "    constant CONST_NUM_OPERANDS : natural := "
        num_opd_str += str(n)
        num_opd_str += "; \n"
        AlteraUtils.writeLine(vhdl_filename, 7, num_opd_str)
        AlteraUtils.quartusFit(qpf_filename);
        num_alm = AlteraUtils.getSummaryALM(fit_filename)
        exploration_res[n-1,:] = np.array([n,num_alm])
        np.savetxt(csv_filename, exploration_res, fmt='%d', delimiter=',')

if __name__ == '__main__':
    vhdl_filename = "MOA-Exploration/DataTypes.vhd"
    fit_filename = "MOA-Exploration/output_files/ParallelMOA.fit.summary"
    qpf_filename = "MOA-Exploration/ParallelMOA.qpf"
    csv_filename = "reports/MOA6bits.csv"
    ExploreMOA(vhdl_filename = vhdl_filename,
               fit_filename = fit_filename,
               qpf_filename = qpf_filename,
               num_opd_max = 200,
               stride= 5,
               csv_filename = csv_filename)
