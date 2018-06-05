import sys
import os
import time
import numpy as np
import AlteraUtils

def ExploreMOA(vhdl_filename,
               fit_filename,
               qpf_filename,
               num_opd_max,
               csv_filename):
    nopd = np.zeros([])                                       # Number of operands
    alms = np.zeros([])                             # Corresponding number of ALMs
    exploration_res = np.zeros([num_opd_max-1,2]);
    for n in range(1,num_opd_max,2):
        num_opd_str = "    constant CONST_NUM_OPERANDS : natural := "
        num_opd_str += str(n)
        num_opd_str += "; \n"
        AlteraUtils.writeLine(vhdl_filename, 7, num_opd_str)
        AlteraUtils.quartusFit(qpf_filename);
        num_alm = AlteraUtils.getALM(fit_filename)
        exploration_res[n-1,:] = np.array([n,num_alm])
        np.savetxt(csv_filename, exploration_res, fmt='%d', delimiter=',')

if __name__ == '__main__':
    vhdl_filename = "MOA-Exploration/DataTypes.vhd"
    fit_filename = "MOA-Exploration/output_files/ParallelMOA.fit.summary"
    qpf_filename = "MOA-Exploration/ParallelMOA.qpf"
    csv_filename = "MOA16bits.csv"
    ExploreMOA(vhdl_filename = vhdl_filename,
               fit_filename = fit_filename,
               qpf_filename = qpf_filename,
               num_opd_max = 10,
               csv_filename = csv_filename)
