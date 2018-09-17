import sys
import os
import time
import numpy as np
import AlteraUtils


def ExploreWinBuffer(vhdl_filename,
                     fit_filename,
                     qpf_filename,
                     num_buff_max,
                     stride,
                     csv_filename):

    # I am tired: Please excuse me
    nb_iter = 0
    for i in range(0, num_buff_max, stride):
        nb_iter += 1

    # [iter, num_alm, num_ram]
    exploration_res = np.zeros([nb_iter, 3])

    for n in range(0, num_buff_max, stride):
        num_buff_str = "   NB_IN_FLOWS : integer := "
        num_buff_str += str(n)
        num_buff_str += "\n"
        AlteraUtils.writeLine(vhdl_filename, 20, num_buff_str)
        AlteraUtils.quartusFit(qpf_filename)
        num_alm = AlteraUtils.getSummaryALM(fit_filename)
        num_ram = AlteraUtils.getSummaryRAM(fit_filename)
        exploration_res[n - 1, :] = np.array([n, num_alm, num_ram])
        np.savetxt(csv_filename, exploration_res, fmt='%d', delimiter=',')

def PlotData(csv_filename):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    params = {
        'grid.color': 'k',
        'grid.linestyle': 'dashdot',
        'grid.linewidth': 0.6,
        'font.family': 'Linux Biolinum O',
        'axes.labelsize': 15,
        'font.size': 15,
        'legend.fontsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.facecolor': 'white'
    }
    rcParams.update(params)

    data = np.loadtxt(csv_filename, delimiter=',')
    win_buff_num = data[:,0]
    alm_num = data[:,1]
    ram_num = data[:,2]

    plt.figure(figsize=(4, 3))
    plt.grid()
    plt.plot(win_buff_num, alm_num, 'b^-')
    plt.ylabel('ALMs')
    plt.savefig("Results/WinBuffALM.pdf", bbox_inches ='tight')
    plt.figure(figsize=(4, 3))
    plt.plot(win_buff_num, ram_num, 'b^-')
    plt.grid()
    plt.ylabel('M20K')
    plt.savefig("Results/WinBuffRAM.pdf", bbox_inches ='tight')


if __name__ == '__main__':
    vhdl_filename = "WinBuffer-Exploration/hdl/TensorExtractor.vhd"
    qpf_filename = "WinBuffer-Exploration/quartus/TensorExtractor.qpf"
    fit_filename = "WinBuffer-Exploration/quartus/output_files/TensorExtractor.fit.summary"
    csv_filename = "Results/Exploration/WinBuffer.csv"

    # ExploreWinBuffer(vhdl_filename=vhdl_filename,
    #                  fit_filename=fit_filename,
    #                  qpf_filename=qpf_filename,
    #                  num_buff_max=20,
    #                  stride=1,
    #                  csv_filename=csv_filename)

    PlotData(csv_filename)
