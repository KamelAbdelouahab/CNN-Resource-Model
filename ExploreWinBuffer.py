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
    win_buff_num = data[:, 0]
    alm_num = data[:, 1]
    ram_num = data[:, 2]

    plt.figure(figsize=(4, 3))
    plt.grid()
    plt.plot(win_buff_num, alm_num, 'b^-')
    plt.ylabel('ALMs')
    plt.savefig("Results/WinBuffALM.pdf", bbox_inches='tight')
    plt.figure(figsize=(4, 3))
    plt.plot(win_buff_num, ram_num, 'b^-')
    plt.grid()
    plt.ylabel('M20K')
    plt.savefig("Results/WinBuffRAM.pdf", bbox_inches='tight')


def WindowBufferNum(protoFile, modelFile):
    buffer_ori = np.array([])
    buffer_wbf = np.array([])
    layers = []
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format
    parsible_net = caffe_pb2.NetParameter()
    text_format.Merge(open(protoFile).read(), parsible_net)
    cnn = caffe.Net(protoFile, 1, weights=modelFile)
    params = cnn.params
    blobs = cnn.blobs
    bitwidth = 8

    for layer in parsible_net.layer:
        if (layer.type == 'Input'):
            l = layer.name
            H = blobs[l].data.shape[2]
            W = blobs[l].data.shape[3]

        if (layer.type == 'Convolution'):
            l = layer.name
            N = layer.convolution_param.num_output if len(
                layer.convolution_param.kernel_size) else 1
            C = params[l][0].data.shape[1]
            J = layer.convolution_param.kernel_size[0] if len(
                layer.convolution_param.kernel_size) else 1
            K = J
            p = layer.convolution_param.pad[0] if len(
                layer.convolution_param.pad) else 0
            s = layer.convolution_param.stride[0] if len(
                layer.convolution_param.stride) else 1
            U = blobs[l].data.shape[2]
            V = blobs[l].data.shape[3]
            H = blobs[l].data.shape[2]
            W = blobs[l].data.shape[3]
            buffer_ori = np.append(
                buffer_ori, (bitwidth * N * C * J * W - N * C * W + N * C * K) / 1000)
            buffer_wbf = np.append(
                buffer_wbf, (bitwidth * C * J * W - C * W + C * K) / 1000)
            layers.append(l)

        if (layer.type == 'Pooling'):
            l = layer.name
            N = blobs[l].data.shape[1]
            C = blobs[l].data.shape[1]
            K = layer.pooling_param.kernel_size
            p = layer.pooling_param.pad
            s = layer.pooling_param.stride
            U = blobs[l].data.shape[2]
            V = blobs[l].data.shape[3]
            H = blobs[l].data.shape[2]
            W = blobs[l].data.shape[3]

    return [layers, buffer_ori, buffer_wbf]


if __name__ == '__main__':
    [layers, buffer_ori, buffer_wbf] = WindowBufferNum('/home/kamel/Seafile/CNN-Models/alexnet.prototxt',
                                                       '/home/kamel/Seafile/CNN-Models/alexnet.caffemodel')
    width = 0.3
    layer_num = np.linspace(1, len(layers), len(layers), endpoint=True)
    plt.figure(figsize=(6, 4))
    plt.bar(layer_num - 0.5 * width, buffer_ori,
            width, edgecolor='k', color='g')
    plt.bar(layer_num + 0.5 * width, buffer_wbf,
            width, edgecolor='k', color='b')
    plt.xticks(layer_num, layers)
    plt.title('Memory allocated to Window Buffers (KBits)')
    plt.yscale('log', basey=10)
    plt.legend(['Original', 'Factorized'])
    plt.grid()
    plt.savefig("WBFRes.pdf", bbox_inches='tight')

    vhdl_filename = "WinBuffer-Exploration/hdl/TensorExtractor.vhd"
    qpf_filename = "WinBuffer-Exploration/quartus/TensorExtractor.qpf"
    fit_filename = "WinBuffer-Exploration/quartus/output_files/TensorExtractor.fit.summary"
    csv_filename = "Results/Exploration/WinBuffer.csv"

    ExploreWinBuffer(vhdl_filename=vhdl_filename,
                     fit_filename=fit_filename,
                     qpf_filename=qpf_filename,
                     num_buff_max=20,
                     stride=1,
                     csv_filename=csv_filename)

    PlotData(csv_filename)
