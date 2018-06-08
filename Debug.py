import numpy as np
import AlteraUtils
import Models
import matplotlib.pyplot as plt
import caffe
from AnalyseWeights import *

if __name__ == '__main__':
    network_name = "alexnet"
    layer_name = 'conv1'
    bitwidth = 6

    proto_file = "example/caffe/" + network_name + "_conv1.prototxt"
    model_file = "C:/Users/Kamel/Seafile/CNN-Models/" + network_name + ".caffemodel"
    fit_rpt_filename = "example/"
    fit_rpt_filename += network_name
    fit_rpt_filename += "_" + layer_name
    fit_rpt_filename += "_" + str(bitwidth) + "bits.txt"
    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params[layer_name][0].data

    nb_null, nb_negative, nb_pow_two, nb_bit_one  = kernelStats(conv, bitwidth)

    instance_name = ";          |MCM:MCM_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    np_alm = np.array(list(alm.items()))[:,1]
    A = np.vstack([nb_bit_one, np.ones(len(nb_bit_one))]).T
    m0, m1 = np.linalg.lstsq(A, np_alm)[0]
    plt.scatter(nb_bit_one,np_alm,marker='o')
    plt.plot(nb_bit_one, m0*nb_bit_one + m1)
    plt.axis([0, 80, 0, 500])
    plt.show()

    instance_name = ";          |MOA:MOA_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    np_alm = np.array(list(alm.items()))[:,1]
    A = np.vstack([nb_null, np.ones(len(nb_null))]).T
    m0, m1 = np.linalg.lstsq(A, np_alm)[0]
    plt.scatter(nb_null, np_alm,marker='o')
    plt.plot(nb_null, m0*nb_null + m1)
    plt.axis([0, 80, 0, 2000])
    plt.show()

    #instance_name = ";       |DotProduct"
