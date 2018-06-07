import numpy as np
import AlteraUtils
import Models
import matplotlib.pyplot as plt
import caffe
from AnalyseWeights import *

if __name__ == '__main__':
    project_root = ""
    proto_file = "example/caffe/lenet.prototxt"
    model_file = "example/caffe/lenet.caffemodel"
    bitwidth = 6

    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params['conv2'][0].data
    conv = Models.quantizeWeight(conv,bitwidth)

    nb_pow_two = np.zeros(conv.shape[0], dtype=int)
    nb_ones = np.zeros(conv.shape[0], dtype=int)
    nb_null = np.zeros(conv.shape[0], dtype=int)
    for n in range(conv.shape[0]):
        nb_pow_two[n] = nbPowerTwo(conv[n,:])
        nb_ones[n] = nbOnes(conv[n,:],bitwidth)
        nb_null[n] = nbNull(conv[n,:])
    # Pourcentage
    nb_pow_two = 100* nb_pow_two / (conv.shape[1]*conv.shape[2]*conv.shape[3])
    nb_null = 100* nb_null / (conv.shape[1]*conv.shape[2]*conv.shape[3])
    nb_ones = 100* nb_ones / (conv.shape[1]*conv.shape[2]*conv.shape[3]*6)

    fit_rpt_filename = "example/quartus/lenet5_conv2_6bits.txt"

    # instance_name = ";          |MCM:MCM_i|"
    # alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    # np_alm = np.array(list(alm.items()))[:,1]  # Dict -> Numpy array
    # plt.scatter(nb_pow_two,np_alm,marker='o')
    #
    # instance_name = ";          |MOA:MOA_i|"
    # alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    # np_alm = np.array(list(alm.items()))[:,1]
    # plt.scatter(nb_null,np_alm,marker='^')

    instance_name = ";       |DotProduct"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    np_alm = np.array(list(alm.items()))[:,1]
    plt.scatter(nb_ones,np_alm,marker='o')

    plt.axis([0, 80, 0, 1200])
    plt.show()
