import numpy as np
from scipy import stats
import AlteraUtils
import AnalyseWeights
import Models
from sklearn import linear_model
import matplotlib.pyplot as plt
from contextlib import contextmanager
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe



def test():
    # Test on Lenet5 conv2 Data
    network_name = "lenet"
    layer_name = "conv2"
    proto_file = "example/caffe/lenet.prototxt"
    model_file = "example/caffe/lenet.caffemodel"
    bitwidth = 6
    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params[layer_name][0].data
    fit_rpt_filename = "Results\lenet_conv2_6bits.txt"
    nb_null, nb_ngtv, nb_pow2, nb_bit1  = AnalyseWeights.kernelStats(conv, bitwidth)

    ## SCM Model
    instance_name = ";          |MCM:MCM_i|" # From SCM Model
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    alm_true = np.array(list(alm.items()))[:,1]

    X = np.array([nb_bit1, nb_pow2], dtype=float).T
    coefs = np.array([0.46495855, -0.59831905])
    alm_model = X[:,0]*coefs[0] + X[:,1]*coefs[1]
    err = (alm_true - alm_model)/alm_model
    print(err)

    ## MOA Model
    instance_name = ";          |MOA:MOA_i|" # From MOA Model
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    alm_true = np.array(list(alm.items()))[:,1]

    X = np.array([nb_null, nb_pow2], dtype=float).T
    coefs = np.array([-6.87825333, -5.25932361])
    alm_model = X[:,0]*coefs[0] + X[:,1]*coefs[1]
    err = (alm_true - alm_model)/alm_model
    print(err)


if __name__ == '__main__':
    test()
