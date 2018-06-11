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
    fit_rpt_filename = Models.parseLayerResource(network_name, layer_name, bitwidth)
    nb_null, nb_negative, nb_pow_two, nb_bit_one  = AnalyseWeights.kernelStats(conv, bitwidth)

    ## SCM Model
    instance_name = ";          |MCM:MCM_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]

    # Propotion to actual number (nb_null -> nb_opd)
    nb_null = nb_null * conv[1].size/100
    nb_pow = nb_pow_two * conv[1].size/100
    nb_bit_one = nb_bit_one * conv[1].size/100 * bitwidth

    X = np.array([nb_bit_one, nb_null, nb_pow], dtype=float)
    X = X.T
    model = Models.modelSCM()
    alm_model = model.predict(X)
    alm_true = nb_alm
    print(np.array(alm_true))
    print(np.array(alm_model))

if __name__ == '__main__':
    test()
