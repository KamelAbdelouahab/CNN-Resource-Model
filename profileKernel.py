import numpy as np
from scipy import stats
from sklearn import linear_model
import matplotlib.pyplot as plt
import os

import AlteraUtils
import AnalyseWeights
import Models

os.environ['GLOG_minloglevel'] = '2'
import caffe

if __name__ == '__main__':
    proto_file = "/home/kamel/Seafile/CNN-Models/squeezenet.prototxt"
    model_file = "/home/kamel/Seafile/CNN-Models/squeezenet.caffemodel"
    layer_name = 'conv1'
    fit_rpt_filename = "Results/alexnet_conv1_6bits.txt"
    bitwidth = 6

    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params[layer_name][0].data

    print("Model:" + model_file)
    print("Layer:" + layer_name)
    print("Bitwidth:" + str(bitwidth))

    #Models.resourceByEntity(fit_rpt_filename)
    #AnalyseWeights.kernelStatsNetwork(net, bitwidth)
    AnalyseWeights.kernelStatsTotal(conv, bitwidth)
    #Models.profileKernel(conv, fit_rpt_filename, bitwidth)
