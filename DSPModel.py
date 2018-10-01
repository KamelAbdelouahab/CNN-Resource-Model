import numpy as np
import AnalyseWeights
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe

import matplotlib.pyplot as plt
from matplotlib import rcParams

def DSPModel(proto,weight,layer_name,bitwidth):
    net = caffe.Net(proto,weight,caffe.TEST)
    theta = net.params[layer_name][0].data
    theta = AnalyseWeights.quantizeWeight(theta,bitwidth)
    nb_multipliers = np.count_nonzero(theta)
    return nb_multipliers

if __name__ == '__main__':
    proto  = "/home/kamel/Seafile/CNN-Mappings/Haddoc2/Lenets/I2/caffe/I2.prototxt"
    weight = "/home/kamel/Seafile/CNN-Mappings/Haddoc2/Lenets/I2/caffe/I2.caffemodel"
    bitwidth = 6
    mult1 = DSPModel(proto,weight,'conv1',bitwidth)
    mult2 = DSPModel(proto,weight,'conv2',bitwidth)
    #mult3 = DSPModel(proto,weight,'conv3',bitwidth)
    print("nb_mult1 %d" %mult1)
    print("nb_mult2 %d" %mult2)
    #print("nb_mult3 %d" %mult3)
    #print("nb_total %d" %(mult1 + mult2 + mult3))
