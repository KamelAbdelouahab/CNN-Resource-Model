import numpy as np
import AlteraUtils
import AnalyseWeights
#from scipy import stats
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe

import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
    'grid.color' : 'k',
    'grid.linestyle': 'dashdot',
    'grid.linewidth': 0.6,
    'font.family': 'Linux Biolinum O',
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.facecolor' : 'white'
   }
rcParams.update(params)

reports =  [
          'Results/alexnet-conv1-6bits.txt',
          'Results/deepcomp-conv1-6bits.txt',
          # 'Results/lenet-conv1-6bits.txt',
          # 'Results/lenet-conv2-6bits.txt',
          'Results/squeezenet-conv1-6bits.txt',
          'Results/vgg16-conv1_1-6bits.txt',
          'Results/vgg16-conv1_2-6bits.txt',
          'Results/vgg16-conv2_1-6bits.txt',
          'Results/yolov3tiny-conv1-6bits.txt'
          ]

cnn_inst = "; |cnn_process"
scm_inst = ";          |MCM:MCM_i|"
moa_inst = ";          |MOA:MOA_i|"
buf_inst = ";       |TensorExtractor:TensorExtractor_i|"

#print(AlteraUtils.getALM(reports[1], cnn_inst)[0])
for r in reports:
    cnn_alm = int(AlteraUtils.getALM(r, cnn_inst)[0])
    scm_alm = np.sum(AlteraUtils.getALM(r, scm_inst), dtype=int)
    moa_alm = np.sum(AlteraUtils.getALM(r, moa_inst), dtype=int)
    buf_alm = np.sum(AlteraUtils.getALM(r, buf_inst), dtype=int)
    alm = [scm_alm, 100*scm_alm/cnn_alm, moa_alm, 100*moa_alm/cnn_alm, buf_alm, 100*moa_alm/cnn_alm, cnn_alm]
    #print(alm)
    print("%s \t& %d (%1.2f) \t& %d (%1.2f) \t& %d (%1.2f)"
          %(r,  scm_alm, 100*scm_alm/cnn_alm, moa_alm, 100*moa_alm/cnn_alm, buf_alm, 100*buf_alm/cnn_alm))
