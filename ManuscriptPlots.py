from sklearn import linear_model
from scipy import stats
import Models as ModelDHM
import AlteraUtils
import numpy as np
import AlteraUtils
import AnalyseWeights
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
    'grid.color': 'k',
    'grid.linestyle': 'dashdot',
    'grid.linewidth': 0.6,
    'font.family': 'Linux Biolinum O',
    'font.size': 18,
    'axes.facecolor': 'white'
}
rcParams.update(params)

def main():
    network_name = 'alexnet'
    layer_name   = 'conv1'
    model_root = "/home/kamel/Seafile/CNN-Models/"
    bitwidth = 6
    [conv, bias]     = ModelDHM.importLayerParams(network_name, layer_name, bitwidth)
    fit_rpt_filename = ModelDHM.importFitReport(network_name, layer_name, bitwidth)
    X = ModelDHM.generateFeatures(conv, bias, bitwidth)
    instance_name = ";          |MOA:MOA_i|"
    y = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    y = ModelDHM.correctEntityALM(conv, y, bitwidth)
    lr = linear_model.LinearRegression()
    x = X[:,0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    ModelDHM.plotLinearModel(x, y, slope,  'MOA', r_value**2, "./MOA-ZERO.pdf", intercept)


if __name__ == '__main__':
    main()
