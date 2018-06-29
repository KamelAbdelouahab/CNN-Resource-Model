import numpy as np
from scipy import stats
import AlteraUtils
import AnalyseWeights
from sklearn import linear_model
from matplotlib import rcParams
params = {
   'axes.labelsize': 8,
   'text.fontsize': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [4.5, 4.5],
   'axes.facecolor' : 'white'
   }
rcParams.update(params)
import matplotlib.pyplot as plt
from contextlib import contextmanager
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe

def paramCorrelation(conv, bitwidth, fit_rpt_filename):
    # Analyse the correlation between the metrics used to model the hardware
    nb_null, nb_ngtv, nb_pow2, nb_bit1  = AnalyseWeights.kernelStats(conv, bitwidth)
    ## Correlation between metrics
    print('\nCorrelation between metrics:=================================')
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow2, nb_bit1)
    rho, pval = stats.spearmanr(nb_pow2, nb_bit1)
    print("Correlation nb_pow2 and nb_bit1, r2 = %.4f , p = %.4f" % (r_value**2, rho))

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_null, nb_bit1)
    rho, pval = stats.spearmanr(nb_null, nb_bit1)
    print("Correlation nb_null and nb_bit1, r2 = %.4f , p = %.4f" % (r_value**2, rho))

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_null, nb_pow2)
    rho, pval = stats.spearmanr(nb_null, nb_pow2)
    print("Correlation nb_null and nb_pow2, r2 = %.4f , p = %.4f" % (r_value**2, rho))

def resourceByEntity(fit_rpt_filename):
    # Displays the hardware resources allocated to each entity and plots their histogram
    scm_inst = ";          |MCM:MCM_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, scm_inst)
    scm_alm = np.sum(np.array(list(alm.items()))[:,1])

    moa_inst = ";          |MOA:MOA_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, moa_inst)
    moa_alm = np.sum(np.array(list(alm.items()))[:,1])
    total = moa_alm + scm_alm

    dotprod_inst = ";       |DotProduct:"
    alm = AlteraUtils.getALM(fit_rpt_filename, dotprod_inst)
    dotprod_alm = (np.array(list(alm.items()))[:,1])

    print("\nResources utilization by entity ============================")
    print("Single Constant Multipliers (SCMs):\t %d \t %.2f %%" % (scm_alm, 100*scm_alm/total))
    print("Multi Operand Adders (MOAs):\t\t %d \t %.2f %%" % (moa_alm, 100*moa_alm/total))
    print("DotProduct Range:\t\t\t %d \t %d" % (np.min(dotprod_alm), np.max(dotprod_alm)))
    s = fit_rpt_filename.split('/')[-1] #Remove path
    t = s.split('.')[0] # Remove extention
    u = t.split('_')[0:-1] # Remove Bitwidth
    title = u[0] + " " + u[1]
    save_path = fit_rpt_filename.split('.')[0] + ".pdf"
    # plt.hist(dotprod_alm, bins='auto')
    # plt.grid()
    # plt.title(title)
    # plt.show()
    # ##plt.savefig(save_path)

def modelMOA(conv, bitwidth, fit_rpt_filename):
    # Builds a model of the hardware cost of Multi-Operand Adders
    nb_null, nb_ngtv, nb_pow2, nb_bit1  = AnalyseWeights.kernelStats(conv, bitwidth)
    where_full_null = AnalyseWeights.whereNull(conv, bitwidth)
    instance_name = ";          |MOA:MOA_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]
    nb_alm = np.insert(nb_alm, 0, where_full_null)

    ## Removing irrational costs
    # nb_alm[91] = 1170
    # nb_alm[93] = 730

    X = np.array([nb_null, nb_pow2], dtype=float)
    X = X.T

    ## Multiple Linear Regression
    clf = linear_model.LinearRegression()
    clf.fit(X, nb_alm)
    coefs = clf.coef_
    Y = clf.predict(X)

    # plt.scatter(nb_bit1, nb_alm, marker='o')
    # plt.plot(nb_bit1, slope*(nb_bit1) + intercept, color='red')
    # plt.show()

    ## Linear regression
    print('\nMOA Model ===================================================')
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow2, nb_alm)
    rho, pval = stats.spearmanr(nb_pow2, nb_alm)
    print( "Modeled MOA with nb_pow2 LR, %s = %.4f, p = %.4f" % ("r²", r_value**2, rho))

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_bit1, nb_alm)
    rho, pval = stats.spearmanr(nb_bit1, nb_alm)
    print( "Modeled MOA with nb_bit1 LR, %s = %.4f, p = %.4f" % ("r²", r_value**2, rho))

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_null, nb_alm)
    rho, pval = stats.spearmanr(nb_null, nb_alm)
    print( "Modeled MOA with nb_null LR, %s = %.4f, p = %.4f" % ("r²", r_value**2, rho))

    rho, pval = stats.spearmanr(Y, nb_alm)
    print( "Modeled MOA with CLR, %s = %.4f, p = %.4f" % ("r²", clf.score(X, nb_alm), rho))

    ## 3D Plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(nb_null, nb_pow2, nb_alm)
    ax.set_xlabel('Number of Nulls/Kernel')
    ax.set_ylabel('Number of Pow2/Kernel')
    ax.set_zlabel('Resource (ALMs)')
    plt.show()

def modelSCM(conv, bitwidth, fit_rpt_filename):
    # Builds a model of the hardware cost of Single Constant Multiplier
    nb_null, nb_ngtv, nb_pow2, nb_bit1  = AnalyseWeights.kernelStats(conv, bitwidth)
    where_full_null = AnalyseWeights.whereNull(conv, bitwidth)
    instance_name = ";          |MCM:MCM_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]
    nb_alm = np.insert(nb_alm, 0, where_full_null)

    ## Removing irrational costs
    # AnalyseWeights.removeShit(nb_alm)

    ## Linear regression
    print('\nSCM Model ===================================================')
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow2, nb_alm)
    rho, pval = stats.spearmanr(nb_pow2, nb_alm)
    print( "Modeled SCM with nb_pow2 LR, %s = %.4f , %s = %.4f" % ("r²", r_value**2, "p", rho))

    rho, pval = stats.spearmanr(nb_bit1, nb_alm)
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_bit1, nb_alm)
    print( "Modeled SCM with nb_bit1 LR, %s = %.4f , %s = %.4f" % ("r²", r_value**2, "p", rho))

    rho, pval = stats.spearmanr(nb_null, nb_alm)
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_null, nb_alm)
    print( "Modeled SCM with nb_null LR, %s = %.4f , %s = %.4f" % ("r²", r_value**2, "p", rho))

    rho, pval = stats.spearmanr(nb_pow2+nb_null, nb_alm)
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow2+nb_null, nb_alm)
    print( "Modeled SCM with nb_pow2 + nb_null LR, %s = %.4f , %s = %.4f" % ("r²", r_value**2, "p", rho))

    X = np.array([nb_bit1, nb_pow2], dtype=float)
    X = X.T

    ## Multiple Linear Regression
    clf = linear_model.LinearRegression()
    clf.fit(X, nb_alm)
    Y = clf.predict(X)
    rho, pval = stats.spearmanr(Y, nb_alm)
    print( "Modeled SCM with CLR, %s = %.4f , p = %.4f" % ("r²", clf.score(X, nb_alm), rho))
    print('=============================================================')

    ## 3D Plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(nb_bit1, nb_pow2, nb_alm)
    #ax.set_xlabel('Number of Nulls/Kernel')
    ax.set_xlabel('Number of Bit1/Kernel')
    ax.set_ylabel('Number of Pow2/Kernel')
    ax.set_zlabel('Resource (ALMs)')
    plt.show()

def profileKernel(conv_layer, fit_rpt_filename, bitwidth):
    # Display the value of the less "resource-intensive" kernels
    instance_name = ";       |DotProduct:"
    alm_dict = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    alm = (np.array(list(alm_dict.items()))[:,1])
    tresh_low = 0.1 * np.max(alm)
    tresh_high = 0.9 * np.max(alm)
    print("Cheap Kernels:")
    for n in range(conv_layer.shape[0]):
        if (alm[n] < tresh_low):
            print(AnalyseWeights.quantizeWeight(conv_layer[n], bitwidth))
    print("Expensive Kernels:")
    for n in range(conv_layer.shape[0]):
        if (alm[n] > tresh_high):
            print(AnalyseWeights.quantizeWeight(conv_layer[n], bitwidth))

if __name__ == '__main__':
    proto_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet.prototxt"
    model_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet.caffemodel"
    layer_name = 'conv1'
    fit_rpt_filename = "Results/alexnet_conv1_6bits.txt"
    bitwidth = 6

    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params[layer_name][0].data

    print("Model:" + model_file)
    print("Layer:" + layer_name)
    print("Bitwidth:" + str(bitwidth))

    AnalyseWeights.kernelStatsTotal(conv, bitwidth)
    resourceByEntity(fit_rpt_filename)
    paramCorrelation(conv, bitwidth, fit_rpt_filename)
    modelMOA(conv, bitwidth, fit_rpt_filename)
    modelSCM(conv, bitwidth, fit_rpt_filename)
