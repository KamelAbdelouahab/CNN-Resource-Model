import numpy as np
from scipy import stats
import AlteraUtils
from AnalyseWeights import *
from sklearn import linear_model
import matplotlib.pyplot as plt
from contextlib import contextmanager
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe

def parseLayerParam(network_name, layer_name, bitwidth):
    proto_file = "example/caffe/" + network_name + "_conv1.prototxt"
    model_file = "C:/Users/Kamel/Seafile/CNN-Models/" + network_name + ".caffemodel"
    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params[layer_name][0].data
    return conv

def parseLayerResource(network_name, layer_name, bitwidth):
    fit_rpt_filename = "Results/"
    fit_rpt_filename += network_name
    fit_rpt_filename += "_" + layer_name
    fit_rpt_filename += "_" + str(bitwidth) + "bits.txt"
    return fit_rpt_filename

def paramCorrelation():
    network_name = "alexnet"
    layer_name = 'conv1'
    bitwidth = 6

    conv = parseLayerParam(network_name, layer_name, bitwidth)
    fit_rpt_filename = parseLayerResource(network_name, layer_name, bitwidth)

    nb_null, nb_negative, nb_pow_two, nb_bit_one  = kernelStats(conv, bitwidth)
    nb_opd = (100 - nb_null) * conv[1].size/100
    nb_null = nb_null * conv[1].size/100
    nb_pow = nb_pow_two * conv[1].size/100
    nb_bit_one = nb_bit_one * conv[1].size/100 * bitwidth

    ## Correlation between metrics
    print('\nCorrelation between metrics:=================================')
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow, nb_bit_one)
    rho, pval = stats.spearmanr(nb_pow, nb_bit_one)
    print("Correlation nb_pow2 and nb_bit1, r2 = %.4f , p = %.4f" % (r_value**2, rho))

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_null, nb_bit_one)
    rho, pval = stats.spearmanr(nb_null, nb_bit_one)
    print("Correlation nb_null and nb_bit1, r2 = %.4f , p = %.4f" % (r_value**2, rho))


def modelMOA():
    # Modeling on Alexnet Data
    network_name = "alexnet"
    layer_name = 'conv1'
    bitwidth = 6

    conv = parseLayerParam(network_name, layer_name, bitwidth)
    fit_rpt_filename = parseLayerResource(network_name, layer_name, bitwidth)

    nb_null, nb_negative, nb_pow_two, nb_bit_one  = kernelStats(conv, bitwidth)
    instance_name = ";          |MOA:MOA_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]

    ## Removing irrational costs
    nb_alm[91] = 1170
    nb_alm[93] = 730

    ## Propotion to actual number (nb_null -> nb_opd)
    nb_opd = (100 - nb_null) * conv[1].size/100
    nb_null = nb_null * conv[1].size/100
    nb_pow = nb_pow_two * conv[1].size/100
    nb_bit_one = nb_bit_one * conv[1].size/100 * bitwidth

    X = np.array([nb_null, nb_pow], dtype=float)
    X = X.T

    ## Multiple Linear Regression
    clf = linear_model.LinearRegression()
    clf.fit(X, nb_alm)
    coefs = clf.coef_
    Y = clf.predict(X)

    # plt.scatter(nb_bit_one, nb_alm, marker='o')
    # plt.plot(nb_bit_one, slope*(nb_bit_one) + intercept, color='red')
    # plt.show()

    ## Linear regression
    print('\nMOA Model ===================================================')
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow, nb_alm)
    rho, pval = stats.spearmanr(nb_pow, nb_alm)
    print( "Modeled MOA with nb_pow2 LR, %s = %.4f, p = %.4f" % ("r²", r_value**2, rho))

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_bit_one, nb_alm)
    rho, pval = stats.spearmanr(nb_bit_one, nb_alm)
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
    ax.scatter(nb_null, nb_pow, nb_alm)
    ax.set_xlabel('Number of Nulls/Kernel')
    ax.set_ylabel('Number of Pow2/Kernel')
    ax.set_zlabel('Resource (ALMs)')
    plt.show()
    return clf

def modelSCM():
    # Modeling on Alexnet Data
    network_name = "alexnet"
    layer_name = 'conv1'
    bitwidth = 6
    conv = parseLayerParam(network_name, layer_name, bitwidth)
    fit_rpt_filename = parseLayerResource(network_name, layer_name, bitwidth)
    nb_null, nb_negative, nb_pow_two, nb_bit_one  = kernelStats(conv, bitwidth)

    instance_name = ";          |MCM:MCM_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]

    ## X_train
    nb_null = nb_null * conv[1].size/100
    nb_pow = nb_pow_two * conv[1].size/100
    nb_bit_one = nb_bit_one * conv[1].size/100 * bitwidth
    ## Removing irrational costs
    removeShit(nb_alm)

    # plt.figure(0)
    # plt.scatter(nb_pow + nb_null, nb_alm, marker='o')
    # plt.show()
    # plt.plot(nb_bit_one, slope*(nb_bit_one) + intercept, color='red')
    # plt.show()

    ## Linear regression
    print('\nSCM Model ===================================================')
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow, nb_alm)
    rho, pval = stats.spearmanr(nb_pow, nb_alm)
    print( "Modeled SCM with nb_pow2 LR, %s = %.4f , %s = %.4f" % ("r²", r_value**2, "p", rho))

    rho, pval = stats.spearmanr(nb_bit_one, nb_alm)
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_bit_one, nb_alm)
    print( "Modeled SCM with nb_bit1 LR, %s = %.4f , %s = %.4f" % ("r²", r_value**2, "p", rho))

    rho, pval = stats.spearmanr(nb_null, nb_alm)
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_null, nb_alm)
    print( "Modeled SCM with nb_null LR, %s = %.4f , %s = %.4f" % ("r²", r_value**2, "p", rho))

    rho, pval = stats.spearmanr(nb_pow+nb_null, nb_alm)
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow+nb_null, nb_alm)
    print( "Modeled SCM with nb_pow2 + nb_null LR, %s = %.4f , %s = %.4f" % ("r²", r_value**2, "p", rho))

    X = np.array([nb_bit_one, nb_null, nb_pow], dtype=float)
    X = X.T

    ## Multiple Linear Regression
    clf = linear_model.LinearRegression()
    clf.fit(X, nb_alm)
    coefs = clf.coef_
    Y = clf.predict(X)
    rho, pval = stats.spearmanr(Y, nb_alm)
    print( "Modeled SCM with CLR, %s = %.4f , p = %.4f" % ("r²", clf.score(X, nb_alm), rho))
    print('=============================================================')

    ## 3D Plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(nb_null, nb_pow, nb_alm)
    ax.set_xlabel('Number of Nulls/Kernel')
    ax.set_ylabel('Number of Pow2/Kernel')
    ax.set_zlabel('Resource (ALMs)')
    plt.show()
    return clf

def resourceByEntity(fit_rpt_filename= "Results/alexnet_conv1_6bits.txt"):
    scm_inst = ";          |MCM:MCM_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, scm_inst)
    scm_alm = np.sum(np.array(list(alm.items()))[:,1])

    moa_inst = ";          |MOA:MOA_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, moa_inst)
    moa_alm = np.sum(np.array(list(alm.items()))[:,1])

    total = moa_alm + scm_alm

    print("\nResources utilization by entity ============================")
    print("Single Constant Multipliers (SCMs):\t %d \t %.2f %%" % (scm_alm, 100*scm_alm/total))
    print("Multi Operand Adders (MOAs):\t\t %d \t %.2f %%" % (moa_alm, 100*moa_alm/total))

def costMOA(kernel):
    pass

def costSCM(kernel):
    pass

def costDotProduct(kernel):
    pass

def costActivation():
    pass

def costTensorExtractor(image_width, kernel_size):
    pass

def costConv(kernel_list,bitwidth):
    pass

if __name__ == '__main__':
    paramCorrelation()
    resourceByEntity()
    modelMOA()
    modelSCM()
    # resourceByEntity("Results\lenet_conv2_6bits.txt")
