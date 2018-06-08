import numpy as np
from scipy import stats
import AlteraUtils
import caffe
from AnalyseWeights import *
from sklearn import linear_model
import matplotlib.pyplot as plt

def modelMOA():
    # Modeling on Alexnet Data
    network_name = "alexnet"
    layer_name = 'conv1'
    bitwidth = 6
    proto_file = "example/caffe/" + network_name + "_conv1.prototxt"
    model_file = "C:/Users/Kamel/Seafile/CNN-Models/" + network_name + ".caffemodel"
    fit_rpt_filename = "Results/"
    fit_rpt_filename += network_name
    fit_rpt_filename += "_" + layer_name
    fit_rpt_filename += "_" + str(bitwidth) + "bits.txt"
    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params[layer_name][0].data
    nb_null, nb_negative, nb_pow_two, nb_bit_one  = kernelStats(conv, bitwidth)
    instance_name = ";          |MOA:MOA_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]
    ## Removing irrational costs
    nb_alm[91] = 1170
    nb_alm[93] = 730

    # Propotion to actual number (nb_null -> nb_opd)
    nb_opd = (100 - nb_null) * conv[1].size/100
    nb_bit_one = nb_bit_one * conv[1].size/100 * bitwidth

    ## 3D Plot
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(nb_opd, nb_bit_one, nb_alm)
    # plt.show()
    X = np.array([nb_opd, nb_bit_one], dtype=float)
    X = X.T

    # Multiple Linear Regression
    clf = linear_model.LinearRegression()
    clf.fit(X, nb_alm)
    coefs = clf.coef_

    # plt.scatter(nb_bit_one, nb_alm, marker='o')
    # plt.plot(nb_bit_one, slope*(nb_bit_one) + intercept, color='red')
    # plt.show()

    ## Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_bit_one, nb_alm)
    print( "Modeled MOA with N_bits1 LR, %s = %.4f " % ("r²", r_value**2))
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_opd, nb_alm)
    print( "Modeled MOA with N_opd LR, %s = %.4f " % ("r²", r_value**2))
    print( "Modeled MOA with CLR, %s = %.4f " % ("r²", clf.score(X, nb_alm)))

def testMOA():
    ## Testing on Lenet Data
    network_name = "lenet"
    layer_name = 'conv2'
    proto_file = "example/caffe/" + network_name + ".prototxt"
    model_file = "example/caffe/" + network_name + ".caffemodel"
    fit_rpt_filename = "Results/"
    fit_rpt_filename += network_name
    fit_rpt_filename += "_" + layer_name
    fit_rpt_filename += "_" + str(bitwidth) + "bits.txt"
    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params[layer_name][0].data
    nb_null, nb_negative, nb_pow_two, nb_bit_one  = kernelStats(conv, bitwidth)
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]
    nb_opd = (100 - nb_null) * conv[1].size/100 # Propotion to actual number
    model_alm = slope * nb_opd + intercept
    err = (np.sum(model_alm) - np.sum(nb_alm))/np.sum(nb_alm)
    print(err)
    plt.figure(1)
    plt.scatter(nb_opd, nb_alm, marker='o', color='blue')
    plt.scatter(nb_opd, model_alm, marker='o', color='red')
    plt.axis([0, 160, 0, 1000])
    plt.show()

def modelSCM():
    # Modeling on Alexnet Data
    network_name = "alexnet"
    layer_name = 'conv1'
    bitwidth = 6
    proto_file = "example/caffe/" + network_name + "_conv1.prototxt"
    model_file = "C:/Users/Kamel/Seafile/CNN-Models/" + network_name + ".caffemodel"
    fit_rpt_filename = "Results/"
    fit_rpt_filename += network_name
    fit_rpt_filename += "_" + layer_name
    fit_rpt_filename += "_" + str(bitwidth) + "bits.txt"
    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params[layer_name][0].data
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
    print('=============================================================')
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow, nb_alm)
    print( "Modeled MCM with nb_pow LR, %s = %.4f " % ("r²", r_value**2))

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_bit_one, nb_alm)
    print( "Modeled MOA with nb_bit_one LR, %s = %.4f " % ("r²", r_value**2))

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_null, nb_alm)
    print( "Modeled MOA with nb_null LR, %s = %.4f " % ("r²", r_value**2))

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_pow+nb_null, nb_alm)
    print( "Modeled MOA with nb_pow+nb_null LR, %s = %.4f " % ("r²", r_value**2))

    X = np.array([nb_bit_one, nb_null, nb_pow], dtype=float)
    X = X.T

    # Multiple Linear Regression
    clf = linear_model.LinearRegression()
    clf.fit(X, nb_alm)
    coefs = clf.coef_
    print( "Modeled MOA with CLR, %s = %.4f " % ("r²", clf.score(X, nb_alm)))
    print('=============================================================')

def testMOA():
    ## Testing on Lenet Data
    network_name = "lenet"
    layer_name = 'conv2'
    proto_file = "example/caffe/" + network_name + ".prototxt"
    model_file = "example/caffe/" + network_name + ".caffemodel"
    fit_rpt_filename = "Results/"
    fit_rpt_filename += network_name
    fit_rpt_filename += "_" + layer_name
    fit_rpt_filename += "_" + str(bitwidth) + "bits.txt"
    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params[layer_name][0].data
    nb_null, nb_negative, nb_pow_two, nb_bit_one  = kernelStats(conv, bitwidth)
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]
    nb_opd = (100 - nb_null) * conv[1].size/100 # Propotion to actual number
    model_alm = slope * nb_opd + intercept
    err = (np.sum(model_alm) - np.sum(nb_alm))/np.sum(nb_alm)
    print(err)
    plt.figure(1)
    plt.scatter(nb_opd, nb_alm, marker='o', color='blue')
    plt.scatter(nb_opd, model_alm, marker='o', color='red')
    plt.axis([0, 160, 0, 1000])
    plt.show()


def costMOA(n_opd):
    pass

def costSCM(multiplicand,lut):
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
    modelSCM()
