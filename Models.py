import numpy as np
import os
import caffe
import AlteraUtils

def quantizeWeight(float_weight,bitwidth):
    scale_factor = 2**(bitwidth-1) - 1
    scaled_data = np.round(float_weight * scale_factor)
    return np.array(scaled_data, dtype=int)

def costMOA(n_opd):
    # coef = 3.23636364
    coef = [6.4, 1.2]
    return float(coef[0] * n_opd + coef[1])

def costSCM(multiplicand,lut):
    return float(lut[lut[:,0]==multiplicand,1])
    # print(multiplicand)
    # return 1

def costDotProduct(kernel):
    n_opds = np.count_nonzero(kernel)
    j_moa = costMOA(n_opds)

    j_scm = 0
    lut = np.loadtxt("SCM8bits.csv", delimiter=',')
    for i in range(kernel.shape[0]):
        j_scm += costSCM(kernel[i],lut)
    return (j_scm,j_moa)

def costActivation():
    pass

def costTensorExtractor(image_width, kernel_size):
    pass

def costConv(kernel_list,bitwidth):
    j_conv = np.zeros([kernel_list.shape[0],2],dtype=float)
    for n in range(kernel_list.shape[0]):
        j_conv[n,0:2] = costDotProduct(kernel_list[n].flatten())
    return(j_conv)

def trueConv(fit_rpt_filename):
    instance_name = ";          |MOA:MOA_i|"
    true_moa = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    instance_name = ";          |MCM:MCM_i|"
    true_scm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    true_conv = np.zeros([len(true_moa),2],dtype=float)
    true_conv[:,0] = np.fromiter(true_scm.values(), dtype=float)
    true_conv[:,1] = np.fromiter(true_moa.values(), dtype=float)
    return(true_conv)

if __name__ == '__main__':
    project_root = ""
    proto_file = "example/caffe/lenet.prototxt"
    model_file = "example/caffe/lenet.caffemodel"
    bitwidth = 8

    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv1 = net.params['conv1'][0].data
    conv1 = quantizeWeight(conv1,bitwidth)
    model = costConv(conv1,bitwidth)

    fit_rpt_filename = "example/quartus/FittingReport.txt"
    true_val = trueConv(fit_rpt_filename)
    # print(model)
    # print(true_val)
    print(100*model[:,1]/true_val[:,1])
    # AlteraUtils.dispSortedDict(true_moa)
    # AlteraUtils.dispSortedDict(true_scm)
