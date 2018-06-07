import numpy as np
import AlteraUtils
import Models
import matplotlib.pyplot as plt
import caffe

def NumberOfOnes(num,bitwidth):
    bin_rep = np.binary_repr(int(num), width=bitwidth)
    return len([c for c in bin_rep if c =='1'])

def DistanceOfOnes(bin_rep):
    pos = []
    iter = 0
    for d in bin_rep:
        iter += 1
        if d == '1' : pos.append(iter)
    return pos

def isPowerTwo(num):
    return ((num & (num - 1)) == 0)

def nbPowerTwo(kernel):
    nb_pow_two = 0
    for c in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            for k in range(kernel.shape[2]):
                nb_pow_two += int(isPowerTwo(kernel[c,j,k]))
    return nb_pow_two

def nbOnes(kernel,bitwidth):
    nb_ones = 0
    for c in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            for k in range(kernel.shape[2]):
                nb_ones += NumberOfOnes(kernel[c,j,k], bitwidth)
    return nb_ones

def nbNull(kernel):
    nb_null = 0
    for c in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            for k in range(kernel.shape[2]):
                nb_null += (kernel[c,j,k] == 0)
    return nb_null

if __name__ == '__main__':
    project_root = ""
    proto_file = "example/caffe/lenet.prototxt"
    model_file = "example/caffe/lenet.caffemodel"
    bitwidth = 6

    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv = net.params['conv2'][0].data
    conv = Models.quantizeWeight(conv,bitwidth)

    nb_pow_two = np.zeros(conv.shape[0], dtype=int)
    nb_ones = np.zeros(conv.shape[0], dtype=int)
    nb_null = np.zeros(conv.shape[0], dtype=int)
    for n in range(conv.shape[0]):
        nb_pow_two[n] = nbPowerTwo(conv[n,:])
        nb_ones[n] = nbOnes(conv[n,:],bitwidth)
        nb_null[n] = nbNull(conv[n,:])
    # Pourcentage
    nb_pow_two = 100* nb_pow_two / (conv.shape[1]*conv.shape[2]*conv.shape[3])
    nb_null = 100* nb_null / (conv.shape[1]*conv.shape[2]*conv.shape[3])

    # Read number of ALMs
    # instance_name = ";       |DotProduct:"
    instance_name = ";          |MCM:MCM_i|"
    # instance_name = ";          |MOA:MOA_i|"
    fit_rpt_filename = "example/quartus/FittingConv2.txt"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    np_alm = np.array(list(alm.items()))[:,1]  # Dict -> Numpy array

    plt.scatter(nb_null,np_alm,marker='^')
    plt.scatter(nb_pow_two,np_alm,marker='o')
    plt.show()
