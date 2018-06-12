import numpy as np
import AlteraUtils
import Models
import matplotlib.pyplot as plt
import caffe

def quantizeWeight(float_weight,bitwidth):
    scale_factor = 2**(bitwidth-1) - 1
    scaled_data = np.round(float_weight * scale_factor)
    return np.array(scaled_data, dtype=int)

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
    return num !=0 and ((num & (num - 1)) == 0)

def nbNgtv(kernel):
    nb_ngtv = 0
    for c in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            for k in range(kernel.shape[2]):
                nb_ngtv += (kernel[c,j,k] < 0)
    return nb_ngtv

def nbPow2(kernel):
    nb_pow2 = 0
    for c in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            for k in range(kernel.shape[2]):
                nb_pow2 += int(isPowerTwo(kernel[c,j,k]))
    return nb_pow2

def nbBit1(kernel,bitwidth):
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

def kernelStatsTotal(conv_layer,bitwidth):
    nb_null, nb_ngtv, nb_pow2, nb_bit1 = 0,0,0,0
    conv_layer = quantizeWeight(conv_layer,bitwidth)
    for n in range(conv_layer.shape[0]):
        nb_null += nbNull(conv_layer[n,:])
        nb_ngtv += nbNgtv(conv_layer[n,:])
        nb_pow2 += nbPow2(conv_layer[n,:])
        nb_bit1 += nbBit1(conv_layer[n,:], bitwidth)

    nb_null = 100* nb_null / conv_layer.size
    nb_ngtv = 100* nb_ngtv / conv_layer.size
    nb_pow2 = 100* nb_pow2 / conv_layer.size
    nb_bit1 = 100* nb_bit1 / (conv_layer.size*bitwidth)
    print('=============================================================')
    print( "%s: %d " % ("Number of weights", conv_layer.size))
    print( "%s:\t %.2f %%" % ("Pourcentage of Null Kernels", nb_null))
    print( "%s:\t %.2f %%" % ("Pourcentage of Power2 Kernels", nb_pow2))
    print( "%s: %.2f %%" % ("Pourcentage of Negative Kernels", nb_ngtv))
    print( "%s:\t\t %.2f %%" % ("Pourcentage of Bits=1", nb_bit1))
    return nb_null, nb_ngtv, nb_pow2, nb_bit1

def kernelStats(conv_layer,bitwidth):
    nb_pow2 = np.zeros(conv_layer.shape[0], dtype=int)
    nb_ngtv = np.zeros(conv_layer.shape[0], dtype=int)
    nb_bit1 = np.zeros(conv_layer.shape[0], dtype=int)
    nb_null = np.zeros(conv_layer.shape[0], dtype=int)

    conv_layer = quantizeWeight(conv_layer,bitwidth)
    for n in range(conv_layer.shape[0]):
        nb_null[n] = nbNull(conv_layer[n,:])
        nb_pow2[n] = nbPow2(conv_layer[n,:])
        nb_ngtv[n] = nbNgtv(conv_layer[n,:])
        nb_bit1[n] = nbBit1(conv_layer[n,:], bitwidth)

    return nb_null, nb_ngtv, nb_pow2, nb_bit1

def removeShit(nb_alm):
    nb_alm[1] = 410
    nb_alm[87] = 420
    nb_alm[91] = 280
    nb_alm[93] = 380
    return nb_alm

def isNull(kernel):
    return (np.count_nonzero(kernel) == 0)

def whereNull(conv,bitwidth):
    conv = quantizeWeight(conv,bitwidth)
    l = []
    for n in range(conv.shape[0]):
        if(isNull(conv[n,:])):
            l.append(n)
    return l

if __name__ == '__main__':
    proto_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet.prototxt"
    model_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet.caffemodel"
    layer_name = 'conv1'
    bitwidth = 6

    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv_layer = net.params[layer_name][0].data
    print('=======================================')
    print("Model:" + model_file)
    print("Layer:" + layer_name)
    print("Bitwidth:" + str(bitwidth))
    nb_null, nb_ngtv, nb_pow2 , nb_ones= kernelStatsTotal(conv_layer, bitwidth)
    # nb_null, nb_ngtv, nb_pow2,nb_bit1  = kernelStats(conv_layer, bitwidth)
    # alm = 0.9 * np.random.rand(conv_layer.shape[0])
    # plt.scatter(nb_pow2,alm)
    # plt.axis([0, 100, 0, 1])
    # plt.show()
