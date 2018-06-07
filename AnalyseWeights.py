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
    return num !=0 and ((num & (num - 1)) == 0)

def nbNegative(kernel):
    nb_neg = 0
    for c in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            for k in range(kernel.shape[2]):
                nb_neg += (kernel[c,j,k] < 0)
    return nb_neg

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

def main(conv_layer,bitwidth):
    nb_null, nb_negative, nb_pow_two = 0,0,0
    conv_layer = Models.quantizeWeight(conv_layer,bitwidth)
    for n in range(conv_layer.shape[0]):
        nb_null += nbNull(conv_layer[n,:])
        nb_negative += nbNegative(conv_layer[n,:])
        nb_pow_two += nbPowerTwo(conv_layer[n,:])

    nb_null = 100* nb_null / conv_layer.size
    nb_negative = 100* nb_negative / conv_layer.size
    nb_pow_two = 100* nb_pow_two / conv_layer.size

    return nb_null, nb_negative, nb_pow_two

if __name__ == '__main__':
    proto_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet.prototxt"
    model_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet.caffemodel"
    layer_name = 'conv2'
    bitwidth = 6

    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv_layer = net.params[layer_name][0].data
    nb_null, nb_negative, nb_pow_two  = main(conv_layer, bitwidth)
    print("Model:" + model_file)
    print("Layer:" + layer_name)
    print("Bitwidth:" + str(bitwidth))
    print('=======================================')
    print( "%s : %.2f " % ("Null Kernel", nb_null))
    print( "%s : %.2f " % ("Negative Kernels", nb_negative))
    print( "%s : %.2f " % ("Power2 Kernels", nb_pow_two))
