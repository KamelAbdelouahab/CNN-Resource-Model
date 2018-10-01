import numpy as np
import AlteraUtils
import Models
import matplotlib.pyplot as plt
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe

def quantizeWeight(float_weight,bitwidth):
    scale_factor = 2**(bitwidth-1) - 1
    scaled_data = np.round(float_weight * scale_factor)
    return np.array(scaled_data, dtype=int)

def histogram(weights, bitwidth):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    params = {
       'axes.labelsize': 10,
       'font.size': 10,
       'legend.fontsize': 12,
       'xtick.labelsize': 12,
       'ytick.labelsize': 12,
       'figure.figsize': [4.5, 4.5],
       'axes.facecolor' : 'white'
       }
    rcParams.update(params)

    weights = quantizeWeight(weights,bitwidth)
    weights = weights/(2**(bitwidth-1) - 1)
    plt.hist(weights.flatten(), bins=200)
    #plt.title("Histogram of weights. Quantized with %d Bits" %(bitwidth))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.axis([-0.3, 0.3, 0, 6000])
    plt.grid()
    plt.show()

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

def kernelStatsNetwork(cnn,bitwidth):
    nb_null, nb_ngtv, nb_pow2, nb_bit1 = 0,0,0,0
    num_params_total = 0
    params = cnn.params
    blobs  = cnn.blobs
    # Print CNN shape
    for l in cnn._layer_names:
        layerId = list(cnn._layer_names).index(l)
        layerType =  cnn.layers[layerId].type
        if (layerType == 'Convolution'):
            conv_layer = params[l][0].data
            conv_layer = quantizeWeight(conv_layer,bitwidth)
            num_params_total +=  conv_layer.size
            print("---------------")
            for n in range(conv_layer.shape[0]):
                nb_null += nbNull(conv_layer[n,:])
                nb_pow2 += nbPow2(conv_layer[n,:])

    nb_null = 100* nb_null / num_params_total
    nb_pow2 = 100* nb_pow2 / num_params_total
    print('Kernel Stats')
    print( "%s\t%d "     % ("\tWeights", num_params_total))
    print( "%s\t%.2f %%" % ("\tnb_null", nb_null))
    print( "%s\t%.2f %%" % ("\tnb_pow2", nb_pow2))


    return nb_null, nb_pow2


def kernelStatsTotal(conv_layer,bitwidth):
    nb_null, nb_ngtv, nb_pow2, nb_bit1 = 0,0,0,0
    conv_layer = quantizeWeight(conv_layer,bitwidth)
    for n in range(conv_layer.shape[0]):
        nb_null += nbNull(conv_layer[n,:])
        nb_ngtv += nbNgtv(conv_layer[n,:])
        nb_pow2 += nbPow2(conv_layer[n,:])
        nb_bit1 += nbBit1(conv_layer[n,:], bitwidth)
    nb_null = 100* nb_null / conv_layer.size
    nb_pow2 = 100* nb_pow2 / conv_layer.size
    nb_ngtv = 100* nb_ngtv / conv_layer.size
    nb_bit1 = 100* nb_bit1 / (conv_layer.size*bitwidth)
    print('Kernel Stats')
    print( "%s\t%d "     % ("\tWeights", conv_layer.size))
    print( "%s\t%.2f %%" % ("\tnb_null", nb_null))
    print( "%s\t%.2f %%" % ("\tnb_pow2", nb_pow2))
    #print( "%s\t%.2f %%" % ("\tnb_ngtv", nb_ngtv))
    #print( "%s\t%.2f %%" % ("\tnb_bit1", nb_bit1))
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

def isNull(kernel):
    return (np.count_nonzero(kernel) == 0)

def whereNull(conv,bitwidth):
    conv = quantizeWeight(conv,bitwidth)
    l = []
    for n in range(conv.shape[0]):
        if(isNull(conv[n,:])):
            l.append(n)
    return l

def ppBitwidth(conv, bias, bitwidth):
    # Returns the number of bits at the output of each mult
    conv = quantizeWeight(conv,bitwidth)
    bias = quantizeWeight(bias,bitwidth)
    sum_pp_bitwidth = np.zeros(conv.shape[0])
    for n in range (sum_pp_bitwidth.shape[0]):
        kernel = conv[n]
        bitwidth_extension = np.zeros(kernel.shape)
        bitwidth_extension [kernel != 0] = bitwidth
        bitwidth_extension [kernel != 0] += np.round(np.log2(np.abs(kernel[kernel != 0])))
        sum_pp_bitwidth[n] =  np.sum(bitwidth_extension)
        if(bias[n]!=0):
            sum_pp_bitwidth[n] += np.round(np.log2(np.abs(bias[n])))
    return sum_pp_bitwidth


if __name__ == '__main__':
    bitwidth = 8
    proto_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet.prototxt"
    model_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet.caffemodel"
    layer_name = 'conv1'
    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv_layer = net.params[layer_name][0].data
    kernelStatsTotal(conv_layer,bitwidth)
    histogram(conv_layer, bitwidth)

    proto_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet_compressed.prototxt"
    model_file = "C:/Users/Kamel/Seafile/CNN-Models/alexnet_compressed.caffemodel"
    net = caffe.Net(proto_file,model_file,caffe.TEST)
    conv_layer = net.params[layer_name][0].data
    kernelStatsTotal(conv_layer,bitwidth)
    histogram(conv_layer, bitwidth)
