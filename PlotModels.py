import numpy as np
from scipy import stats
import AlteraUtils
import caffe
from AnalyseWeights import *
from scipy import stats
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


if __name__ == '__main__':
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
    nb_alm = removeShit(nb_alm)
    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_bit_one, nb_alm)
    # print( "%s : %.2f " % ("r²", r_value**2))

    plt.figure(0)
    plt.grid()
    plt.scatter(nb_bit_one, nb_alm,
                marker='o',
                label="Constant\nMultiplication\nBloc")
    plt.plot(np.sort(nb_bit_one),
            slope*np.sort(nb_bit_one) + intercept,
            linewidth=2,
            color='#B22400')
    legend = plt.legend(loc=2)
    plt.xlabel('Number of bits set to 1 per 3D kernel (%)')
    plt.ylabel('Logic Resources (ALMs)')
    plt.text(10, 40, "r$^2$ = 0.68", fontsize=15, color='red')
    plt.axis([0, 60, 0, 700])
    #plt.show()
    plt.savefig('Results/Jscm.pdf', bbox_inches='tight')


    plt.figure(1)
    instance_name = ";          |MOA:MOA_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]
    nb_alm[91] = 1170
    nb_alm[93] = 730

    slope, intercept, r_value, p_value, std_err = stats.linregress(nb_null, nb_alm)
    # print( "%s : %.2f " % ("r²", r_value**2))


    plt.grid()
    plt.scatter(nb_null,
                nb_alm,
                marker='o',
                label="Multi Operand\nAdder Bloc")
    plt.plot(np.sort(nb_null),
             slope*np.sort(nb_null) + intercept,
             linewidth=2,
             color='#B22400')
    legend = plt.legend(loc=1)
    plt.xlabel('Number of null values per 3D kernel(%)')
    plt.ylabel('Logic Resources (ALMs)')
    plt.text(40, 250, "r$^2$ = 0.82", fontsize=15, color='red')
    plt.axis([0, 80, 0, 2000])
    #plt.show()
    plt.savefig('Results/Jmoa.pdf', bbox_inches='tight')
