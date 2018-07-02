import numpy as np
import AlteraUtils
import AnalyseWeights
from scipy import stats
import os
os.environ['GLOG_minloglevel'] = '4'
import caffe
import matplotlib.pyplot as plt
from matplotlib import rcParams
params = {
   'axes.labelsize': 8,
   'font.size': 8,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'figure.figsize': [4.5, 4.5],
   'axes.facecolor' : 'white'
   }
rcParams.update(params)


def dispMetaData(caffe_net, layer, bitwidth):
    print("=================================")
    print("Model:" + caffe_net)
    print("Layer:" + layer)
    print("Bitwidth:" + str(bitwidth))
    print("=================================")

def linearRegs(X, nb_alm):
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,0], nb_alm)
    print("\tnb_null\t%.4f " % (r_value**2))
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,1], nb_alm)
    print("\tnb_pow2\t%.4f " % (r_value**2))
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,2], nb_alm)
    print("\tnb_efbw\t%.4f " % (r_value**2))
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,3], nb_alm)
    print("\tnb_bit1\t%.4f " % (r_value**2))
    slope, intercept, r_value, p_value, std_err = stats.linregress(X[:,2], nb_alm)
    return (slope, intercept)

def genLinearReg(X, nb_alm):
    from sklearn import linear_model
    lm = linear_model.LinearRegression()
    lm.fit(X, nb_alm)
    print("\tGLMs\t%.4f \n" % (lm.score(X, nb_alm)))
    print("\n")

def ordinaryLeastSquares(X, nb_alm):
    import statsmodels.api as sm
    model = sm.OLS(nb_alm, X)
    results = model.fit()
    print(results.summary(xname=['Null', 'Pow2', 'Efbw', 'Bit1']))
    return model

def plot3D(x,y,z, label_x, label_y, label_z="Resource (ALMs)"):
    # 3D Plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z)
    plt.show()

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

    dotprod_inst = ";       |DotProduct:"
    alm = AlteraUtils.getALM(fit_rpt_filename, dotprod_inst)
    dotprod_alm = (np.array(list(alm.items()))[:,1])

    te_inst = ";       |TensorExtractor:TensorExtractor_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, te_inst)
    te_alm = (np.array(list(alm.items()))[:,1])

    total = moa_alm + scm_alm + te_alm
    print("\nResources utilization by entity")
    print("\tSCM:\t%d (%.2f %%)" % (scm_alm, 100*scm_alm/total))
    print("\tMOA:\t%d (%.2f %%)" % (moa_alm, 100*moa_alm/total))
    print("\tTE :\t%d (%.2f %%)" % (te_alm,  100*te_alm/total))
    print("\n")
    s = fit_rpt_filename.split('/')[-1] #Remove path
    t = s.split('.')[0] # Remove extention
    u = t.split('-')[0:-1] # Remove Bitwidth
    title = u[0] + " " + u[1]
    save_path = fit_rpt_filename.split('.')[0] + ".pdf"
    # plt.figure()
    # plt.hist(dotprod_alm/np.max(dotprod_alm), bins='auto', normed=True)
    # plt.grid()
    # plt.title(title)
    # plt.savefig(save_path)

def modelMOA(conv, bias, bitwidth, fit_rpt_filename):
    # Builds a model of the hardware cost of Multi-Operand Adders
    nb_null, nb_ngtv, nb_pow2, nb_bit1  = AnalyseWeights.kernelStats(conv, bitwidth)
    nb_efbw = AnalyseWeights.ppBitwidth(conv, bias, bitwidth)
    where_full_null = AnalyseWeights.whereNull(conv, bitwidth)
    instance_name = ";          |MOA:MOA_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]
    nb_alm = np.insert(nb_alm, 0, where_full_null)
    X = np.array([nb_null, nb_pow2, nb_efbw, nb_bit1], dtype=float)
    X = X.T

    print("MOA Model: R-squared")
    linearRegs(X, nb_alm)
    genLinearReg(X, nb_alm)
    #ordinaryLeastSquares(X, nb_alm)

def modelSCM(conv, bias, bitwidth, fit_rpt_filename):
    # Builds a model of the hardware cost of Single Constant Multiplier
    nb_null, nb_ngtv, nb_pow2, nb_bit1  = AnalyseWeights.kernelStats(conv, bitwidth)
    where_full_null = AnalyseWeights.whereNull(conv, bitwidth)
    nb_efbw = AnalyseWeights.ppBitwidth(conv, bias, bitwidth)
    X = np.array([nb_null, nb_pow2, nb_efbw, nb_bit1], dtype=float)
    X = X.T
    instance_name = ";          |MCM:MCM_i|"
    alm = AlteraUtils.getALM(fit_rpt_filename, instance_name)
    nb_alm = np.array(list(alm.items()))[:,1]
    nb_alm = np.insert(nb_alm, 0, where_full_null)
    print("SCM Model: R-squared")
    linearRegs(X, nb_alm)
    genLinearReg(X, nb_alm)
    #ordinaryLeastSquares(X, nb_alm)

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

def main():
    network_names = ['alexnet', 'squeezenet', 'alexnet_compressed', 'vgg16']
    # network_names = ['alexnet', 'squeezenet', 'alexnet_compressed']
    model_root = "C:/Users/Kamel/Seafile/CNN-Models/"
    bitwidth = 6
    for network_name in network_names:
        if ('vgg' in network_name):
            layer_names =  ['conv1_1', 'conv1_2', 'conv2_1']
        else:
            layer_names = ['conv1']
        for layer_name in layer_names:
            proto_file = model_root + network_name + ".prototxt"
            model_file = model_root + network_name + ".caffemodel"
            fit_rpt_filename =  "Results/"
            fit_rpt_filename +=  network_name
            fit_rpt_filename +=  "-" + layer_name
            fit_rpt_filename +=  "-" + str(bitwidth) + "bits.txt"
            dispMetaData(network_name, layer_name, bitwidth)
            net = caffe.Net(proto_file,model_file,caffe.TEST)
            conv = net.params[layer_name][0].data
            bias = net.params[layer_name][1].data
            AnalyseWeights.kernelStatsTotal(conv, bitwidth)
            resourceByEntity(fit_rpt_filename)
            modelMOA(conv=conv,
                     bias=bias,
                     bitwidth=bitwidth,
                     fit_rpt_filename=fit_rpt_filename)
            modelSCM(conv=conv,
                     bias=bias,
                     bitwidth=bitwidth,
                     fit_rpt_filename=fit_rpt_filename)

if __name__ == '__main__':
    main()
