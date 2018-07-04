import numpy as np
import pandas as pd
import AlteraUtils
import AnalyseWeights
import Models
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe

def valModel(conv, bias, bitwidth, fit_rpt_filename, scm_coefs, moa_coefs):
    X = Models.generateFeatures(conv, bias, bitwidth)
    scm_alm = AlteraUtils.getALM(fit_rpt_filename, ";          |MCM:MCM_i|")
    moa_alm = AlteraUtils.getALM(fit_rpt_filename, ";          |MOA:MOA_i|" )
    scm_alm = Models.correctEntityALM(conv, scm_alm, bitwidth)
    moa_alm = Models.correctEntityALM(conv, moa_alm, bitwidth)
    scm_pred = np.dot(X,(scm_coefs.T))
    moa_pred = np.dot(X,(moa_coefs.T))

    print("True SCM = %.2f \t Predicted SCM = %.2f \t Delta = %.2f" %(np.sum(scm_alm),np.sum(scm_pred), 100*(np.sum(scm_alm)-np.sum(scm_pred))/np.sum(scm_alm)))
    print("True MOA = %.2f \t Predicted MOA = %.2f \t Delta = %.2f" %(np.sum(moa_alm),np.sum(moa_pred), 100*(np.sum(moa_alm)-np.sum(moa_pred))/np.sum(moa_alm)))

def testModel(conv, bias, bitwidth,  scm_coefs, moa_coefs):
    X = Models.generateFeatures(conv, bias, bitwidth)
    scm_pred = np.dot(X,(scm_coefs.T))
    moa_pred = np.dot(X,(moa_coefs.T))
    print("Predicted SCM = %.2f" %np.sum(scm_pred))
    print("Predicted MOA = %.2f" %np.sum(moa_pred))
    print("Predicted ALM = %.2f" %np.sum(scm_pred+moa_pred))


if __name__ == '__main__':
    # Modeling
    network_name = "alexnet"
    layer_name = "conv1"
    bitwidth = 6
    [conv, bias]   = Models.importLayerParams("vgg16", "conv2_1", 6)
    fit_rpt_filename = Models.importFitReport("vgg16", "conv2_1", 6)
    scm_coefs = Models.modelSCM(conv, bias, bitwidth, fit_rpt_filename)[0]
    moa_coefs = Models.modelMOA(conv, bias, bitwidth, fit_rpt_filename)[0]


    # Evaluation
    [conv, bias]   = Models.importLayerParams("vgg16", "conv1_2", 6)
    fit_rpt_filename = Models.importFitReport("vgg16", "conv1_2", 6)
    valModel(conv, bias, bitwidth, fit_rpt_filename, scm_coefs, moa_coefs)
