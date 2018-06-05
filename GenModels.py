import sys
import numpy as np
import matplotlib.pyplot as plt

def NumberOfOnes(bin_rep):
    return len([c for c in bin_rep if c =='1'])

def DistanceOfOnes(bin_rep):
    pos = []
    iter = 0
    for d in bin_rep:
        iter += 1
        if d == '1' : pos.append(iter)
    return pos


def GenModelSCM(csv_filename):
    lut = np.loadtxt(csv_filename, delimiter=',')
    alm = lut[:,1]
    num_ones = np.zeros(alm.shape[0])
    bitwidth = 8
    for i in range(lut.shape[0]):
        multiplicand = lut[i,0]
        bin_multiplicand = np.binary_repr(int(multiplicand), width=bitwidth)
        num_ones[i] = NumberOfOnes(bin_multiplicand)
    plt.scatter(num_ones,alm)
    plt.show()

def GenModelMOA(csv_filename):
    lut = np.loadtxt(csv_filename, delimiter=',')
    alm = lut[:,1]
    n_opd = lut[:,0]
    # Force 0 operands to have 0 ALMs
    # n_opd = np.insert(n_opd,0,0)
    # alm = np.insert(alm,0,0)

    # MMSE SOlution of linear matrix solution
    A = np.vstack([n_opd, np.ones(len(n_opd))]).T
    m0, m1 = np.linalg.lstsq(A, alm)[0]

    # Plot the MOA Model
    plt.scatter(n_opd,alm)
    plt.plot(n_opd, m0*n_opd)
    plt.show()

    # Return Coefs
    print (m0, m1)
    return (m0, m1)

if __name__ == '__main__':
    # dummy = "10110111"
    # print(DistanceOfOnes(dummy))
    #ModelSCM(csv_filename="SCM8bits.csv")
    GenModelMOA(csv_filename="MOA16bits.csv")
