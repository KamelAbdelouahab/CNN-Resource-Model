import numpy as np
import os

def J_moa(n_opd):
    coef = 3.23636364 #Generated with GenModels
    return float(coef * n_opd)

def J_scm(multiplicand,lut):
    return int(lut[lut[:,0]==multiplicand,1]) # More efficient?

def J_dp(kernel):
    n_opds = np.count_nonzero(kernel)
    j_moa = J_moa(n_opds)

    j_scm = 0
    lut = np.loadtxt("SCM8bits.csv", delimiter=',')
    for i in range(kernel.shape[0]):
        j_scm += J_scm(kernel[i],lut)
    return (j_scm,j_moa)

def main():
    pass

if __name__ == '__main__':
    k = np.array([1, 2, 3, 4, 5, 6, 0, 0, 2])
    (scm,moa) = J_dp(k)
    print(scm,moa)
