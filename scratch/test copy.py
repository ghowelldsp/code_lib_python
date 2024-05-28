import scipy.io as io
import numpy as np

testDict = {}

mat = io.loadmat('scratch/01_ALB_IMP_DEQ_reformatted_lp', testDict)


freq = mat['impDataLumpParams']['measImpData']

# np.savez('scratch/01_ALB_IMP_DEQ_reformatted', f=freq[:,0], Z=cpxImped[:,0])

from pymatreader import read_mat

data = read_mat('scratch/01_ALB_IMP_DEQ_reformatted_lp.mat')['impDataLumpParams']

x = 1
