import scipy.io as io
import numpy as np

testDict = {}

mat = io.loadmat('scratch/01_ALB_IMP_DEQ_reformatted', testDict)

freq = mat['ImpedanceMagnitude'][3,0]
mag = mat['ImpedanceMagnitude'][3,1]
phase = mat['ImpedancePhase'][3,1]

cpxImped = mag * np.exp(1j * (phase * np.pi/180.0))

np.savez('scratch/01_ALB_IMP_DEQ_reformatted', f=freq[:,0], Z=cpxImped[:,0])

x = 1
