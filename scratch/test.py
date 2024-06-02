import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

testDict = {}

# mat = io.loadmat('01_ALB_IMP_DEQ_reformatted', testDict)

# freq = mat['ImpedanceMagnitude'][3,0]
# mag = mat['ImpedanceMagnitude'][3,1]
# phase = mat['ImpedancePhase'][3,1]

# cpxImped = mag * np.exp(1j * (phase * np.pi/180.0))

# np.savez('01_ALB_IMP_DEQ', freq=freq[:,0], mag=mag[:,0], phase=phase[:,0])

# x = 1

impedanceData1 = np.load(f'01_ALB_IMP_DEQ_impedData.npy', allow_pickle=True)
fVec1 = impedanceData1.item().get('fVec')
Himp1 = impedanceData1.item().get('Himp')

# ref
impedanceData2 = np.load(f'01_ALB_IMP_DEQ_reformatted.npz', allow_pickle=True)
fVec2 = impedanceData2['f']
Himp2 = impedanceData2['Z']

plt.figure()
plt.plot(fVec1, np.abs(Himp1))
plt.plot(fVec2, np.abs(Himp2), '--')
plt.show()
