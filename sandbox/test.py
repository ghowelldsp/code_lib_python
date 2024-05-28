import numpy as np
import scipy.signal as sig

bOrig = np.array([-2,2])
aOrig = np.array([1,3])

z, p, k = sig.tf2zpk(bOrig, aOrig)

print(z)
print(p)

b, a = sig.zpk2tf(z, p, k)

print(b)
print(a)

z1 = np.array([-1.0000000000000002])
p1 = np.array([-1.7140819937296339+0j])

b, a = sig.zpk2tf(z1, p1, k)

print(b)
print(a)

print(testFnc(2))

def testFnc(x):
    
    y = x + 2
    
    return y
