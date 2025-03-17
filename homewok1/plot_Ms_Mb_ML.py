# Plot ML,mb,Ms versus r7 if r7 is variable.
# ML = 0.67*I0 + 1
# mb = 0.48*I0 + 1.66
# Ms = -1.1 + 0.62*Ii + 1.3*10e-3*ri + 1.62*log10(ri)
import numpy as np
import matplotlib.pyplot as plt
from math import log10
i7 = 68 # unit: gal
r7 = 20 # uint: km
mL = 0.67 * i7 + 1
mb = 0.48 * i7 + 1.66
Ms = -1.1 + 0.62*i7 + 1.3*10e-3*r7 + 1.62*log10(r7)

print('The answer of problem 3(a) is:')
print('ML = ', mL)
print('mb = ', mb)
print('Ms = ', Ms)
# plot ML,mb,Ms versus r7
ri = np.linspace(1, 50, 0.5)
for i in ri:
    mL = 0.67 * 68 + 1
    mb = 0.48 * 68 + 1.66
    Ms = -1.1 + 0.62*68 + 1.3*10e-3*i + 1.62*log10(i)
    
