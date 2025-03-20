# Plot ML,mb,Ms versus r7 if r7 is variable.
# ML = 0.67*I0 + 1
# mb = 0.48*I0 + 1.66
# Ms = -1.1 + 0.62*Ii + 1.3*10e-3*ri + 1.62*log10(ri)
import numpy as np
import matplotlib.pyplot as plt
from math import log10
i7 = 7 
r7 = 20 # uint: km
mL = 0.67 * i7 + 1
mb = 0.48 * i7 + 1.66
Ms = round((-1.1 + 0.62*i7 + 1.3*10e-3*r7 + 1.62*log10(r7)), 4)
print('The answer of problem 3(a) is:')
print('ML = ', mL)
print('mb = ', mb)
print('Ms = ', Ms)

# plot ML,mb,Ms versus r7
ri = np.linspace(1, 100)
def get_magnetude(r):
    mL = 0.67 * i7 + 1
    mb = 0.48 * i7 + 1.66
    Ms = -1.1 + 0.62*i7 + 1.3*10e-3*r + 1.62*log10(r)
    return mL, mb, Ms

mL = []
mb = []
Ms = []
for r in ri:
    mL_, mb_, Ms_ = get_magnetude(r)
    mL.append(mL_)
    mb.append(mb_)
    Ms.append(Ms_)

plt.figure()
plt.plot(ri, mL, label = 'ML', color = 'blue', linestyle = '-',linewidth = 2)
plt.plot(ri, mb, label = 'mb', color = 'green', linestyle = 'dotted',linewidth = 2)
plt.plot(ri, Ms, label = 'Ms', color = 'red', linestyle = '--',linewidth = 2)
plt.xlabel('r7(km)')
plt.ylabel('ML, mb, Ms')
plt.ylim(0, 10)
plt.title('ML, mb, Ms versus r7')
plt.legend(loc='lower right')
plt.show()
