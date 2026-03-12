import numpy as np
import matplotlib.pyplot as plt

a = 6.91
c = 19.26
dz = 1.62

veca = np.array([a, 0, 0])
vecb = np.array([-np.cos(np.pi/3)*a, np.sin(np.pi/3)*a, 0])
vecc = np.array([0, 0, c])

cr1 = 1/3*veca + 2/3*vecb
cr2 = 2/3*veca + 1/3*vecb

It1 = 0.318*veca + 0.0*vecb + 0.60*vecc
It2 = 0.0*veca + 0.318*vecb + 0.60*vecc
It3 = 0.682*veca + 0.682*vecb + 0.60*vecc

Ib1 = 0.682*veca + 0.0*vecb + 0.4*vecc
Ib2 = 0.0*veca + 0.682*vecb + 0.4*vecc
Ib3 = 0.318*veca + 0.318*vecb + 0.4*vecc

print('cr1: ', cr1)
print('cr2: ', cr2)

plt.plot(0, 0, 'k.')
plt.arrow(0, 0, veca[0], veca[1], head_width=0.5, head_length=0.5, fc='k', ec='k')
plt.arrow(0, 0, vecb[0], vecb[1], head_width=0.5, head_length=0.5, fc='k', ec='k')
# plt.arrow(0, 0, vecc[0], vecc[1], head_width=0.5, head_length=0.5, fc='k', ec='k')
for n1 in range(-3, 3):
    for n2 in range(-3, 3):
        cr1p = cr1 + n1*veca + n2*vecb + 0*vecc
        cr2p = cr2 + n1*veca + n2*vecb + 0*vecc

        o = n1*veca + n2*vecb + 0*vecc

        plt.plot(o[0], o[1], 'k.')
        plt.arrow(o[0], o[1], veca[0], veca[1], head_width=0.1, head_length=0.1, fc='k', ec='k')
        plt.arrow(o[0], o[1], vecb[0], vecb[1], head_width=0.1, head_length=0.1, fc='k', ec='k')

        plt.plot(cr1p[0], cr1p[1], 'ro')
        plt.plot(cr2p[0], cr2p[1], 'ro')

        # Iodine atoms
        It1p = It1 + n1*veca + n2*vecb + 0*vecc
        It2p = It2 + n1*veca + n2*vecb + 0*vecc
        It3p = It3 + n1*veca + n2*vecb + 0*vecc

        Ib1p = Ib1 + n1*veca + n2*vecb + 0*vecc
        Ib2p = Ib2 + n1*veca + n2*vecb + 0*vecc
        Ib3p = Ib3 + n1*veca + n2*vecb + 0*vecc

        plt.plot(It1p[0], It1p[1], 'go')
        plt.plot(It2p[0], It2p[1], 'go')
        plt.plot(It3p[0], It3p[1], 'go')
        # plt.plot(Ib1p[0], Ib1p[1], 'bo')
        # plt.plot(Ib2p[0], Ib2p[1], 'bo')
        # plt.plot(Ib3p[0], Ib3p[1], 'bo')

# plt.plot(cr1[0], cr1[1], 'ro')
# plt.plot(cr2[0], cr2[1], 'ro')

plt.xlim(-30, 30)
plt.ylim(-30, 30)
plt.show()

z = [24, 24, 53, 53, 53, 53, 53, 53]*4
print(z)

batch = [0]*8 + [1]*8 + [2]*8 + [3]*8
print(batch)
