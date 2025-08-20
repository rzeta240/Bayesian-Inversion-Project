import numpy
import matplotlib.pyplot as plt

n = 100000

x_0 = numpy.random.normal(loc = 36, scale = 0.4, size = n)
v = numpy.random.normal(loc = 5, scale = 0.2, size = n)

def mapping(a, b):
    result = []

    if len(a) == len(b):
        for i in range(len(a)):
            result.append(a[i] + 2 * b[i])
    
    return result

plt.hist(mapping(x_0,v))
plt.show()