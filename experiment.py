'''
Created on May 21, 2014

@author: jason
'''


from kagglecode import AMS
import numpy
from matplotlib import pyplot

# alpha = 1.0
# beta = 1.0
params = [(1.0,1.0),
          (1.0,5.0),
          (5.0,1.0),
          (0.1,1.0),
          (1.0,0.1)]
n = 1000000
T = numpy.arange(0,1.0,.01)
# T = 0.5

# def generate(alpha, beta, n):
#     
#     return p, y
# 
# def guess(T, p):
#     
#     return y_hat

def simulate(alpha, beta, n, T):
    p = numpy.random.beta(alpha, beta, n)
    y = numpy.random.binomial(1, p) == 1
    result = []
    for t in T:
        y_hat = p >= t
        b = numpy.sum(y_hat & (~y))
        s = numpy.sum(y_hat & y)
        ams = AMS(s,b)
        result.append(ams)
    return result

# print AMS(s,b)

pyplot.figure()
for alpha, beta in params:
    pyplot.plot(T, simulate(alpha, beta, n, T), label='a=%f, b=%f' % (alpha, beta))
#     print simulate(alpha, beta, n, T)
pyplot.legend(loc=0,prop={'size':8})
pyplot.show()

