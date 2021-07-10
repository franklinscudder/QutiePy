from qutiepy import *
from random import randint
from math import gcd, log2, ceil
from copy import deepcopy

########### Quantum Part #####################

def makeFxGate(a, q, N):
    matrix = np.zeros((2**q, 2**q))
    for inp in range(2**q):
        out = (a * inp) % N
        matrix[out, inp] += 1
    
    gate = genericGate(q)
    gate.matrix = matrix
    
    return gate

def shor(a, q, N):
    reg = register(q)
    had_q = hadamard(q)
    
    reg = had_q(reg)
    
    

############ Classical Part ##################

N = 3 * 5

Qmax = 2 * N * N
q = ceil(log2(Qmax))

a = randint(2, N-1)
K = gcd(a, N)

if K != 1:
    print("Done (gcd):", K)
    quit()

else:
    r = shor(a, q, N)
    makeFxGate(a, q, N)

