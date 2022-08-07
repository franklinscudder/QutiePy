from qutiepy import *
from random import randint
from math import gcd, log2, ceil
from copy import deepcopy

########### Quantum Part #####################

def makeFxGate(a, q, r, N):     # Performs (a * x) % N, where x.NBits = q
    matrix = np.zeros((2**q, 2**q))
    for inp in range(2**q):
        out = (a * inp) % N
        matrix[out, inp] += 1
    
    gate = genericGate(q)
    gate.matrix = matrix
    
    return gate

def makeFxCircuit(a, q, r, N):
    gates = []
    
    for i in range(q):
        gate = makeFxGate(a, q, r, N)
        
        exp = 2**i
        for j in range(exp - 1):
            gate = gate(gate)
        
        print(q - i)
        gate.addControlBits([q - i])
        _gates = [gate, identity(i)] if i else [gate]
        gate = parallelGate(_gates)
    
        gates.append(gate)
    
    print(q + r)
    print([g.NBits for g in gates])
    return serialGate(gates)

def shor(a, q, r, N):
    outReg = register(q)
    had_q = hadamard(q)
    
    outReg = had_q(outReg)
    
    inReg = register(r).setAmps([0, 1] + ([0]*((2**r)-2)))
    
    U = makeFxCircuit(a, q, r, N)
    
    qft = parallelGate([QFT(q), identity(r)])
    
    reg = prod(outReg, inReg)
    
    out = qft(U(reg))
    
    print(out.observe())
    
    

############ Classical Part ##################

N = 2 * 3
r = ceil(log2(N))

Qmax = 2 * N * N
q = ceil(log2(Qmax))

a = randint(2, N-1)
K = gcd(a, N)

if K != 1:
    print("Done (gcd):", K)
    quit()

else:
    res = shor(a, q, r, N)
    makeFxGate(a, q, r, N)

