"""
A more general implementation of Shor's algorithm, factoring N.

UNDER DEVELOPMENT, NONFUNCTIONAL!!!

"""

import qutiepy as qu
import numpy as np
from random import randint
from math import gcd, log2, ceil

########### Quantum Part #####################


def makeFxGate(a, q, r, N):     # Performs (a * x) % N, where x.NBits = q
    matrix = np.zeros((2**q, 2**q))
    for inp in range(2**q):
        out = (a * inp) % N
        matrix[out, inp] += 1
    
    gate = qu.GenericGate(q)
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
        gate.add_control_bits([q - i])
        _gates = [gate, qu.Identity(i)] if i else [gate]
        gate = qu.ParallelGate(_gates)
    
        gates.append(gate)
    
    print(q + r)
    print([g.N_bits for g in gates])
    return qu.serial_gate(gates)


def shor(a, q, r, N):
    outReg = qu.Register(q)
    had_q = qu.Hadamard(q)
    
    outReg = had_q(outReg)
    
    inReg = qu.Register(r).set_amps([0, 1] + ([0] * ((2 ** r) - 2)))
    
    U = makeFxCircuit(a, q, r, N)
    
    qft = qu.ParallelGate([qu.QFT(q), qu.Identity(r)])
    
    reg = qu.prod(outReg, inReg)
    
    out = qft(U(reg))
    
    print(out.observe())
    

############ Classical Part ##################

N = 3 * 2
r = ceil(log2(N))

Qmax = 2 * N * N
q = ceil(log2(Qmax))

a = randint(2, N - 1)
K = gcd(a, N)

if K != 1:
    print("Done (gcd):", K)
    quit()

else:
    res = shor(a, q, r, N)
    print(res)
