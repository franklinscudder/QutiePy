import qutiepy as qu
import numpy as np
from math import gcd

"""
This script uses Shor's algorithm to factor the number 15 with x = 11.

Taken from:
https://arxiv.org/pdf/1804.03719.pdf

Implementation by T. Findlay, 7/21
"""

if __name__ == "__main__":
    
    N = 15
    M = 2 ** 3
    x = 11
    
    reg = qu.Register(5)
    had1 = qu.ParallelGate([qu.Identity(2), qu.Hadamard(3)])
    cnot1 = qu.ParallelGate([qu.Identity(1), qu.CNot(), qu.Hadamard(2)])
    cnot2 = qu.ParallelGate([qu.PauliX(1).add_control_bits([2]), qu.Identity(2)])
    had2 = qu.ParallelGate([qu.Identity(3), qu.Hadamard(1), qu.Identity(1)])
    phs1 = qu.ParallelGate([qu.Identity(3), qu.Phase(1, np.pi / 2).add_control_bits([-1])])
    had3 = qu.ParallelGate([qu.Identity(4), qu.Hadamard(1)])
    phs2 = qu.ParallelGate([qu.Identity(2), qu.Phase(1, np.pi / 4).add_control_bits([1]), qu.Identity(1)])
    phs3 = qu.ParallelGate([qu.Identity(2), qu.Phase(1, np.pi / 2).add_control_bits([2])])
    
    all_gates = [had1, cnot1, cnot2, had2, phs1, had3, phs2, phs3]
    
    shor = qu.serial_gate(all_gates)
    
    res = shor(reg)
    
    p = 0
    while p == 0:
        out = res.observe(collapseStates=False)
        p = out % 8
        
    print("Period Found:", p)
    
    # Since M = 8, we can conclude that r divides M/p = 8/4 = 2, hence r = 2
    r = M / p
    
    """
    Then 15 divides
    (x^r - 1) = (11^2 - 1) = (11 - 1)(11 + 1) = 10 * 12
    """
    
    xp = x ** p
    y1, y2 = xp - 1, xp + 1
    factors = gcd(y1, N), gcd(y2, N)
    
    print("Factors of 15 are:", factors[0], factors[1])
