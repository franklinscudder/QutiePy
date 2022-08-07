import sys
sys.path.insert(0,'..')    ### TEMPORARY!!!!!

from qutiepy import *
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
    
    reg = Register(5)
    had1 = ParallelGate([Identity(2), Hadamard(3)])
    cnot1 = ParallelGate([Identity(1), CNot(), Hadamard(2)])
    cnot2 = ParallelGate([PauliX(1).add_control_bits([2]), Identity(2)])
    had2 = ParallelGate([Identity(3), Hadamard(1), Identity(1)])
    phs1 = ParallelGate([Identity(3), Phase(1, np.pi/2).add_control_bits([-1])])
    had3 = ParallelGate([Identity(4), Hadamard(1)])
    phs2 = ParallelGate([Identity(2), Phase(1, np.pi/4).add_control_bits([1]), Identity(1)])
    phs3 = ParallelGate([Identity(2), Phase(1, np.pi/2).add_control_bits([2])])
    
    all_gates = [had1, cnot1, cnot2, had2, phs1, had3, phs2, phs3]
    
    shor = serial_gate(all_gates)
    
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
    (x^r − 1) = (11^2 − 1) = (11 − 1)(11 + 1) = 10 * 12
    """
    
    xp = x ** p
    y1 ,y2 = xp - 1, xp + 1
    factors = gcd(y1, N), gcd(y2, N) 
    
    print("Factors of 15 are:", factors[0], factors[1])
    
    