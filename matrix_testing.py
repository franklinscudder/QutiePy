
import numpy as np
import sys
from scipy.linalg import block_diag

sys.path.insert(0, "..")

from qutiepy_new_naming_convention import *

# swap on LSBs controlled by MSB

inputs =  [0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111, 0b1000, 0b1001, 0b1010, 0b1011, 0b1100, 0b1101, 0b1110, 0b1111]
#inputs = [int(i, 2) for i in inputs]

targets = [0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111, 0b1000, 0b1010, 0b1001, 0b1011, 0b1100, 0b1110, 0b1101, 0b1111]
#targets = [int(t, 2) for t in targets]

L = np.eye(16)
U = np.kron(np.eye(4), Swap().matrix)
          
print(L,"\n\n", U)

M = block_diag(L, U)

print(M.shape)

for n, inp in enumerate(inputs):
    state_vector = np.array([int(i == inp) for i in range(len(inputs))])
    
    result = M * state_vector
    
    print(f"Target: {targets[n]}")
    print(f"Result: {result}\n")
