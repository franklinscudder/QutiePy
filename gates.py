"""
qutiepy.gates.py
==========================
The file containing the gate classes of QutiePy
"""

import numpy as np
import scipy.linalg as sp


class genericGate:
    """
    Base class for callable quantum logic gates.
    
    Parameters
    ----------
    NBits : int
        Size of the registers that the gate will take as input/output.

    Attributes
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
    
    matrix : 2D numpy array
        The 2^NBits by 2^NBits matrix representation of the gate.

    """
    def __init__(self, NBits):
        _checkNBits(NBits)
        self.NBits = NBits
        self.matrix = np.identity(2 ** NBits)
    
    def __call__(self, arg):
        if issubclass(type(arg), genericGate):
            out = genericGate(self.NBits + arg.NBits)
            out.matrix = np.matmul(self.matrix, arg.matrix)
            return out

        elif type(arg) == register:
            result = register(arg.NBits)
            result.amps = np.matmul(self.matrix, arg.amps)
            return result
        
        else:
            raise TypeError("Gates can only be called on gates or registers! Got type: " +  str(type(arg)))
    
    def __str__(self):
        stri = str(self.NBits) + "-bit " + type(self).__name__ + " Gate, Matrix:\n\r"
        stri = stri + self.matrix.__str__()
        return stri

class hadamard(genericGate):
    """ A callable hadamard gate object. 
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
    
    """
    def __init__(self, NBits):
        super(hadamard, self).__init__(NBits)
        self.matrix = sp.hadamard(2 ** NBits) * (2**(-0.5*(NBits)))

class phaseShift(genericGate):
    """ A callable phase-shift gate object. 
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
    
    phi : float
        The phase angle through which to shift the amplitudes.
    """
    def __init__(self, NBits, phi):
        super(phaseShift, self).__init__(NBits)
        singleMatrix = np.array([[1,0],[0,np.exp(phi * 1j)]])
        self.matrix = _toNBitMatrix(singleMatrix, NBits)

class pauliX(genericGate):
    """ A callable Pauli-X gate object. 
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
    """
    def __init__(self, NBits):
        super(pauliX, self).__init__(NBits)
        singleMatrix = np.array([[0,1],[1,0]])    
        self.matrix = _toNBitMatrix(singleMatrix, NBits)

class pauliY(genericGate):
    """ A callable Pauli-Y gate object. 
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
    """
    def __init__(self, NBits):
        super(pauliY, self).__init__(NBits)
        singleMatrix = np.array([[0,-1j],[1j,0]])    
        self.matrix = _toNBitMatrix(singleMatrix, NBits) 

class pauliZ(phaseShift):
    """ A callable Pauli-Z gate object. 
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
    """
    def __init__(self, NBits):
        super(pauliZ, self).__init__(NBits, np.pi)

class cNot(genericGate):
    """ A callable CNOT gate object. 
    
    """ 
    
    def __init__(self):     # (first bit is the control bit)
        super(cNot, self).__init__(2)   
        self.matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def _checkNBits(NBits):
    """ Validate the NBits input. """
    if NBits < 1:
        raise ValueError("NBits must be a positive integer!")
    
    if type(NBits) != int:
        raise TypeError("NBits must be a positive integer!")

def _toNBitMatrix(m, NBits, skipBits=[]):   # add ability to skip bits???
    """ Take a single-bit matrix of a gate and return the NBit equivalent matrix """
    m0 = m
    mOut = m
    for i in range(NBits - 1):
        mOut = np.kron(mOut, m0)
    
    return mOut
 
from .main import register