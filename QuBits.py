"""
QuBits.py
==========================
The main file containing all components of QuBits
"""
import numpy as np
import scipy.linalg as sp
import random

class register:
    """ An N-bit quantum register object, with 2^N states. """
    def __init__(self, NBits):
        checkNBits(NBits)
        self.NBits = NBits
        self.NStates = 2 ** NBits
        self.amps = np.zeros(self.NStates, dtype=np.dtype(complex))
        self.amps[0] = 1 + 0j
    
    def probabilities(self):
        """ Return the probability associated with observing each state. """
        return np.array([abs(i)**2 for i in self.amps])
    
    def observe(self, collapseStates=True):
        """ 'Observe' the register and return an integer representation of the
            observed state according to the probability amplitudes.

            If collapseStates=True, adjust the amplitudes to reflect the collapsed
            state.
        """
        probs = self.probabilities()
        choice = random.choices(range(self.NStates), probs)[0]

        if collapseStates:
            amps = [0]*self.NStates
            amps[choice] = self.amps[choice]
            self.setAmps(amps)
        
        return choice
    
    def __str__(self):
        stri = ""
        for state, amp in enumerate(self.amps):
            stri = stri + f' {amp:.3f}'.rjust(15) + " |" + str(state).ljust(2) + "> +\r\n"
        return stri.rstrip("+\r\n")
    
    def prod(self, B):
        """ Return the tensor product of self and B, 'Joining' two registers 
            into a single larger register with self at the MSB and 'B' at the LSB. 
        """
        result = register(self.NBits + B.NBits)
        result.amps = np.kron(self.amps, B.amps)    
        return result
    
    def bloch(self, eps=1e-12):
        """ Return the angles theta and phi used in the Bloch sphere representation 
        of a single qubit register. (eps is used to avoid infinite and NaN results,
        set to 0 to disable) 
        """
        if self.NBits != 1:
            raise ValueError("bloch() can only be called on 1-bit registers.")

        theta = 2 * np.arccos(self.amps[0])
        phi = np.log((self.amps[1]+eps)/(np.sin(theta/2)+eps))/1j   # cast as float here?

        return theta, phi
    
    def density(self):
        """ Return the density matrix of the register. """
        out = np.outer(self.amps, np.asmatrix(self.amps).H)
        
        return np.real(out)
    
    def reducedPurities(self):
        """ Return the reduced purity of each bit of the register, i.e.:
            Tr[Tr_i(D)^2]
            where D is the full density matrix of the register and Tr_i is 
            the partial trace over the subspace of bit index 'i'.
        """
        d = self.density()
        purities = []
        idxs = list(range(self.NBits))

        for i in idxs:
            theseidxs = idxs.copy()
            theseidxs.remove(i)
            pt = partial_trace(d, theseidxs , [2]*(self.NBits), True)
            pt2 = np.matmul(pt,pt)
            purities.append(round(np.trace(pt2),8))
            # round used to reduce precision and remove small errors


        return purities
    
    def setAmps(self, amps):
        """ Manually set the qubit probability amplitudes ensuring they remain properly normalised. """
        if len(amps) != self.NStates:
            raise ValueError("Length of iterable 'amps' must be equal to NStates.")
        
        probs = np.array([abs(i)**2 for i in amps])
        self.amps = np.array(amps,  dtype=np.dtype(complex)) / sum(probs)**0.5

class genericGate:
    """ A base class for callable quantum logic gates. """
    def __init__(self, NBits):
        checkNBits(NBits)
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
    """ Create a callable hadamard gate object """
    def __init__(self, NBits):
        super(hadamard, self).__init__(NBits)
        self.matrix = sp.hadamard(2 ** NBits) * (2**(-0.5*(NBits)))

class phaseShift(genericGate):
    """ Create a callable phase-shift gate object """
    def __init__(self, NBits, phi):
        super(phaseShift, self).__init__(NBits)
        singleMatrix = np.array([[1,0],[0,np.exp(phi * 1j)]])
        self.matrix = toNBitMatrix(singleMatrix, NBits)

class pauliX(genericGate):
    """ Create a callable Pauli-X gate object """
    def __init__(self, NBits):
        super(pauliX, self).__init__(NBits)
        singleMatrix = np.array([[0,1],[1,0]])    
        self.matrix = toNBitMatrix(singleMatrix, NBits)

class pauliY(genericGate):
    """ Create a callable Pauli-Y gate object """
    def __init__(self, NBits):
        super(pauliY, self).__init__(NBits)
        singleMatrix = np.array([[0,-1j],[1j,0]])    
        self.matrix = toNBitMatrix(singleMatrix, NBits) 

class pauliZ(phaseShift):
    """ Create a callable Pauli-Z gate object """
    def __init__(self, NBits):
        super(pauliZ, self).__init__(NBits, np.pi)

class cNot(genericGate):
    """ Create Controlled-Not gate object (first bit is the control bit) """
    def __init__(self):
        super(cNot, self).__init__(2)   
        self.matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def checkNBits(NBits):
    """ Validate the NBits input. """
    if NBits < 1:
        raise ValueError("NBits must be a positive integer!")
    
    if type(NBits) != int:
        raise TypeError("NBits must be a positive integer!")
    


def prod(regA, regB):
    """ 'Join' two registers into a single larger register with regA at the MSB and regB at the LSB. """
    result = register(regA.NBits + regB.NBits)
    result.amps = np.kron(regA.amps, regB.amps)  
    return result

def toNBitMatrix(m, NBits, skipBits=[]):   # add ability to skip bits
    """ Take a single-bit matrix of a gate and return the NBit equivalent matrix """
    m0 = m
    mOut = m
    for i in range(NBits - 1):
        mOut = np.kron(mOut, m0)
    
    return mOut

def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace (Thanks to slek120 on StackExchange)

    ρ_a = Tr_b(ρ)

    Parameters
    ----------
    ρ : 2D array
        Matrix to trace
    keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    ρ_a : 2D array
        Traced matrix
    """
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim+i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims,2))
    rho_a = np.einsum(rho_a, idx1+idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)

if __name__ == "__main__":
    b0 = register(1)
    b1 = register(1)
    h = hadamard(1)
    cN = cNot()
    p = phaseShift(2,2412.134)

    r = p(cN(h(b0).prod(b1)))


    print(r.density())
    print(r.probabilities())
    print(r.reducedPurities())
    print(r)
    print(r.observe())
    print(r)

