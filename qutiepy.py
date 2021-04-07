"""
qutiepy.py
==========================
The main file containing the core components of QutiePy, including classes for gates and the register class as well as a 
few functions which handle on gate and register objects:
"""
import numpy as np
import scipy.linalg as sp
import random
import warnings




class register:
    """
    N-bit quantum register. The initial state is (1+0j)|0>.
    
    Parameters
    ----------
    NBits : int
        Size of the register in bits.

    Attributes
    ----------
    NBits : int
        Number of bits in the register.
    
    NStates : int
        Number of states the register can occupy (2^NBits).
    
    amps : list
        The complex probability amplitudes of each state of the register.

    """
    def __init__(self, NBits):
        _checkNBits(NBits)
        self.NBits = NBits
        self.NStates = 2 ** NBits
        self.amps = np.zeros(self.NStates, dtype=np.dtype(complex))
        self.amps[0] = 1 + 0j
    
    def probabilities(self):
        """ Return the probability associated with observing each state. 
        
        Returns
        ----------
        probabilities : numpy array
            The probabilities p(X=x_i) of observing each state.
        """
        return np.array([abs(i)**2 for i in self.amps])
    
    def observe(self, bit=-1, collapseStates=True, fmt="int"):
        """ 'Observe' the register and return an integer representation of the
            observed state according to the probability amplitudes.
            
            If a bit argument is supplied, only that bit will be observed and it's
            state returned as an integer (1 or 0).

            If collapseStates=True, adjust the amplitudes to reflect the collapsed
            state.
            
            Parameters
            ----------
            bit : int, optional
                The index of the bit to be observed.
            
            collapseStates : bool, optional
                Flag to set whether probability amplitudes will be affected by this
                observation.
            
            fmt : string, optional
                One of "int" (default), "bin" or "hex"
            
            Returns
            ----------
            state : int or string
                The observed state of the register or bit.
        """
        if bit == -1:
            probs = self.probabilities()
            choice = random.choices(range(self.NStates), probs)[0]

            if collapseStates:
                amps = [0]*self.NStates
                amps[choice] = 1
                self.setAmps(amps)
            
            try:
                fmt = fmt.lower()
            except:
                raise TypeError("fmt must be a string")
                
            if fmt not in ["hex","bin","int"]:
                raise ValueError("Format must be one of 'hex','bin' or 'int'")
                
            if fmt == "int":
                return choice
            if fmt == "bin":
                return f'{0:0{self.NBits}b}'.format(choice)
            if fmt == "hex":
                return hex(choice)
            
        else:
            if bit > self.NBits - 1:
                raise IndexError("Value of 'bit' cannot be greater than self.NBits.")
            probs = self.probabilities()
            bitProbs = [sum([probs[i] for i in range(self.NStates) if format(i, f"0{self.NBits}b")[-bit] == "0"]), sum([probs[i] for i in range(self.NStates) if format(i, f"0{self.NBits}b")[-bit] == "1"])]
            bitChoice = random.choices([0,1], bitProbs)[0]
            
            if collapseStates:
                amps = self.amps
                zeroIndices = [i for i in range(self.NStates) if format(i, f"0{self.NBits}b")[-bit] != str(bitChoice)]
                for i in zeroIndices:
                    amps[i] = 0.0+0.0j
                self.setAmps(amps)
                
            return bitChoice
    
    def __str__(self):
        stri = ""
        for state, amp in enumerate(self.amps):
            stri = stri + f" {amp:.3f}".rjust(15) + " |" + str(state).ljust(2) + "> +\r\n"
        return stri.rstrip("+\r\n")
    
    def prod(self, B):
        """ Return the tensor product of self and B, 'Joining' two registers 
            into a single larger register with self at the MSB and 'B' at the LSB.
            
            Parameters
            ----------
            B : register
                The register to be appended to self.
            
            Returns
            ----------
            result : register
                The resulting joined register.
                
            See Also
            ----------
            prod : Equivalent function
        """
        result = register(self.NBits + B.NBits)
        result.amps = np.kron(self.amps, B.amps)    
        return result
    
    def bloch(self, eps=1e-12):
        """ Return the angles theta and phi used in the Bloch sphere representation 
        of a single qubit register (eps is used to avoid infinite and NaN results,
        set to 0 to disable).
        
        Returns
        ----------
        theta : float
            The angle theta = 2*arccos(amps[0]).
        
        phi : float
            The angle phi = Ln[(amps[1]+eps)/(sin(theta/2)+eps)].
        
        """
        if self.NBits != 1:
            raise ValueError("bloch() can only be called on 1-bit registers.")

        theta = 2 * np.arccos(self.amps[0])
        phi = np.real(np.log((self.amps[1]+eps)/(np.sin(theta/2)+eps))/1j)   # cast as float here?

        return theta, phi
    
    def density(self):
        """ Return the density matrix of the register. 
        
        Returns
        ----------
        density : numpy array
            The density matrix.
        """
        density = np.outer(self.amps, np.asmatrix(self.amps).H)
        
        return np.real(density)
    
    def reducedPurities(self):
        """ Return the reduced purity of each bit of the register, i.e.:
            Tr[Tr_i(D)^2]
            where D is the full density matrix of the register and Tr_i is 
            the partial trace over the subspace of bit index 'i'.
            
            Returns
            ----------
            purities : numpy array
                The reduced purity of each qubit in the register.
        """
        d = self.density()
        purities = []
        idxs = list(range(self.NBits))

        for i in idxs:
            theseidxs = idxs.copy()
            theseidxs.remove(i)
            pt = _partial_trace(d, theseidxs , [2]*(self.NBits), True)
            pt2 = np.matmul(pt,pt)
            purities.append(round(np.trace(pt2),8))
            # round used to reduce precision and remove small errors


        return np.array(purities)
    
    def setAmps(self, amps):
        """ Manually set the qubit probability amplitudes in-place, ensuring they remain properly normalised. 
        
        Parameters
        ----------
        amps : iterable
            The relative complex amplitudes to be applied to the register with normalisation.
        """
        if len(amps) != self.NStates:
            raise ValueError("Length of iterable 'amps' must be equal to NStates.")
        
        probs = np.array([abs(i)**2 for i in amps])
        self.amps = np.array(amps,  dtype=np.dtype(complex)) / sum(probs)**0.5



def prod(regA, regB):
    """ 'Join' two registers into a single larger register with regA at the MSB and regB at the LSB 
    by performing the Kronecker product between their state vectors \|A>\|B> = \|AB>. 
    
    Parameters
    ----------
    regA : register
        The register to be placed at the MSB.
    
    regB : register
        The register to be placed at the LSB.
    
    Returns
    ----------
    result : register
        The resulting joined register.
        
    See Also
    ----------
    register.prod : Equivalent class method.
    """
    result = register(regA.NBits + regB.NBits)
    result.amps = np.kron(regA.amps, regB.amps)  
    return result


def _partial_trace(rho, keep, dims, optimize=False):
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
    
    NControlBits : bool
        Indicates whether a control bit has been added using .addControlBits().
        
    isInverse : bool
        Indicates whether the object has been created by a call to H().
    """
    def __init__(self, NBits):
        _checkNBits(NBits)
        self.NBits = NBits
        self.matrix = np.identity(2 ** NBits)
        self.NControlBits = 0
        self.isInverse = False
    
    def __call__(self, arg):
        if issubclass(type(arg), genericGate):
            if arg.NBits != self.NBits:
                raise ValueError("Gates to be compounded must have the same NBits.")
            
            out = compoundGate(self.NBits)
            out.matrix = np.matmul(self.matrix, arg.matrix)
            return out

        elif type(arg) == register:
            result = register(arg.NBits)
            result.amps = np.matmul(self.matrix, arg.amps)
            return result
        
        else:
            raise TypeError("Gates can only be called on gates or registers! Got type: " +  str(type(arg)))
    
    def __str__(self):
        
        if self.NControlBits:
            cont =  "(" + str(self.NControlBits)  + " control qubits) "
        else:
            cont = ""
            
        if self.isInverse:
            inv = "Inverse "
        else:
            inv = ""
        
        stri = str(self.NBits) + "-qubit " + cont + inv + type(self).__name__ + " Gate, Matrix:\n\r"
        stri = stri + self.matrix.__str__()
        return stri
    
    def inverse(self):
        """ Alias of self.H().
    
        See Also
        ----------
        genericGate.H : This method maps to self.H()
        
        """
        return self.H()
        
    def H(self):
        """ Return an inverse copy of self, i.e. a gate whose matrix representation is 
            the Hermitian adjoint of self.matrix.
    
        Returns
        ----------
        gate : Gate Object
            The gate performing the inverse operation of self.
        
        """
        try:
            gate = (type(self))(self.NBits)
        except:
            gate = genericGate(self.NBits)
            
        gate.matrix = np.array(np.asmatrix(self.matrix).H)
        gate.isInverse = not self.isInverse
        
        return gate
    
    def addControlBits(self, NControlBits):
        """ Add control bits to this single bit gate. This is an in-place operation. 
    
        Parameters
        ----------
        
        NControlBits : int
            The number of control bits to add to the gate.
    
        Returns
        ----------
        result : bool
            True if successful.
        
        """
        if self.NBits != 1:
            raise ValueError("Control bits can only be added to single-bit gates")
        
        self.NBits += NControlBits
        NStates = 2 ** self.NBits
        oldMatrix = self.matrix
        
        self.matrix = np.eye(NStates)
        self.matrix[NStates-2:NStates, NStates-2:NStates] = oldMatrix
        
        self.NControlBits = NControlBits
        return True

class compoundGate(genericGate):
    """ A class returned when gates are compounded. See genericGate attributes.
    """
    
    
    def __init__(self, NBits):
        super(compoundGate, self).__init__(NBits)
    
    def __str__(self):
        stri = str(self.NBits) + "-qubit Compound Gate, Matrix:\n\r"
        stri = stri + self.matrix.__str__()
        return stri
        

class hadamard(genericGate):
    """ A callable hadamard gate object. 
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
        
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, NBits, skipBits = []):
        super(hadamard, self).__init__(NBits)
        self.matrix = _toNBitMatrix(sp.hadamard(2) * (2**(-0.5)), NBits, skipBits)

class phaseShift(genericGate):
    """ A callable phase-shift gate object. 
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
    
    phi : float
        The phase angle through which to shift the amplitudes.
        
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, NBits, phi, skipBits=[]):
        super(phaseShift, self).__init__(NBits)
        singleMatrix = np.array([[1,0],[0,np.exp(phi * 1j)]])
        self.matrix = _toNBitMatrix(singleMatrix, NBits, skipBits)

class pauliX(genericGate):
    """ A callable Pauli-X gate object, AKA the quantum NOT gate.
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
     
    skipBits : list of int, optional
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, NBits, skipBits=[]):
        super(pauliX, self).__init__(NBits)
        singleMatrix = np.array([[0,1],[1,0]])    
        self.matrix = _toNBitMatrix(singleMatrix, NBits, skipBits)

class pauliY(genericGate):
    """ A callable Pauli-Y gate object. 
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
       
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, NBits, skipBits=[]):
        super(pauliY, self).__init__(NBits)
        singleMatrix = np.array([[0,-1j],[1j,0]])    
        self.matrix = _toNBitMatrix(singleMatrix, NBits, skipBits) 

class pauliZ(phaseShift):
    """ A callable Pauli-Z gate object. 
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
       
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, NBits, skipBits=[]):
        super(pauliZ, self).__init__(NBits, np.pi, skipBits)

class cNot(genericGate):
    """ A callable CNOT gate object. 
    
    The first bit (LSB) in the register on which this gate is called is the control bit.
    """ 
    
    def __init__(self):     # (first bit is the control bit)
        super(cNot, self).__init__(2)   
        self.matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

class ccNot(genericGate):
    """ A callable CCNOT (Toffoli) gate object.
    
    The first two bits (LSBs) in the register on which this gate is called are the control bits. 
    """
    def __init__(self):
        super(ccNot, self).__init__(3)
        self.matrix = np.eye(8)
        self.matrix[6:8, 6:8] = np.array([[0,1],[1,0]])

class swap(genericGate):
    """ A callable SWAP gate object. For two qubits in a register (A,B), outputs (B,A).
    """
    def __init__(self):
        super(swap, self).__init__(2)
        self.matrix = np.eye(4)
        self.matrix[1:3, 1:3] = np.array([[0,1],[1,0]])

class sqrtSwap(genericGate):
    """ A callable sqrt(SWAP) gate object.
    """
    def __init__(self):
        super(sqrtSwap, self).__init__(2)
        self.matrix = np.eye(4)
        self.matrix[1:3, 1:3] = np.array([[0.5+0.5j,0.5-0.5j],[0.5-0.5j,0.5+0.5j]])

class sqrtNot(genericGate):
    """ A callable sqrt(not) or sqrt(Pauli-X) gate object.
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
       
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, NBits, skipBits=[]):
        super(sqrtNot, self).__init__(NBits)
        self.matrix = _toNBitMatrix(np.array([[0.5+0.5j,0.5-0.5j],[0.5-0.5j,0.5+0.5j]]), NBits, skipBits)

class fredkin(genericGate):
    """ A callable Fredkin (CCSWAP) gate object.
    """
    def __init__(self):
        super(fredkin, self).__init__(3)
        self.matrix = np.eye(8)
        self.matrix[5:7,5:7] = np.array([[0,1],[1,0]])
        
class identity(genericGate):
    """ A callable identity gate object.
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
    
    """
    def __init__(self, NBits):
        super(identity, self).__init__(NBits)
        self.matrix = np.eye(2**NBits)

class QFT(genericGate):
    """ A callable quantum Fourier transform (QFT) gate object.
    
    Parameters
    ----------
    NBits : int
        Number of bits that the gate takes as input/output.
    
    omega : complex
        
    
    """
    def __init__(self, NBits):
        super(QFT, self).__init__(NBits)
        self.matrix = _QFTMatrix(2**NBits)
        
class parallelGate(genericGate):   # test me
    """ A gate class to combine gates in parallel.
    
    Parameters
    ----------
    gates : array of gate objects
        The gates to be combined in parallel, with the first gate acting on the
        LSB of an applied register.
       
    """
    def __init__(self, gates):
        try:
            self.NBits = sum([g.NBits for g in gates])
        except:
            raise TypeError("gates must be an iterable of gate objects.")
            
        super(parallelGate, self).__init__(self.NBits)
        
        matrix = gates[0].matrix
        for gate in gates[1:]:
            matrix = np.kron(matrix, gate.matrix)
        
        self.matrix = matrix

def serialGates(gates):
    """ Combine an iterable of gate objects in series.
    
    Parameters
    ----------
    gates : iterable of gate objects
        The gate objects to be combined in serial. The operator to be applied first should be at index 0.
    
    Returns
    ----------
    comp : compoundGate object
        The result of combining the supplied operators in series.
        
    """
    comp = gates[0]
    for g in gates[1:]:
        comp = g(comp)
    
    return comp

def setSeed(seed):
    """ Set the RNG seed for reproducibility.
    
    Parameters
    ----------
    seed : string
        The seed to be used by the RNG.
    
    Returns
    ----------
    result : bool
        True if successful.
        
    """
    
    if type(seed) != str:
        raise ValueError("The seed must be a string")
    
    random.seed(seed)
    return True
    
    

def _checkNBits(NBits):
    """ Validate the NBits input. """
    if NBits < 1:
        raise ValueError("NBits must be a positive integer!")
    
    if type(NBits) != int:
        raise TypeError("NBits must be a positive integer!")
    
    if NBits > 12:
        warnings.warn("Using more than ~12 qubits in a gate or register will use a lot of resources and is not recommended!")

def _toNBitMatrix(m, NBits, skipBits=[]):   
    """ Take a single-bit matrix of a gate and return the NBit equivalent matrix. """
    
    if skipBits != []:
        raise DeprecationWarning("Using skipBits is no longer recommended, use parallelGate to combine identity gates with your desired operator.")
            
    I = np.eye(2)
    
    if 0 in skipBits:
        mOut = I
    else:
        mOut = m
    
    for i in range(1, NBits):
        if i in skipBits:
            mOut = np.kron(I, mOut)
        else:
            mOut = np.kron(m, mOut)

    return mOut

# def _toControlled(gate):    # Not working, addControlBits seems to work.
    # rootGate = genericGate(2)
    # rootGate.matrix = np.kron(np.eye(2), sp.sqrtm(gate.matrix)) # single bit gate
    # rootGateT = genericGate(2)
    # rootGateT.matrix = np.kron(np.eye(2), np.array(np.asmatrix(sp.sqrtm(gate.matrix)).H))
    # cn = cNot()
    # controlled = cn(rootGateT(cn(rootGate)))
    
    # return controlled

def _QFTMatrix(N):
    omg = np.e ** (2*1j*np.pi/N)
    matrix = np.zeros((N,N), dtype=complex)
    for x in range(N):
        for y in range(N):
            matrix[x,y] = omg ** (x*y)
    
    return matrix / np.sqrt(N)
    

if __name__ == "__main__":
    print("Why are you running the source file as __main__???")
    r = register(4)
    print(r.observe(fmt="hex"))

    
    
    
    
    




