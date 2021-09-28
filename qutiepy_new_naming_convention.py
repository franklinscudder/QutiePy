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

"""
TODO:

 - add control offset to default controlled gates.
 - check new name conventions.
 - change MSB convention, should only require changing .observe() et al.
 - update docs to represent new changes
 - reversion, release...
 
 """

class Register:
    """
    N-bit quantum register. The initial state is (1+0j)|0>.
    
    Parameters
    ----------
    N_bits : int
        Size of the register in bits.

    Attributes
    ----------
    N_bits : int
        Number of bits in the register.
    
    N_states : int
        Number of states the register can occupy (2^N_bits).
    
    amps : list
        The complex probability amplitudes of each state of the register.

    """
    def __init__(self, N_bits):
        _check_N_bits(N_bits)
        self.N_bits = N_bits
        self.N_states = 2 ** N_bits
        self.amps = np.zeros(self.N_states, dtype=np.dtype(complex))
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
            choice = random.choices(range(self.N_states), probs)[0]

            if collapseStates:
                amps = [0]*self.N_states
                amps[choice] = 1
                self.set_amps(amps)
            
            try:
                fmt = fmt.lower()
            except:
                raise TypeError("fmt must be a string")
                
            if fmt not in ["hex","bin","int"]:
                raise ValueError("Format must be one of 'hex','bin' or 'int'")
                
            if fmt == "int":
                return choice
            if fmt == "bin":
                return f'{0:0{self.N_bits}b}'.format(choice)
            if fmt == "hex":
                return hex(choice)
            
        else:
            if bit > self.N_bits - 1:
                raise IndexError("Value of 'bit' cannot be greater than self.N_bits.")
            probs = self.probabilities()
            bitProbs = [sum([probs[i] for i in range(self.N_states) if format(i, f"0{self.N_bits}b")[-bit] == "0"]), sum([probs[i] for i in range(self.N_states) if format(i, f"0{self.N_bits}b")[-bit] == "1"])]
            bitChoice = random.choices([0,1], bitProbs)[0]
            
            if collapseStates:
                amps = self.amps
                zeroIndices = [i for i in range(self.N_states) if format(i, f"0{self.N_bits}b")[-bit] != str(bitChoice)]
                for i in zeroIndices:
                    amps[i] = 0.0+0.0j
                self.set_amps(amps)
                
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
        result = Register(self.N_bits + B.N_bits)
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
        if self.N_bits != 1:
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
    
    def reduced_purities(self):
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
        idxs = list(range(self.N_bits))

        for i in idxs:
            theseidxs = idxs.copy()
            theseidxs.remove(i)
            pt = _partial_trace(d, theseidxs , [2]*(self.N_bits), True)
            pt2 = np.matmul(pt,pt)
            purities.append(round(np.trace(pt2),8))
            # round used to reduce precision and remove small errors


        return np.array(purities)
    
    def set_amps(self, amps):
        """ Manually set the qubit probability amplitudes in-place, ensuring they remain properly normalised. 
        
        Parameters
        ----------
        amps : iterable
            The relative complex amplitudes to be applied to the register with normalisation.
        """
        if len(amps) != self.N_states:
            raise ValueError("Length of iterable 'amps' must be equal to N_states.")
        
        probs = np.array([abs(i)**2 for i in amps])
        self.amps = np.array(amps,  dtype=np.dtype(complex)) / sum(probs)**0.5



def prod(regA, regB):
    """ 'Join' two registers into a single larger register with regA at the MSB and regB at the LSB 
    by performing the Kronecker product between their state vectors \|A>\|B> = \|AB>. 
    
    Parameters
    ----------
    regA : Register
        The register to be placed at the MSB.
    
    regB : Register
        The register to be placed at the LSB.
    
    Returns
    ----------
    result : Register
        The resulting joined register.
        
    See Also
    ----------
    Register.prod : Equivalent class method.
    """
    result = Register(regA.N_bits + regB.N_bits)
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

class GenericGate:
    """
    Base class for callable quantum logic gates.
    
    Parameters
    ----------
    N_bits : int
        Size of the registers that the gate will take as input/output.

    Attributes
    ----------
    N_bits : int
        Number of bits that the gate takes as input/output.
    
    matrix : 2D numpy array
        The 2^N_bits by 2^N_bits matrix representation of the gate.
    
    NControlBits : bool
        Indicates whether a control bit has been added using .add_control_bits().
        
    isInverse : bool
        Indicates whether the object has been created by a call to H().
    """
    def __init__(self, N_bits):
        _check_N_bits(N_bits)
        self.N_bits = N_bits
        self.matrix = np.identity(2 ** N_bits)
        self.NControlBits = 0
        self.isInverse = False
    
    def __call__(self, arg):
        if issubclass(type(arg), GenericGate):
            if arg.N_bits != self.N_bits:
                raise ValueError("Gates to be compounded must have the same N_bits.")
            
            out = CompoundGate(self.N_bits)
            out.matrix = np.matmul(self.matrix, arg.matrix)
            return out

        elif type(arg) == Register:
            result = Register(arg.N_bits)
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
        
        stri = str(self.N_bits) + "-qubit " + cont + inv + type(self).__name__ + " Gate, Matrix:\n\r"
        stri = stri + self.matrix.__str__()
        return stri
    
    def inverse(self):
        """ Alias of self.H().
    
        See Also
        ----------
        GenericGate.H : This method maps to self.H()
        
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
            gate = (type(self))(self.N_bits)
        except:
            gate = GenericGate(self.N_bits)
            
        gate.matrix = np.array(np.asmatrix(self.matrix).H)
        gate.isInverse = not self.isInverse
        
        return gate
    
    def add_control_bits(self, controlOffsets):
        """ Add control bits to this single bit gate.  
    
        Parameters
        ----------
        
        controlOffsets : array of int != 0
            The bit index offsets relative to this gate to control off of.
    
        Returns
        ----------
        result : gate object
            The resulting controlled gate.
        
        """
        #if self.N_bits != 1:
            #raise ValueError("Control bits can only be added to single-bit gates")
        
        if not all([type(i) == int for i in controlOffsets]):
            raise TypeError("Elements of controlOffsets must be integers.")
        
        if not all([i != 0 for i in controlOffsets]):
            raise ValueError("Elements of controlOffsets must be non-zero.")
        
        positives = [i for i in controlOffsets if i > 0]
        negatives = [i for i in controlOffsets if i < 0]
        
        matrix = np.copy(self.matrix)
        
        if positives:
            for i in range(1, max(positives) + 1):
                
                if i in positives:
                    mat_ = np.copy(matrix)
                    shape_ = mat_.shape[0]
                    matrix = np.eye(shape_*2)
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        matrix[:-shape_, :-shape_] = mat_
                else:
                    matrix = np.kron(matrix, np.eye(2))
                    
        if negatives:
            for i in range(-1, min(negatives) - 1, -1):
                
                if i in negatives:
                    mat_ = np.copy(matrix)
                    shape_ = mat_.shape[0]
                    matrix = np.eye(shape_*2, dtype=complex)
                    matrix[shape_:, shape_:] = mat_
                else:
                    matrix = np.kron(np.eye(2), matrix)
        
        N_states = matrix.shape[0]
        N_bits = int(round(np.log2(N_states))) ## int?
        
        try:
            gate = (type(self))(N_bits)
        except:
            gate = GenericGate(N_bits)
        
        gate.matrix = matrix
        gate.NControlBits = len(controlOffsets)
        
        return gate
    

class CompoundGate(GenericGate):
    """ A class returned when gates are compounded. See GenericGate attributes.
    """
    
    
    def __init__(self, N_bits):
        super(CompoundGate, self).__init__(N_bits)
    
    def __str__(self):
        stri = str(self.N_bits) + "-qubit Compound Gate, Matrix:\n\r"
        stri = stri + self.matrix.__str__()
        return stri
        

class Hadamard(GenericGate):
    """ A callable Hadamard gate object. 
    
    Parameters
    ----------
    N_bits : int
        Number of bits that the gate takes as input/output.
        
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, N_bits, skipBits = []):
        super(Hadamard, self).__init__(N_bits)
        self.matrix = _to_N_bit_matrix(sp.hadamard(2) * (2**(-0.5)), N_bits, skipBits)

class Phase(GenericGate):
    """ A callable phase-shift gate object. 
    
    Parameters
    ----------
    N_bits : int
        Number of bits that the gate takes as input/output.
    
    phi : float
        The phase angle through which to shift the amplitudes.
        
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, N_bits, phi, skipBits=[]):
        super(Phase, self).__init__(N_bits)
        singleMatrix = np.array([[1,0],[0,np.exp(phi * 1j)]])
        self.matrix = _to_N_bit_matrix(singleMatrix, N_bits, skipBits)

class PauliX(GenericGate):
    """ A callable Pauli-X gate object, AKA the quantum NOT gate.
    
    Parameters
    ----------
    N_bits : int
        Number of bits that the gate takes as input/output.
     
    skipBits : list of int, optional
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, N_bits, skipBits=[]):
        super(PauliX, self).__init__(N_bits)
        singleMatrix = np.array([[0,1],[1,0]])    
        self.matrix = _to_N_bit_matrix(singleMatrix, N_bits, skipBits)

class PauliY(GenericGate):
    """ A callable Pauli-Y gate object. 
    
    Parameters
    ----------
    N_bits : int
        Number of bits that the gate takes as input/output.
       
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, N_bits, skipBits=[]):
        super(PauliY, self).__init__(N_bits)
        singleMatrix = np.array([[0,-1j],[1j,0]])    
        self.matrix = _to_N_bit_matrix(singleMatrix, N_bits, skipBits) 

class PauliZ(Phase):
    """ A callable Pauli-Z gate object. 
    
    Parameters
    ----------
    N_bits : int
        Number of bits that the gate takes as input/output.
       
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, N_bits, skipBits=[]):
        super(PauliZ, self).__init__(N_bits, np.pi, skipBits)

class CNot(GenericGate):
    """ A callable CNOT gate object. 
    
    The first bit (LSB) in the register on which this gate is called is the control bit.
    """ 
    
    def __init__(self):     # (first bit is the control bit)
        super(CNot, self).__init__(2)   
        self.matrix = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

class CCNot(GenericGate):
    """ A callable CCNOT (Toffoli) gate object.
    
    The first two bits (LSBs) in the register on which this gate is called are the control bits. 
    """
    def __init__(self):
        super(CCNot, self).__init__(3)
        self.matrix = np.eye(8)
        self.matrix[6:8, 6:8] = np.array([[0,1],[1,0]])

class Swap(GenericGate):
    """ A callable SWAP gate object. For two qubits in a register (A,B), outputs (B,A).
    """
    def __init__(self):
        super(Swap, self).__init__(2)
        self.matrix = np.eye(4)
        self.matrix[1:3, 1:3] = np.array([[0,1],[1,0]])

class SqrtSwap(GenericGate):
    """ A callable sqrt(SWAP) gate object.
    """
    def __init__(self):
        super(SqrtSwap, self).__init__(2)
        self.matrix = np.eye(4)
        self.matrix[1:3, 1:3] = np.array([[0.5+0.5j,0.5-0.5j],[0.5-0.5j,0.5+0.5j]])

class SqrtNot(GenericGate):
    """ A callable sqrt(not) or sqrt(Pauli-X) gate object.
    
    Parameters
    ----------
    N_bits : int
        Number of bits that the gate takes as input/output.
       
    skipBits : list of int, optional, deprecated
        Indices of bits in a register that this gate will not operate on.
    """
    def __init__(self, N_bits, skipBits=[]):
        super(SqrtNot, self).__init__(N_bits)
        self.matrix = _to_N_bit_matrix(np.array([[0.5+0.5j,0.5-0.5j],[0.5-0.5j,0.5+0.5j]]), N_bits, skipBits)

class Fredkin(GenericGate):
    """ A callable Fredkin (CCSWAP) gate object.
    """
    def __init__(self):
        super(Fredkin, self).__init__(3)
        self.matrix = np.eye(8)
        self.matrix[5:7,5:7] = np.array([[0,1],[1,0]])
        
class Identity(GenericGate):
    """ A callable identity gate object.
    
    Parameters
    ----------
    N_bits : int
        Number of bits that the gate takes as input/output.
    
    """
    def __init__(self, N_bits):
        super(Identity, self).__init__(N_bits)
        self.matrix = np.eye(2**N_bits)

class QFT(GenericGate):
    """ A callable quantum Fourier transform (QFT) gate object.
    
    Parameters
    ----------
    N_bits : int
        Number of bits that the gate takes as input/output.
    
    omega : complex
        
    
    """
    def __init__(self, N_bits):
        super(QFT, self).__init__(N_bits)
        self.matrix = _QFT_matrix(2**N_bits)
        
class ParallelGate(GenericGate):   # test me
    """ A gate class to combine gates in parallel.
    
    Parameters
    ----------
    gates : array of gate objects
        The gates to be combined in parallel, with the first gate acting on the
        LSB of an applied register.
       
    """
    def __init__(self, gates):
        try:
            self.N_bits = sum([g.N_bits for g in gates])
        except:
            raise TypeError("gates must be an iterable of gate objects.")
            
        super(ParallelGate, self).__init__(self.N_bits)
        
        matrix = gates[0].matrix
        for gate in gates[1:]:
            matrix = np.kron(matrix, gate.matrix)
        
        self.matrix = matrix

def serial_gate(gates):
    """ Combine an iterable of gate objects in series.
    
    Parameters
    ----------
    gates : iterable of gate objects
        The gate objects to be combined in serial. The operator to be applied first should be at index 0.
    
    Returns
    ----------
    comp : CompoundGate object
        The result of combining the supplied operators in series.
        
    """
    comp = gates[0]
    for g in gates[1:]:
        comp = g(comp)
    
    return comp

def set_seed(seed):
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
    
    

def _check_N_bits(N_bits):
    """ Validate the N_bits input. """
    if N_bits < 1:
        raise ValueError("N_bits must be a positive integer!")
    
    if type(N_bits) != int:
        raise TypeError("N_bits must be a positive integer!")
    
    if N_bits > 12:
        warnings.warn("Using more than ~12 qubits in a gate or register will use a lot of resources and is not recommended!")

def _to_N_bit_matrix(m, N_bits, skipBits=[]):   
    """ Take a single-bit matrix of a gate and return the NBit equivalent matrix. """
    
    if skipBits != []:
        raise DeprecationWarning("Using skipBits is no longer recommended, use ParallelGate to combine identity gates with your desired operator.")
            
    I = np.eye(2)
    
    if 0 in skipBits:
        mOut = I
    else:
        mOut = m
    
    for i in range(1, N_bits):
        if i in skipBits:
            mOut = np.kron(I, mOut)
        else:
            mOut = np.kron(m, mOut)

    return mOut

def _QFT_matrix(N):
    omg = np.e ** (2*1j*np.pi/N)
    matrix = np.zeros((N,N), dtype=complex)
    for x in range(N):
        for y in range(N):
            matrix[x,y] = omg ** (x*y)
    
    return matrix / np.sqrt(N)
    

if __name__ == "__main__":
    print("Why are you running the source file as __main__???")
    
    
    
    
    




