Welcome to QutiePy!
=======================================

Package Description
=====================

QutiePy is a python3 package designed to provide an easy and practical toolset for simulated quantum computing aimed users with any experience level in quantum theory
and a working knowledge of pythonic OOP.

*QutiePy is currently a work-in-progress and is being developed by a first-timer. Please report any issues or suggestions on the GitHub repo at <https://github.com/franklinscudder/QutiePy/issues>.*

Usage Example
==================

This script demonstrates the basic usage of the module, constructing the Bell state 'Phi+'. A Bell state is a pair of qubits which exhibit maximal entanglement. They are the simplest example of this phenomenon.

*This script can be found in the examples folder of your package installation.*

.. code-block:: python

   import qutiepy as qu

   # Create two 1-qubit registers, initialised to 0 by default
   r1 = qu.register(1)
   r2 = qu.register(1)

   # Create a Hadamard and a CNOT gate
   h = qu.hadamard(1)
   cn = qu.cNot()

   # Apply the hadamard operation to r1
   r1 = h(r1)

   # 'Join' the two bits together into a single register
   r = qu.prod(r1,r2)

   # Apply the CNOT gate to the new register
   r = cn(r)

   # Analyse the resulting Bell state
   print("The state vector of r:")
   print(r)
   print()
   print("The probabilities of observing each state of r:")
   print(r.probabilities())
   print()
   print("The reduced purities of each qubit in r (0.5 = maximally entangled, 1 = fully unentangled):")
   print(r.reducedPurities())
   print()
   print("The result of observing ten versions of the Bell state r:")
   for i in range(10):
       print(r.observe(collapseStates=False)),

Documentation
=========================

Documentation can be found on [ReadTheDocs](https://qutiepy.readthedocs.io/en/latest/)
