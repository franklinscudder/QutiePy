import qutiepy as qu
import unittest
import numpy as np


class testRegisterClass(unittest.TestCase):
    def testInit(self):
        with self.assertRaises(ValueError):
            qu.register(0)
        
        with self.assertRaises(ValueError):
            qu.register(-1)
        
        with self.assertRaises(ValueError):
            qu.register("WhyAreYouReadingTestScripts?")
        
        r = qu.register(4)
        self.assertEqual(r.NBits, 4)
        self.assertEqual(r.NStates, 16)
        self.assertEqual(r.probabilities(), np.array([1] + [0]*15))
        self.assertEqual(r.observe(), 0)
        
        for cBits in range(1, 7):
            n = pauliX(1)
            n.addControlBits(cBits)
            r = register(1+cBits)
            nEq = 0
    
            for i in range(2**cBits):
                r.setAmps([0]*(2**cBits) + [int(j == i) for j in range(2**cBits)])
                if r.observe(collapseStates=False) != n(r).observe(collapseStates=False):
                    nEq += 1
        
            self.assertEqual(nEq, 2)

        
        
        
        

if __name__ == "__main__":
    unittest.main()
