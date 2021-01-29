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
        
        
        
        

if __name__ == "__main__":
    unittest.main()
