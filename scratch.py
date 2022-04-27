from qutiepy_new_naming_convention import *


r = Register(4)

n = PauliX(1)

n = n.add_control_bits([-3])

print(n)

r.set_amps([1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
print(r)
print()

print(n(r))

