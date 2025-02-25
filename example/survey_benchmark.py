from sppl.compilers.ast_to_spe import Id
from sppl.compilers.ast_to_spe import IfElse
from sppl.compilers.ast_to_spe import Sample
from sppl.compilers.ast_to_spe import Sequence
from sppl.compilers.sppl_to_python import SPPL_Compiler
from sppl.distributions import atomic
from sppl.distributions import choice
from sppl.distributions import uniform
from sppl.math_util import allclose
from sppl.sets import Interval
from sppl.spe import ExposedSumSPE
from sppl.compilers.sppl_to_python import SPPL_Compiler
import os
import time
import numpy as np

isclose = lambda a, b : abs(a-b) < 1e-10


data = '''
A ~= choice({'young' : 0.3,'adult' : 0.5,'old' : 0.2})


S ~= choice({'M' : 0.6,'F' : 0.4})


if (A == 'young'):
	if (S == 'M'):
		E ~= choice({'high' : 0.75, 'uni' : 0.25})
	else:
		E ~= choice({'high' : 0.64, 'uni' : 0.36})
elif(A == 'adult'):
	if(S == 'M'):
		E ~= choice({'high' : 0.72, 'uni' : 0.28})
	else:
		E ~= choice({'high' : 0.7, 'uni' : 0.30000000000000004})
else:
	if(S == 'M'):
		E ~= choice({'high' : 0.88, 'uni' : 0.12})
	else:
		E ~= choice({'high' : 0.9, 'uni' : 0.09999999999999998})


if (E == 'high'):
	O ~= choice({'emp' : 0.96, 'self' : 0.040000000000000036})
else:
	O ~= choice({'emp' : 0.92, 'self' : 0.07999999999999996})


if (E == 'high'):
	R ~= choice({'small' : 0.25, 'big' : 0.75})
else:
	R ~= choice({'small' : 0.2, 'big' : 0.8})


if (O == 'emp'):
	if (R == 'small'):
		T ~= choice({'car' : 0.48, 'train' : 0.42, 'other' : 0.10000000000000009})
	else:
		T ~= choice({'car' : 0.58, 'train' : 0.24, 'other' : 0.18000000000000005})
else:
	if(R == 'small'):
		T ~= choice({'car' : 0.56, 'train' : 0.36, 'other' : 0.07999999999999996})
	else:
		T ~= choice({'car' : 0.7, 'train' : 0.21, 'other' : 0.09000000000000008})


'''
compiler = SPPL_Compiler(data)
namespace = compiler.execute_module()
model=namespace.model
A = Id('A')
E = Id('E')
O = Id('O')
R = Id('R')
S = Id('S')
T = Id('T')
events = [E << {'uni'},R << {'small'},A << {'old'},S << {'F'},E << {'uni'},T << {'train'},O << {'emp'},T << {'train'},O << {'emp'},E << {'high'},T << {'other'},S << {'F'},S << {'F'},R << {'big'},T << {'other'},R << {'big'},O << {'emp'},S << {'F'},O << {'emp'},S << {'F'},T << {'other'},S << {'F'},R << {'small'},E << {'uni'},E << {'high'},R << {'small'},A << {'adult'},S << {'M'},A << {'adult'},E << {'uni'},E << {'uni'},A << {'young'},A << {'young'},R << {'small'},T << {'train'},E << {'uni'},R << {'big'},E << {'high'},T << {'train'},S << {'F'},O << {'emp'},A << {'adult'},O << {'emp'},R << {'big'},A << {'old'},R << {'big'},A << {'adult'},S << {'F'},O << {'self'},S << {'F'},(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'car'}),(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'other'}),(A << {'adult'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'car'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'car'}),(A << {'old'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'car'}),(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'train'}),(A << {'old'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'M'}) & (T << {'car'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'adult'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'car'}),(A << {'old'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'old'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'other'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'M'}) & (T << {'other'}),(A << {'old'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'train'}),(A << {'adult'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'car'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'car'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'train'}),(A << {'young'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'car'}),(A << {'young'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'train'}),(A << {'old'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'other'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'young'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'old'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'car'}),(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'car'}),(A << {'adult'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'M'}) & (T << {'other'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'other'}),(A << {'old'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'car'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'other'}),(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'old'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'train'}),(A << {'young'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'car'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'train'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'old'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'other'})]
runtime=np.zeros(100)
for i in range(100):
	start_time=time.time()
	query_prob=model.prob(events[i])
	end_time = time.time()
	print("--- %s seconds ---" % (end_time - start_time))

	print(query_prob)

	runtime[i]=end_time-start_time
print("single marginal time:%s"%np.mean(runtime[0:50]))
print("all marginal time:%s"%np.mean(runtime[50:100]))
