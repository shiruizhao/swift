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


E ~= choice({'high' : 0.75, 'uni' : 0.25}) if ((A == 'young') and (S == 'M')) else (choice({'high' : 0.64, 'uni' : 0.36}) if ((A == 'young') and (S == 'F')) else (choice({'high' : 0.72, 'uni' : 0.28}) if ((A == 'adult') and (S == 'M')) else (choice({'high' : 0.7, 'uni' : 0.3}) if ((A == 'adult') and (S == 'F')) else (choice({'high' : 0.88, 'uni' : 0.12}) if ((A == 'old') and (S == 'M')) else (choice({'high' : 0.9, 'uni' : 0.1}))))))

O ~= choice({'emp' : 0.96, 'self' : 0.04}) if (E == 'high') else (choice({'emp' : 0.92, 'self' : 0.08}))

R ~= choice({'small' : 0.25, 'big' : 0.75}) if (E == 'high') else (choice({'small' : 0.2, 'big' : 0.8}))

T ~= choice({'car' : 0.48, 'train' : 0.42, 'other' : 0.1}) if ((O == 'emp') and (R == 'small')) else (choice({'car' : 0.58, 'train' : 0.24, 'other' : 0.18}) if ((O == 'emp') and (R == 'big')) else (choice({'car' : 0.56, 'train' : 0.36, 'other' : 0.08}) if ((O == 'self') and (R == 'small')) else (choice({'car' : 0.7, 'train' : 0.21, 'other' : 0.09}))))

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
events = [A << {'old'},S << {'M'},E << {'uni'},O << {'emp'},R << {'big'},E << {'uni'},O << {'self'},O << {'self'},A << {'old'},O << {'self'},E << {'high'},S << {'M'},O << {'self'},A << {'young'},A << {'old'},O << {'emp'},T << {'car'},E << {'uni'},T << {'car'},O << {'emp'},R << {'small'},O << {'emp'},T << {'car'},T << {'train'},S << {'M'},R << {'small'},T << {'train'},R << {'big'},A << {'old'},O << {'emp'},E << {'high'},O << {'emp'},T << {'other'},O << {'emp'},E << {'high'},S << {'F'},R << {'big'},A << {'adult'},A << {'young'},T << {'other'},A << {'adult'},O << {'self'},T << {'other'},E << {'high'},T << {'other'},S << {'M'},O << {'emp'},S << {'M'},O << {'emp'},E << {'uni'},(A << {'old'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'other'}),(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'car'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'train'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'car'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'car'}),(A << {'adult'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'old'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'train'}),(A << {'young'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'other'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'car'}),(A << {'adult'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'car'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'other'}),(A << {'old'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'old'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'M'}) & (T << {'car'}),(A << {'young'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'old'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'car'}),(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'car'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'car'}),(A << {'young'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'train'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'M'}) & (T << {'car'}),(A << {'old'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'car'}),(A << {'adult'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'train'}),(A << {'adult'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'young'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'high'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'other'}),(A << {'young'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'car'}),(A << {'young'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'M'}) & (T << {'train'}),(A << {'old'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'other'}),(A << {'young'}) & (E << {'high'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'train'}),(A << {'old'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'F'}) & (T << {'other'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'big'}) & (S << {'M'}) & (T << {'car'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'car'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'big'}) & (S << {'M'}) & (T << {'train'}),(A << {'young'}) & (E << {'uni'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'F'}) & (T << {'train'}),(A << {'old'}) & (E << {'high'}) & (O << {'emp'}) & (R << {'small'}) & (S << {'M'}) & (T << {'car'}),(A << {'adult'}) & (E << {'uni'}) & (O << {'self'}) & (R << {'small'}) & (S << {'F'}) & (T << {'other'})]
for event in events:
	start_time=time.time()
	query_prob=model.prob(event)
	print("--- %s seconds ---" % (time.time() - start_time))

	print(query_prob)
