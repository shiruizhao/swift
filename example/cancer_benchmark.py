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
Pollution ~= choice({'low' : 0.5,'medium' : 0.4,'high' : 0.1})


Smoker ~= choice({'True' : 0.3,'False' : 0.7})


if (Pollution == 'low'):
	if (Smoker == 'True'):
		Cancer ~= choice({'True' : 0.03, 'False' : 0.97})
	else:
		Cancer ~= choice({'True' : 0.001, 'False' : 0.999})
elif(Pollution == 'medium'):
	if(Smoker == 'True'):
		Cancer ~= choice({'True' : 0.03, 'False' : 0.97})
	else:
		Cancer ~= choice({'True' : 0.001, 'False' : 0.999})
else:
	if(Smoker == 'True'):
		Cancer ~= choice({'True' : 0.05, 'False' : 0.95})
	else:
		Cancer ~= choice({'True' : 0.02, 'False' : 0.98})


if (Cancer == 'True'):
	Dyspnoea ~= choice({'True' : 0.65, 'False' : 0.35})
else:
	Dyspnoea ~= choice({'True' : 0.3, 'False' : 0.7})


if (Cancer == 'True'):
	Xray ~= choice({'positive' : 0.9, 'negative' : 0.09999999999999998})
else:
	Xray ~= choice({'positive' : 0.2, 'negative' : 0.8})


'''
compiler = SPPL_Compiler(data)
namespace = compiler.execute_module()
model=namespace.model
Cancer = Id('Cancer')
Dyspnoea = Id('Dyspnoea')
Pollution = Id('Pollution')
Smoker = Id('Smoker')
Xray = Id('Xray')
events = [Smoker << {'True'},Dyspnoea << {'False'},Pollution << {'low'},Cancer << {'False'},Xray << {'positive'},Smoker << {'False'},Smoker << {'True'},Smoker << {'True'},Cancer << {'False'},Smoker << {'False'},Smoker << {'True'},Cancer << {'False'},Dyspnoea << {'True'},Cancer << {'True'},Dyspnoea << {'False'},Dyspnoea << {'False'},Xray << {'negative'},Smoker << {'True'},Xray << {'negative'},Pollution << {'high'},Dyspnoea << {'False'},Xray << {'positive'},Xray << {'negative'},Cancer << {'False'},Pollution << {'low'},Xray << {'negative'},Dyspnoea << {'True'},Pollution << {'low'},Dyspnoea << {'True'},Xray << {'positive'},Pollution << {'high'},Smoker << {'False'},Pollution << {'high'},Pollution << {'medium'},Dyspnoea << {'True'},Smoker << {'True'},Xray << {'negative'},Dyspnoea << {'False'},Pollution << {'low'},Smoker << {'False'},Pollution << {'low'},Smoker << {'True'},Cancer << {'True'},Dyspnoea << {'True'},Cancer << {'False'},Cancer << {'False'},Smoker << {'True'},Smoker << {'True'},Smoker << {'False'},Xray << {'positive'},(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'})]
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
