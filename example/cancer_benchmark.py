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


Cancer ~= choice({'True' : 0.03, 'False' : 0.97}) if ((Pollution == 'low') and (Smoker == 'True')) else (choice({'True' : 0.001, 'False' : 0.999}) if ((Pollution == 'low') and (Smoker == 'False')) else (choice({'True' : 0.03, 'False' : 0.97}) if ((Pollution == 'medium') and (Smoker == 'True')) else (choice({'True' : 0.001, 'False' : 0.999}) if ((Pollution == 'medium') and (Smoker == 'False')) else (choice({'True' : 0.05, 'False' : 0.95}) if ((Pollution == 'high') and (Smoker == 'True')) else (choice({'True' : 0.02, 'False' : 0.98}))))))

Xray ~= choice({'positive' : 0.9, 'negative' : 0.1}) if (Cancer == 'True') else (choice({'positive' : 0.2, 'negative' : 0.8}))

Dyspnoea ~= choice({'True' : 0.65, 'False' : 0.35}) if (Cancer == 'True') else (choice({'True' : 0.3, 'False' : 0.7}))

'''
compiler = SPPL_Compiler(data)
namespace = compiler.execute_module()
model=namespace.model
Cancer = Id('Cancer')
Dyspnoea = Id('Dyspnoea')
Pollution = Id('Pollution')
Smoker = Id('Smoker')
Xray = Id('Xray')
events = [Cancer << {'True'},Xray << {'positive'},Dyspnoea << {'True'},Xray << {'positive'},Dyspnoea << {'False'},Pollution << {'medium'},Pollution << {'medium'},Dyspnoea << {'True'},Dyspnoea << {'False'},Xray << {'positive'},Cancer << {'False'},Pollution << {'low'},Pollution << {'high'},Dyspnoea << {'False'},Smoker << {'False'},Pollution << {'high'},Smoker << {'True'},Pollution << {'medium'},Dyspnoea << {'True'},Xray << {'positive'},Dyspnoea << {'False'},Smoker << {'True'},Xray << {'positive'},Xray << {'negative'},Dyspnoea << {'True'},Pollution << {'high'},Xray << {'negative'},Pollution << {'low'},Pollution << {'low'},Xray << {'negative'},Cancer << {'True'},Dyspnoea << {'True'},Pollution << {'medium'},Pollution << {'low'},Cancer << {'False'},Xray << {'positive'},Pollution << {'high'},Dyspnoea << {'False'},Dyspnoea << {'True'},Xray << {'positive'},Xray << {'positive'},Smoker << {'True'},Cancer << {'False'},Smoker << {'True'},Dyspnoea << {'False'},Xray << {'negative'},Pollution << {'high'},Dyspnoea << {'False'},Xray << {'positive'},Cancer << {'True'},(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'negative'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'False'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'low'}) & (Smoker << {'True'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'False'}) & (Pollution << {'medium'}) & (Smoker << {'True'}) & (Xray << {'negative'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'True'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'False'}) & (Xray << {'positive'}),(Cancer << {'False'}) & (Dyspnoea << {'True'}) & (Pollution << {'high'}) & (Smoker << {'True'}) & (Xray << {'negative'})]
for event in events:
	start_time=time.time()
	query_prob=model.prob(event)
	print("--- %s seconds ---" % (time.time() - start_time))

	print(query_prob)
