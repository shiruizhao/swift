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
ANAPHYLAXIS ~= choice({'TRUE' : 0.01,'FALSE' : 0.99})


DISCONNECT ~= choice({'TRUE' : 0.1,'FALSE' : 0.9})


ERRCAUTER ~= choice({'TRUE' : 0.1,'FALSE' : 0.9})


ERRLOWOUTPUT ~= choice({'TRUE' : 0.05,'FALSE' : 0.95})


FIO2 ~= choice({'LOW' : 0.05,'NORMAL' : 0.95})


HYPOVOLEMIA ~= choice({'TRUE' : 0.2,'FALSE' : 0.8})


INSUFFANESTH ~= choice({'TRUE' : 0.1,'FALSE' : 0.9})


INTUBATION ~= choice({'NORMAL' : 0.92,'ESOPHAGEAL' : 0.03,'ONESIDED' : 0.05})


KINKEDTUBE ~= choice({'TRUE' : 0.04,'FALSE' : 0.96})


LVFAILURE ~= choice({'TRUE' : 0.05,'FALSE' : 0.95})


MINVOLSET ~= choice({'LOW' : 0.05,'NORMAL' : 0.9,'HIGH' : 0.05})


PULMEMBOLUS ~= choice({'TRUE' : 0.01,'FALSE' : 0.99})


if (INTUBATION == 'NORMAL'):
	if (PULMEMBOLUS == 'TRUE'):
		SHUNT ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
	else:
		SHUNT ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.050000000000000044})
elif(INTUBATION == 'ESOPHAGEAL'):
	if(PULMEMBOLUS == 'TRUE'):
		SHUNT ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
	else:
		SHUNT ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.050000000000000044})
else:
	if(PULMEMBOLUS == 'TRUE'):
		SHUNT ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
	else:
		SHUNT ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})


if (HYPOVOLEMIA == 'TRUE'):
	if (LVFAILURE == 'TRUE'):
		STROKEVOLUME ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		STROKEVOLUME ~= choice({'LOW' : 0.5, 'NORMAL' : 0.49, 'HIGH' : 0.010000000000000009})
else:
	if(LVFAILURE == 'TRUE'):
		STROKEVOLUME ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
	else:
		STROKEVOLUME ~= choice({'LOW' : 0.05, 'NORMAL' : 0.9, 'HIGH' : 0.04999999999999993})


if (ANAPHYLAXIS == 'TRUE'):
	TPR ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
else:
	TPR ~= choice({'LOW' : 0.3, 'NORMAL' : 0.4, 'HIGH' : 0.30000000000000004})


if (MINVOLSET == 'LOW'):
	VENTMACH ~= choice({'ZERO' : 0.05, 'LOW' : 0.93, 'NORMAL' : 0.01, 'HIGH' : 0.009999999999999898})
elif(MINVOLSET == 'NORMAL'):
	VENTMACH ~= choice({'ZERO' : 0.05, 'LOW' : 0.01, 'NORMAL' : 0.93, 'HIGH' : 0.009999999999999898})
else:
	VENTMACH ~= choice({'ZERO' : 0.05, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.9299999999999999})


if (DISCONNECT == 'TRUE'):
	if (VENTMACH == 'ZERO'):
		VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTMACH == 'LOW'):
		VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTMACH == 'NORMAL'):
		VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		VENTTUBE ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
else:
	if(VENTMACH == 'ZERO'):
		VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTMACH == 'LOW'):
		VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTMACH == 'NORMAL'):
		VENTTUBE ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		VENTTUBE ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if (LVFAILURE == 'TRUE'):
	HISTORY ~= choice({'TRUE' : 0.9, 'FALSE' : 0.09999999999999998})
else:
	HISTORY ~= choice({'TRUE' : 0.01, 'FALSE' : 0.99})


if (HYPOVOLEMIA == 'TRUE'):
	if (LVFAILURE == 'TRUE'):
		LVEDVOLUME ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
	else:
		LVEDVOLUME ~= choice({'LOW' : 0.01, 'NORMAL' : 0.09, 'HIGH' : 0.9})
else:
	if(LVFAILURE == 'TRUE'):
		LVEDVOLUME ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		LVEDVOLUME ~= choice({'LOW' : 0.05, 'NORMAL' : 0.9, 'HIGH' : 0.04999999999999993})


if (PULMEMBOLUS == 'TRUE'):
	PAP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.19, 'HIGH' : 0.8})
else:
	PAP ~= choice({'LOW' : 0.05, 'NORMAL' : 0.9, 'HIGH' : 0.04999999999999993})


if (LVEDVOLUME == 'LOW'):
	PCWP ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
elif(LVEDVOLUME == 'NORMAL'):
	PCWP ~= choice({'LOW' : 0.04, 'NORMAL' : 0.95, 'HIGH' : 0.010000000000000009})
else:
	PCWP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.04, 'HIGH' : 0.95})


if (INTUBATION == 'NORMAL'):
	if (KINKEDTUBE == 'TRUE'):
		if (VENTTUBE == 'ZERO'):
			PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'LOW'):
			PRESS ~= choice({'ZERO' : 0.05, 'LOW' : 0.25, 'NORMAL' : 0.25, 'HIGH' : 0.44999999999999996})
		elif(VENTTUBE == 'NORMAL'):
			PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		else:
			PRESS ~= choice({'ZERO' : 0.2, 'LOW' : 0.75, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
	else:
		if(VENTTUBE == 'ZERO'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
		elif(VENTTUBE == 'LOW'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.29, 'NORMAL' : 0.3, 'HIGH' : 0.4})
		elif(VENTTUBE == 'NORMAL'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
		else:
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.9, 'NORMAL' : 0.08, 'HIGH' : 0.010000000000000009})
elif(INTUBATION == 'ESOPHAGEAL'):
	if(KINKEDTUBE == 'TRUE'):
		if(VENTTUBE == 'ZERO'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.3, 'NORMAL' : 0.49, 'HIGH' : 0.19999999999999996})
		elif(VENTTUBE == 'LOW'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.15, 'NORMAL' : 0.25, 'HIGH' : 0.59})
		elif(VENTTUBE == 'NORMAL'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		else:
			PRESS ~= choice({'ZERO' : 0.2, 'LOW' : 0.7, 'NORMAL' : 0.09, 'HIGH' : 0.01000000000000012})
	else:
		if(VENTTUBE == 'ZERO'):
			PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'LOW'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.08, 'HIGH' : 0.9})
		elif(VENTTUBE == 'NORMAL'):
			PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		else:
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.38, 'HIGH' : 0.6})
else:
	if(KINKEDTUBE == 'TRUE'):
		if(VENTTUBE == 'ZERO'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.08, 'HIGH' : 0.9})
		elif(VENTTUBE == 'LOW'):
			PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'NORMAL'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
		else:
			PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		if(VENTTUBE == 'ZERO'):
			PRESS ~= choice({'ZERO' : 0.1, 'LOW' : 0.84, 'NORMAL' : 0.05, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'LOW'):
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
		elif(VENTTUBE == 'NORMAL'):
			PRESS ~= choice({'ZERO' : 0.4, 'LOW' : 0.58, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		else:
			PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if (INTUBATION == 'NORMAL'):
	if (KINKEDTUBE == 'TRUE'):
		if (VENTTUBE == 'ZERO'):
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'LOW'):
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'NORMAL'):
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		else:
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		if(VENTTUBE == 'ZERO'):
			VENTLUNG ~= choice({'ZERO' : 0.3, 'LOW' : 0.68, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'LOW'):
			VENTLUNG ~= choice({'ZERO' : 0.95, 'LOW' : 0.03, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'NORMAL'):
			VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
		else:
			VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
elif(INTUBATION == 'ESOPHAGEAL'):
	if(KINKEDTUBE == 'TRUE'):
		if(VENTTUBE == 'ZERO'):
			VENTLUNG ~= choice({'ZERO' : 0.95, 'LOW' : 0.03, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'LOW'):
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'NORMAL'):
			VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		else:
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		if(VENTTUBE == 'ZERO'):
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'LOW'):
			VENTLUNG ~= choice({'ZERO' : 0.5, 'LOW' : 0.48, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'NORMAL'):
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		else:
			VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
else:
	if(KINKEDTUBE == 'TRUE'):
		if(VENTTUBE == 'ZERO'):
			VENTLUNG ~= choice({'ZERO' : 0.4, 'LOW' : 0.58, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'LOW'):
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'NORMAL'):
			VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
		else:
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		if(VENTTUBE == 'ZERO'):
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'LOW'):
			VENTLUNG ~= choice({'ZERO' : 0.3, 'LOW' : 0.68, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		elif(VENTTUBE == 'NORMAL'):
			VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
		else:
			VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if (LVEDVOLUME == 'LOW'):
	CVP ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
elif(LVEDVOLUME == 'NORMAL'):
	CVP ~= choice({'LOW' : 0.04, 'NORMAL' : 0.95, 'HIGH' : 0.010000000000000009})
else:
	CVP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.29, 'HIGH' : 0.7})


if (INTUBATION == 'NORMAL'):
	if (VENTLUNG == 'ZERO'):
		MINVOL ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'LOW'):
		MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
	elif(VENTLUNG == 'NORMAL'):
		MINVOL ~= choice({'ZERO' : 0.5, 'LOW' : 0.48, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
elif(INTUBATION == 'ESOPHAGEAL'):
	if(VENTLUNG == 'ZERO'):
		MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'LOW'):
		MINVOL ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'NORMAL'):
		MINVOL ~= choice({'ZERO' : 0.5, 'LOW' : 0.48, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
else:
	if(VENTLUNG == 'ZERO'):
		MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'LOW'):
		MINVOL ~= choice({'ZERO' : 0.6, 'LOW' : 0.38, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'NORMAL'):
		MINVOL ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if (INTUBATION == 'NORMAL'):
	if (VENTLUNG == 'ZERO'):
		VENTALV ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'LOW'):
		VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
	elif(VENTLUNG == 'NORMAL'):
		VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
	else:
		VENTALV ~= choice({'ZERO' : 0.03, 'LOW' : 0.95, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
elif(INTUBATION == 'ESOPHAGEAL'):
	if(VENTLUNG == 'ZERO'):
		VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'LOW'):
		VENTALV ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'NORMAL'):
		VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
	else:
		VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.94, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
else:
	if(VENTLUNG == 'ZERO'):
		VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'LOW'):
		VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'NORMAL'):
		VENTALV ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.88, 'NORMAL' : 0.1, 'HIGH' : 0.010000000000000009})


if (VENTALV == 'ZERO'):
	ARTCO2 ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})
elif(VENTALV == 'LOW'):
	ARTCO2 ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})
elif(VENTALV == 'NORMAL'):
	ARTCO2 ~= choice({'LOW' : 0.04, 'NORMAL' : 0.92, 'HIGH' : 0.039999999999999925})
else:
	ARTCO2 ~= choice({'LOW' : 0.9, 'NORMAL' : 0.09, 'HIGH' : 0.010000000000000009})


if (ARTCO2 == 'LOW'):
	if (VENTLUNG == 'ZERO'):
		EXPCO2 ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'LOW'):
		EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'NORMAL'):
		EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
	else:
		EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif(ARTCO2 == 'NORMAL'):
	if(VENTLUNG == 'ZERO'):
		EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'LOW'):
		EXPCO2 ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'NORMAL'):
		EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
	else:
		EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
else:
	if(VENTLUNG == 'ZERO'):
		EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'LOW'):
		EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.010000000000000009})
	elif(VENTLUNG == 'NORMAL'):
		EXPCO2 ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if (FIO2 == 'LOW'):
	if (VENTALV == 'ZERO'):
		PVSAT ~= choice({'LOW' : 1.0, 'NORMAL' : 0.0, 'HIGH' : 0.0})
	elif(VENTALV == 'LOW'):
		PVSAT ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
	elif(VENTALV == 'NORMAL'):
		PVSAT ~= choice({'LOW' : 1.0, 'NORMAL' : 0.0, 'HIGH' : 0.0})
	else:
		PVSAT ~= choice({'LOW' : 0.01, 'NORMAL' : 0.95, 'HIGH' : 0.040000000000000036})
else:
	if(VENTALV == 'ZERO'):
		PVSAT ~= choice({'LOW' : 0.99, 'NORMAL' : 0.01, 'HIGH' : 0.0})
	elif(VENTALV == 'LOW'):
		PVSAT ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
	elif(VENTALV == 'NORMAL'):
		PVSAT ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
	else:
		PVSAT ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if (PVSAT == 'LOW'):
	if (SHUNT == 'NORMAL'):
		SAO2 ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		SAO2 ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
elif(PVSAT == 'NORMAL'):
	if(SHUNT == 'NORMAL'):
		SAO2 ~= choice({'LOW' : 0.01, 'NORMAL' : 0.98, 'HIGH' : 0.010000000000000009})
	else:
		SAO2 ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
else:
	if(SHUNT == 'NORMAL'):
		SAO2 ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})
	else:
		SAO2 ~= choice({'LOW' : 0.69, 'NORMAL' : 0.3, 'HIGH' : 0.010000000000000009})


if (ARTCO2 == 'LOW'):
	if (INSUFFANESTH == 'TRUE'):
		if (SAO2 == 'LOW'):
			if (TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.30000000000000004})
		elif(SAO2 == 'NORMAL'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.30000000000000004})
		else:
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.050000000000000044})
	else:
		if(SAO2 == 'LOW'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.30000000000000004})
		elif(SAO2 == 'NORMAL'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.050000000000000044})
		else:
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.050000000000000044})
elif(ARTCO2 == 'NORMAL'):
	if(INSUFFANESTH == 'TRUE'):
		if(SAO2 == 'LOW'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.30000000000000004})
		elif(SAO2 == 'NORMAL'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.30000000000000004})
		else:
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.99, 'HIGH' : 0.010000000000000009})
	else:
		if(SAO2 == 'LOW'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.30000000000000004})
		elif(SAO2 == 'NORMAL'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.99, 'HIGH' : 0.010000000000000009})
		else:
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.99, 'HIGH' : 0.010000000000000009})
else:
	if(INSUFFANESTH == 'TRUE'):
		if(SAO2 == 'LOW'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
		elif(SAO2 == 'NORMAL'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
		else:
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.3, 'HIGH' : 0.7})
	else:
		if(SAO2 == 'LOW'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
		elif(SAO2 == 'NORMAL'):
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.3, 'HIGH' : 0.7})
		else:
			if(TPR == 'LOW'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			elif(TPR == 'NORMAL'):
				CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
			else:
				CATECHOL ~= choice({'NORMAL' : 0.3, 'HIGH' : 0.7})


if (CATECHOL == 'NORMAL'):
	HR ~= choice({'LOW' : 0.05, 'NORMAL' : 0.9, 'HIGH' : 0.04999999999999993})
else:
	HR ~= choice({'LOW' : 0.01, 'NORMAL' : 0.09, 'HIGH' : 0.9})


if (ERRLOWOUTPUT == 'TRUE'):
	if (HR == 'LOW'):
		HRBP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(HR == 'NORMAL'):
		HRBP ~= choice({'LOW' : 0.3, 'NORMAL' : 0.4, 'HIGH' : 0.30000000000000004})
	else:
		HRBP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.98, 'HIGH' : 0.010000000000000009})
else:
	if(HR == 'LOW'):
		HRBP ~= choice({'LOW' : 0.4, 'NORMAL' : 0.59, 'HIGH' : 0.010000000000000009})
	elif(HR == 'NORMAL'):
		HRBP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		HRBP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if (ERRCAUTER == 'TRUE'):
	if (HR == 'LOW'):
		HREKG ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333334})
	elif(HR == 'NORMAL'):
		HREKG ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333334})
	else:
		HREKG ~= choice({'LOW' : 0.01, 'NORMAL' : 0.98, 'HIGH' : 0.010000000000000009})
else:
	if(HR == 'LOW'):
		HREKG ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333334})
	elif(HR == 'NORMAL'):
		HREKG ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		HREKG ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if (ERRCAUTER == 'TRUE'):
	if (HR == 'LOW'):
		HRSAT ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333334})
	elif(HR == 'NORMAL'):
		HRSAT ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333334})
	else:
		HRSAT ~= choice({'LOW' : 0.01, 'NORMAL' : 0.98, 'HIGH' : 0.010000000000000009})
else:
	if(HR == 'LOW'):
		HRSAT ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333334})
	elif(HR == 'NORMAL'):
		HRSAT ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		HRSAT ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if (HR == 'LOW'):
	if (STROKEVOLUME == 'LOW'):
		CO ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(STROKEVOLUME == 'NORMAL'):
		CO ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
	else:
		CO ~= choice({'LOW' : 0.3, 'NORMAL' : 0.69, 'HIGH' : 0.010000000000000009})
elif(HR == 'NORMAL'):
	if(STROKEVOLUME == 'LOW'):
		CO ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.010000000000000009})
	elif(STROKEVOLUME == 'NORMAL'):
		CO ~= choice({'LOW' : 0.04, 'NORMAL' : 0.95, 'HIGH' : 0.010000000000000009})
	else:
		CO ~= choice({'LOW' : 0.01, 'NORMAL' : 0.3, 'HIGH' : 0.69})
else:
	if(STROKEVOLUME == 'LOW'):
		CO ~= choice({'LOW' : 0.8, 'NORMAL' : 0.19, 'HIGH' : 0.010000000000000009})
	elif(STROKEVOLUME == 'NORMAL'):
		CO ~= choice({'LOW' : 0.01, 'NORMAL' : 0.04, 'HIGH' : 0.95})
	else:
		CO ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if (CO == 'LOW'):
	if (TPR == 'LOW'):
		BP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(TPR == 'NORMAL'):
		BP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	else:
		BP ~= choice({'LOW' : 0.3, 'NORMAL' : 0.6, 'HIGH' : 0.10000000000000009})
elif(CO == 'NORMAL'):
	if(TPR == 'LOW'):
		BP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.010000000000000009})
	elif(TPR == 'NORMAL'):
		BP ~= choice({'LOW' : 0.1, 'NORMAL' : 0.85, 'HIGH' : 0.050000000000000044})
	else:
		BP ~= choice({'LOW' : 0.05, 'NORMAL' : 0.4, 'HIGH' : 0.55})
else:
	if(TPR == 'LOW'):
		BP ~= choice({'LOW' : 0.9, 'NORMAL' : 0.09, 'HIGH' : 0.010000000000000009})
	elif(TPR == 'NORMAL'):
		BP ~= choice({'LOW' : 0.05, 'NORMAL' : 0.2, 'HIGH' : 0.75})
	else:
		BP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.09, 'HIGH' : 0.9})


'''
compiler = SPPL_Compiler(data)
namespace = compiler.execute_module()
model=namespace.model
ANAPHYLAXIS = Id('ANAPHYLAXIS')
ARTCO2 = Id('ARTCO2')
BP = Id('BP')
CATECHOL = Id('CATECHOL')
CO = Id('CO')
CVP = Id('CVP')
DISCONNECT = Id('DISCONNECT')
ERRCAUTER = Id('ERRCAUTER')
ERRLOWOUTPUT = Id('ERRLOWOUTPUT')
EXPCO2 = Id('EXPCO2')
FIO2 = Id('FIO2')
HISTORY = Id('HISTORY')
HR = Id('HR')
HRBP = Id('HRBP')
HREKG = Id('HREKG')
HRSAT = Id('HRSAT')
HYPOVOLEMIA = Id('HYPOVOLEMIA')
INSUFFANESTH = Id('INSUFFANESTH')
INTUBATION = Id('INTUBATION')
KINKEDTUBE = Id('KINKEDTUBE')
LVEDVOLUME = Id('LVEDVOLUME')
LVFAILURE = Id('LVFAILURE')
MINVOL = Id('MINVOL')
MINVOLSET = Id('MINVOLSET')
PAP = Id('PAP')
PCWP = Id('PCWP')
PRESS = Id('PRESS')
PULMEMBOLUS = Id('PULMEMBOLUS')
PVSAT = Id('PVSAT')
SAO2 = Id('SAO2')
SHUNT = Id('SHUNT')
STROKEVOLUME = Id('STROKEVOLUME')
TPR = Id('TPR')
VENTALV = Id('VENTALV')
VENTLUNG = Id('VENTLUNG')
VENTMACH = Id('VENTMACH')
VENTTUBE = Id('VENTTUBE')
events = [VENTLUNG << {'LOW'},CATECHOL << {'HIGH'},PRESS << {'ZERO'},ARTCO2 << {'NORMAL'},VENTALV << {'ZERO'},INSUFFANESTH << {'FALSE'},HYPOVOLEMIA << {'TRUE'},LVFAILURE << {'FALSE'},SHUNT << {'HIGH'},PRESS << {'HIGH'},VENTMACH << {'HIGH'},PRESS << {'HIGH'},HYPOVOLEMIA << {'TRUE'},VENTTUBE << {'LOW'},DISCONNECT << {'TRUE'},BP << {'HIGH'},PVSAT << {'LOW'},HISTORY << {'TRUE'},BP << {'NORMAL'},HREKG << {'NORMAL'},SHUNT << {'HIGH'},INTUBATION << {'ESOPHAGEAL'},PRESS << {'ZERO'},INSUFFANESTH << {'TRUE'},VENTLUNG << {'LOW'},CATECHOL << {'HIGH'},HISTORY << {'FALSE'},PCWP << {'NORMAL'},BP << {'HIGH'},HR << {'LOW'},MINVOL << {'LOW'},INTUBATION << {'ONESIDED'},CATECHOL << {'NORMAL'},LVFAILURE << {'FALSE'},MINVOLSET << {'NORMAL'},MINVOL << {'NORMAL'},DISCONNECT << {'TRUE'},VENTALV << {'HIGH'},CATECHOL << {'HIGH'},ANAPHYLAXIS << {'TRUE'},ERRLOWOUTPUT << {'FALSE'},HR << {'LOW'},HISTORY << {'TRUE'},MINVOLSET << {'LOW'},HISTORY << {'TRUE'},INSUFFANESTH << {'FALSE'},ERRCAUTER << {'FALSE'},CATECHOL << {'HIGH'},PCWP << {'NORMAL'},PRESS << {'ZERO'},(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'HIGH'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'LOW'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'HIGH'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'HIGH'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'HIGH'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'HIGH'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'LOW'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'NORMAL'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'HIGH'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'HIGH'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'HIGH'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'LOW'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'HIGH'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'NORMAL'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'LOW'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'HIGH'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'LOW'}) & (PAP << {'NORMAL'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'HIGH'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'NORMAL'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'NORMAL'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'HIGH'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'LOW'}) & (PAP << {'HIGH'}) & (PCWP << {'HIGH'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'LOW'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'HIGH'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'HIGH'})]
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
