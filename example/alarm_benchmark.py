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


if ((INTUBATION == 'NORMAL') and (PULMEMBOLUS == 'TRUE')):
	SHUNT ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
elif ((INTUBATION == 'NORMAL') and (PULMEMBOLUS == 'FALSE')):
	SHUNT ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.05})
elif ((INTUBATION == 'ESOPHAGEAL') and (PULMEMBOLUS == 'TRUE')):
	SHUNT ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
elif ((INTUBATION == 'ESOPHAGEAL') and (PULMEMBOLUS == 'FALSE')):
	SHUNT ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.05})
elif ((INTUBATION == 'ONESIDED') and (PULMEMBOLUS == 'TRUE')):
	SHUNT ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
else:
	SHUNT ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})


if ((HYPOVOLEMIA == 'TRUE') and (LVFAILURE == 'TRUE')):
	STROKEVOLUME ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((HYPOVOLEMIA == 'TRUE') and (LVFAILURE == 'FALSE')):
	STROKEVOLUME ~= choice({'LOW' : 0.5, 'NORMAL' : 0.49, 'HIGH' : 0.01})
elif ((HYPOVOLEMIA == 'FALSE') and (LVFAILURE == 'TRUE')):
	STROKEVOLUME ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.01})
else:
	STROKEVOLUME ~= choice({'LOW' : 0.05, 'NORMAL' : 0.9, 'HIGH' : 0.05})


if ((ANAPHYLAXIS == 'TRUE')):
	TPR ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	TPR ~= choice({'LOW' : 0.3, 'NORMAL' : 0.4, 'HIGH' : 0.3})


if ((MINVOLSET == 'LOW')):
	VENTMACH ~= choice({'ZERO' : 0.05, 'LOW' : 0.93, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((MINVOLSET == 'NORMAL')):
	VENTMACH ~= choice({'ZERO' : 0.05, 'LOW' : 0.01, 'NORMAL' : 0.93, 'HIGH' : 0.01})
else:
	VENTMACH ~= choice({'ZERO' : 0.05, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.93})


if ((DISCONNECT == 'TRUE') and (VENTMACH == 'ZERO')):
	VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((DISCONNECT == 'TRUE') and (VENTMACH == 'LOW')):
	VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((DISCONNECT == 'TRUE') and (VENTMACH == 'NORMAL')):
	VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((DISCONNECT == 'TRUE') and (VENTMACH == 'HIGH')):
	VENTTUBE ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((DISCONNECT == 'FALSE') and (VENTMACH == 'ZERO')):
	VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((DISCONNECT == 'FALSE') and (VENTMACH == 'LOW')):
	VENTTUBE ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((DISCONNECT == 'FALSE') and (VENTMACH == 'NORMAL')):
	VENTTUBE ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	VENTTUBE ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if ((LVFAILURE == 'TRUE')):
	HISTORY ~= choice({'TRUE' : 0.9, 'FALSE' : 0.1})
else:
	HISTORY ~= choice({'TRUE' : 0.01, 'FALSE' : 0.99})


if ((HYPOVOLEMIA == 'TRUE') and (LVFAILURE == 'TRUE')):
	LVEDVOLUME ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.01})
elif ((HYPOVOLEMIA == 'TRUE') and (LVFAILURE == 'FALSE')):
	LVEDVOLUME ~= choice({'LOW' : 0.01, 'NORMAL' : 0.09, 'HIGH' : 0.9})
elif ((HYPOVOLEMIA == 'FALSE') and (LVFAILURE == 'TRUE')):
	LVEDVOLUME ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	LVEDVOLUME ~= choice({'LOW' : 0.05, 'NORMAL' : 0.9, 'HIGH' : 0.05})


if ((PULMEMBOLUS == 'TRUE')):
	PAP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.19, 'HIGH' : 0.8})
else:
	PAP ~= choice({'LOW' : 0.05, 'NORMAL' : 0.9, 'HIGH' : 0.05})


if ((LVEDVOLUME == 'LOW')):
	PCWP ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.01})
elif ((LVEDVOLUME == 'NORMAL')):
	PCWP ~= choice({'LOW' : 0.04, 'NORMAL' : 0.95, 'HIGH' : 0.01})
else:
	PCWP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.04, 'HIGH' : 0.95})


if ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'ZERO')):
	PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'LOW')):
	PRESS ~= choice({'ZERO' : 0.05, 'LOW' : 0.25, 'NORMAL' : 0.25, 'HIGH' : 0.45})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'NORMAL')):
	PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'HIGH')):
	PRESS ~= choice({'ZERO' : 0.2, 'LOW' : 0.75, 'NORMAL' : 0.04, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'ZERO')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'LOW')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.29, 'NORMAL' : 0.3, 'HIGH' : 0.4})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'NORMAL')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'HIGH')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.9, 'NORMAL' : 0.08, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'ZERO')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.3, 'NORMAL' : 0.49, 'HIGH' : 0.2})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'LOW')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.15, 'NORMAL' : 0.25, 'HIGH' : 0.59})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'NORMAL')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'HIGH')):
	PRESS ~= choice({'ZERO' : 0.2, 'LOW' : 0.7, 'NORMAL' : 0.09, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'ZERO')):
	PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'LOW')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.08, 'HIGH' : 0.9})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'NORMAL')):
	PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'HIGH')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.38, 'HIGH' : 0.6})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'ZERO')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.08, 'HIGH' : 0.9})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'LOW')):
	PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'NORMAL')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'HIGH')):
	PRESS ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'ZERO')):
	PRESS ~= choice({'ZERO' : 0.1, 'LOW' : 0.84, 'NORMAL' : 0.05, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'LOW')):
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'NORMAL')):
	PRESS ~= choice({'ZERO' : 0.4, 'LOW' : 0.58, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	PRESS ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'ZERO')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'LOW')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'NORMAL')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'HIGH')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'ZERO')):
	VENTLUNG ~= choice({'ZERO' : 0.3, 'LOW' : 0.68, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'LOW')):
	VENTLUNG ~= choice({'ZERO' : 0.95, 'LOW' : 0.03, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'NORMAL')):
	VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif ((INTUBATION == 'NORMAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'HIGH')):
	VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'ZERO')):
	VENTLUNG ~= choice({'ZERO' : 0.95, 'LOW' : 0.03, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'LOW')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'NORMAL')):
	VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'HIGH')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'ZERO')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'LOW')):
	VENTLUNG ~= choice({'ZERO' : 0.5, 'LOW' : 0.48, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'NORMAL')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'HIGH')):
	VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'ZERO')):
	VENTLUNG ~= choice({'ZERO' : 0.4, 'LOW' : 0.58, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'LOW')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'NORMAL')):
	VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'TRUE') and (VENTTUBE == 'HIGH')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'ZERO')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'LOW')):
	VENTLUNG ~= choice({'ZERO' : 0.3, 'LOW' : 0.68, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (KINKEDTUBE == 'FALSE') and (VENTTUBE == 'NORMAL')):
	VENTLUNG ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	VENTLUNG ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if ((LVEDVOLUME == 'LOW')):
	CVP ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.01})
elif ((LVEDVOLUME == 'NORMAL')):
	CVP ~= choice({'LOW' : 0.04, 'NORMAL' : 0.95, 'HIGH' : 0.01})
else:
	CVP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.29, 'HIGH' : 0.7})


if ((INTUBATION == 'NORMAL') and (VENTLUNG == 'ZERO')):
	MINVOL ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (VENTLUNG == 'LOW')):
	MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif ((INTUBATION == 'NORMAL') and (VENTLUNG == 'NORMAL')):
	MINVOL ~= choice({'ZERO' : 0.5, 'LOW' : 0.48, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (VENTLUNG == 'HIGH')):
	MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (VENTLUNG == 'ZERO')):
	MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (VENTLUNG == 'LOW')):
	MINVOL ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (VENTLUNG == 'NORMAL')):
	MINVOL ~= choice({'ZERO' : 0.5, 'LOW' : 0.48, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (VENTLUNG == 'HIGH')):
	MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (VENTLUNG == 'ZERO')):
	MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (VENTLUNG == 'LOW')):
	MINVOL ~= choice({'ZERO' : 0.6, 'LOW' : 0.38, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (VENTLUNG == 'NORMAL')):
	MINVOL ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	MINVOL ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if ((INTUBATION == 'NORMAL') and (VENTLUNG == 'ZERO')):
	VENTALV ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (VENTLUNG == 'LOW')):
	VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif ((INTUBATION == 'NORMAL') and (VENTLUNG == 'NORMAL')):
	VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((INTUBATION == 'NORMAL') and (VENTLUNG == 'HIGH')):
	VENTALV ~= choice({'ZERO' : 0.03, 'LOW' : 0.95, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (VENTLUNG == 'ZERO')):
	VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (VENTLUNG == 'LOW')):
	VENTALV ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ESOPHAGEAL') and (VENTLUNG == 'NORMAL')):
	VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif ((INTUBATION == 'ESOPHAGEAL') and (VENTLUNG == 'HIGH')):
	VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.94, 'NORMAL' : 0.04, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (VENTLUNG == 'ZERO')):
	VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (VENTLUNG == 'LOW')):
	VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((INTUBATION == 'ONESIDED') and (VENTLUNG == 'NORMAL')):
	VENTALV ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	VENTALV ~= choice({'ZERO' : 0.01, 'LOW' : 0.88, 'NORMAL' : 0.1, 'HIGH' : 0.01})


if ((VENTALV == 'ZERO')):
	ARTCO2 ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})
elif ((VENTALV == 'LOW')):
	ARTCO2 ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})
elif ((VENTALV == 'NORMAL')):
	ARTCO2 ~= choice({'LOW' : 0.04, 'NORMAL' : 0.92, 'HIGH' : 0.04})
else:
	ARTCO2 ~= choice({'LOW' : 0.9, 'NORMAL' : 0.09, 'HIGH' : 0.01})


if ((ARTCO2 == 'LOW') and (VENTLUNG == 'ZERO')):
	EXPCO2 ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((ARTCO2 == 'LOW') and (VENTLUNG == 'LOW')):
	EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((ARTCO2 == 'LOW') and (VENTLUNG == 'NORMAL')):
	EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((ARTCO2 == 'LOW') and (VENTLUNG == 'HIGH')):
	EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif ((ARTCO2 == 'NORMAL') and (VENTLUNG == 'ZERO')):
	EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((ARTCO2 == 'NORMAL') and (VENTLUNG == 'LOW')):
	EXPCO2 ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((ARTCO2 == 'NORMAL') and (VENTLUNG == 'NORMAL')):
	EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((ARTCO2 == 'NORMAL') and (VENTLUNG == 'HIGH')):
	EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})
elif ((ARTCO2 == 'HIGH') and (VENTLUNG == 'ZERO')):
	EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.97, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((ARTCO2 == 'HIGH') and (VENTLUNG == 'LOW')):
	EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.97, 'HIGH' : 0.01})
elif ((ARTCO2 == 'HIGH') and (VENTLUNG == 'NORMAL')):
	EXPCO2 ~= choice({'ZERO' : 0.97, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	EXPCO2 ~= choice({'ZERO' : 0.01, 'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.97})


if ((FIO2 == 'LOW') and (VENTALV == 'ZERO')):
	PVSAT ~= choice({'LOW' : 1.0, 'NORMAL' : 0.0, 'HIGH' : 0.0})
elif ((FIO2 == 'LOW') and (VENTALV == 'LOW')):
	PVSAT ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.01})
elif ((FIO2 == 'LOW') and (VENTALV == 'NORMAL')):
	PVSAT ~= choice({'LOW' : 1.0, 'NORMAL' : 0.0, 'HIGH' : 0.0})
elif ((FIO2 == 'LOW') and (VENTALV == 'HIGH')):
	PVSAT ~= choice({'LOW' : 0.01, 'NORMAL' : 0.95, 'HIGH' : 0.04})
elif ((FIO2 == 'NORMAL') and (VENTALV == 'ZERO')):
	PVSAT ~= choice({'LOW' : 0.99, 'NORMAL' : 0.01, 'HIGH' : 0.0})
elif ((FIO2 == 'NORMAL') and (VENTALV == 'LOW')):
	PVSAT ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.01})
elif ((FIO2 == 'NORMAL') and (VENTALV == 'NORMAL')):
	PVSAT ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.01})
else:
	PVSAT ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if ((PVSAT == 'LOW') and (SHUNT == 'NORMAL')):
	SAO2 ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((PVSAT == 'LOW') and (SHUNT == 'HIGH')):
	SAO2 ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((PVSAT == 'NORMAL') and (SHUNT == 'NORMAL')):
	SAO2 ~= choice({'LOW' : 0.01, 'NORMAL' : 0.98, 'HIGH' : 0.01})
elif ((PVSAT == 'NORMAL') and (SHUNT == 'HIGH')):
	SAO2 ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((PVSAT == 'HIGH') and (SHUNT == 'NORMAL')):
	SAO2 ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})
else:
	SAO2 ~= choice({'LOW' : 0.69, 'NORMAL' : 0.3, 'HIGH' : 0.01})


if ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'LOW') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'LOW') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'LOW') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.3})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'NORMAL') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'NORMAL') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'NORMAL') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.3})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'HIGH') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'HIGH') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'HIGH') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.05})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'LOW') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'LOW') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'LOW') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.3})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'NORMAL') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'NORMAL') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'NORMAL') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.05})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'HIGH') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'HIGH') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'LOW') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'HIGH') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.95, 'HIGH' : 0.05})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'LOW') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'LOW') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'LOW') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.3})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'NORMAL') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'NORMAL') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'NORMAL') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.3})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'HIGH') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'HIGH') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'HIGH') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.99, 'HIGH' : 0.01})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'LOW') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'LOW') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'LOW') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.7, 'HIGH' : 0.3})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'NORMAL') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'NORMAL') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'NORMAL') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.99, 'HIGH' : 0.01})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'HIGH') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'HIGH') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.05, 'HIGH' : 0.95})
elif ((ARTCO2 == 'NORMAL') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'HIGH') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.99, 'HIGH' : 0.01})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'LOW') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'LOW') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'LOW') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'NORMAL') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'NORMAL') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'NORMAL') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'HIGH') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'HIGH') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'TRUE') and (SAO2 == 'HIGH') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.3, 'HIGH' : 0.7})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'LOW') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'LOW') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'LOW') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.1, 'HIGH' : 0.9})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'NORMAL') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'NORMAL') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'NORMAL') and (TPR == 'HIGH')):
	CATECHOL ~= choice({'NORMAL' : 0.3, 'HIGH' : 0.7})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'HIGH') and (TPR == 'LOW')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
elif ((ARTCO2 == 'HIGH') and (INSUFFANESTH == 'FALSE') and (SAO2 == 'HIGH') and (TPR == 'NORMAL')):
	CATECHOL ~= choice({'NORMAL' : 0.01, 'HIGH' : 0.99})
else:
	CATECHOL ~= choice({'NORMAL' : 0.3, 'HIGH' : 0.7})


if ((CATECHOL == 'NORMAL')):
	HR ~= choice({'LOW' : 0.05, 'NORMAL' : 0.9, 'HIGH' : 0.05})
else:
	HR ~= choice({'LOW' : 0.01, 'NORMAL' : 0.09, 'HIGH' : 0.9})


if ((ERRLOWOUTPUT == 'TRUE') and (HR == 'LOW')):
	HRBP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((ERRLOWOUTPUT == 'TRUE') and (HR == 'NORMAL')):
	HRBP ~= choice({'LOW' : 0.3, 'NORMAL' : 0.4, 'HIGH' : 0.3})
elif ((ERRLOWOUTPUT == 'TRUE') and (HR == 'HIGH')):
	HRBP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.98, 'HIGH' : 0.01})
elif ((ERRLOWOUTPUT == 'FALSE') and (HR == 'LOW')):
	HRBP ~= choice({'LOW' : 0.4, 'NORMAL' : 0.59, 'HIGH' : 0.01})
elif ((ERRLOWOUTPUT == 'FALSE') and (HR == 'NORMAL')):
	HRBP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	HRBP ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if ((ERRCAUTER == 'TRUE') and (HR == 'LOW')):
	HREKG ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333333})
elif ((ERRCAUTER == 'TRUE') and (HR == 'NORMAL')):
	HREKG ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333333})
elif ((ERRCAUTER == 'TRUE') and (HR == 'HIGH')):
	HREKG ~= choice({'LOW' : 0.01, 'NORMAL' : 0.98, 'HIGH' : 0.01})
elif ((ERRCAUTER == 'FALSE') and (HR == 'LOW')):
	HREKG ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333333})
elif ((ERRCAUTER == 'FALSE') and (HR == 'NORMAL')):
	HREKG ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	HREKG ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if ((ERRCAUTER == 'TRUE') and (HR == 'LOW')):
	HRSAT ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333333})
elif ((ERRCAUTER == 'TRUE') and (HR == 'NORMAL')):
	HRSAT ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333333})
elif ((ERRCAUTER == 'TRUE') and (HR == 'HIGH')):
	HRSAT ~= choice({'LOW' : 0.01, 'NORMAL' : 0.98, 'HIGH' : 0.01})
elif ((ERRCAUTER == 'FALSE') and (HR == 'LOW')):
	HRSAT ~= choice({'LOW' : 0.3333333, 'NORMAL' : 0.3333333, 'HIGH' : 0.3333333})
elif ((ERRCAUTER == 'FALSE') and (HR == 'NORMAL')):
	HRSAT ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
else:
	HRSAT ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if ((HR == 'LOW') and (STROKEVOLUME == 'LOW')):
	CO ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((HR == 'LOW') and (STROKEVOLUME == 'NORMAL')):
	CO ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.01})
elif ((HR == 'LOW') and (STROKEVOLUME == 'HIGH')):
	CO ~= choice({'LOW' : 0.3, 'NORMAL' : 0.69, 'HIGH' : 0.01})
elif ((HR == 'NORMAL') and (STROKEVOLUME == 'LOW')):
	CO ~= choice({'LOW' : 0.95, 'NORMAL' : 0.04, 'HIGH' : 0.01})
elif ((HR == 'NORMAL') and (STROKEVOLUME == 'NORMAL')):
	CO ~= choice({'LOW' : 0.04, 'NORMAL' : 0.95, 'HIGH' : 0.01})
elif ((HR == 'NORMAL') and (STROKEVOLUME == 'HIGH')):
	CO ~= choice({'LOW' : 0.01, 'NORMAL' : 0.3, 'HIGH' : 0.69})
elif ((HR == 'HIGH') and (STROKEVOLUME == 'LOW')):
	CO ~= choice({'LOW' : 0.8, 'NORMAL' : 0.19, 'HIGH' : 0.01})
elif ((HR == 'HIGH') and (STROKEVOLUME == 'NORMAL')):
	CO ~= choice({'LOW' : 0.01, 'NORMAL' : 0.04, 'HIGH' : 0.95})
else:
	CO ~= choice({'LOW' : 0.01, 'NORMAL' : 0.01, 'HIGH' : 0.98})


if ((CO == 'LOW') and (TPR == 'LOW')):
	BP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((CO == 'LOW') and (TPR == 'NORMAL')):
	BP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((CO == 'LOW') and (TPR == 'HIGH')):
	BP ~= choice({'LOW' : 0.3, 'NORMAL' : 0.6, 'HIGH' : 0.1})
elif ((CO == 'NORMAL') and (TPR == 'LOW')):
	BP ~= choice({'LOW' : 0.98, 'NORMAL' : 0.01, 'HIGH' : 0.01})
elif ((CO == 'NORMAL') and (TPR == 'NORMAL')):
	BP ~= choice({'LOW' : 0.1, 'NORMAL' : 0.85, 'HIGH' : 0.05})
elif ((CO == 'NORMAL') and (TPR == 'HIGH')):
	BP ~= choice({'LOW' : 0.05, 'NORMAL' : 0.4, 'HIGH' : 0.55})
elif ((CO == 'HIGH') and (TPR == 'LOW')):
	BP ~= choice({'LOW' : 0.9, 'NORMAL' : 0.09, 'HIGH' : 0.01})
elif ((CO == 'HIGH') and (TPR == 'NORMAL')):
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
events = [CATECHOL << {'HIGH'},ANAPHYLAXIS << {'FALSE'},LVEDVOLUME << {'NORMAL'},PAP << {'LOW'},BP << {'LOW'},ERRCAUTER << {'TRUE'},STROKEVOLUME << {'LOW'},HISTORY << {'TRUE'},ERRLOWOUTPUT << {'FALSE'},PRESS << {'LOW'},ANAPHYLAXIS << {'TRUE'},LVEDVOLUME << {'NORMAL'},STROKEVOLUME << {'NORMAL'},CVP << {'HIGH'},VENTTUBE << {'HIGH'},CO << {'NORMAL'},ANAPHYLAXIS << {'TRUE'},SHUNT << {'NORMAL'},HRSAT << {'NORMAL'},HR << {'LOW'},VENTMACH << {'ZERO'},ERRCAUTER << {'FALSE'},PAP << {'LOW'},SHUNT << {'HIGH'},MINVOL << {'HIGH'},PRESS << {'HIGH'},HYPOVOLEMIA << {'FALSE'},VENTMACH << {'LOW'},CATECHOL << {'HIGH'},SHUNT << {'NORMAL'},CATECHOL << {'HIGH'},HRSAT << {'HIGH'},ERRCAUTER << {'TRUE'},TPR << {'LOW'},FIO2 << {'NORMAL'},CATECHOL << {'HIGH'},HRBP << {'HIGH'},LVFAILURE << {'FALSE'},PCWP << {'HIGH'},STROKEVOLUME << {'LOW'},SHUNT << {'HIGH'},ANAPHYLAXIS << {'TRUE'},VENTALV << {'LOW'},PCWP << {'NORMAL'},CO << {'NORMAL'},HISTORY << {'FALSE'},ANAPHYLAXIS << {'TRUE'},MINVOL << {'HIGH'},LVFAILURE << {'TRUE'},HREKG << {'HIGH'},(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'HIGH'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'HIGH'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'HIGH'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'LOW'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'HIGH'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'LOW'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'HIGH'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'HIGH'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'LOW'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'HIGH'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'HIGH'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'LOW'}) & (PAP << {'NORMAL'}) & (PCWP << {'NORMAL'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'HIGH'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'NORMAL'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'NORMAL'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'LOW'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'HIGH'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'ZERO'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'HIGH'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'TRUE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'NORMAL'}) & (PRESS << {'HIGH'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'NORMAL'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'LOW'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'HIGH'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'NORMAL'}) & (VENTALV << {'LOW'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'NORMAL'}) & (CO << {'HIGH'}) & (CVP << {'LOW'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'NORMAL'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'LOW'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'HIGH'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'ZERO'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'LOW'}) & (BP << {'LOW'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'LOW'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'HIGH'}) & (PCWP << {'NORMAL'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'NORMAL'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'ZERO'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'LOW'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'NORMAL'}) & (FIO2 << {'LOW'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'LOW'}) & (VENTALV << {'ZERO'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'NORMAL'}),(ANAPHYLAXIS << {'TRUE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'LOW'}) & (CVP << {'HIGH'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'HIGH'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'FALSE'}) & (HR << {'NORMAL'}) & (HRBP << {'LOW'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'FALSE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'LOW'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'NORMAL'}) & (PCWP << {'LOW'}) & (PRESS << {'LOW'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'NORMAL'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'HIGH'}) & (TPR << {'HIGH'}) & (VENTALV << {'NORMAL'}) & (VENTLUNG << {'HIGH'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'HIGH'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'HIGH'}) & (BP << {'HIGH'}) & (CATECHOL << {'NORMAL'}) & (CO << {'NORMAL'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'FALSE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'LOW'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'HIGH'}) & (HRBP << {'HIGH'}) & (HREKG << {'NORMAL'}) & (HRSAT << {'LOW'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ESOPHAGEAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'FALSE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'LOW'}) & (PCWP << {'LOW'}) & (PRESS << {'NORMAL'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'LOW'}) & (SHUNT << {'NORMAL'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'LOW'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'HIGH'}) & (VENTTUBE << {'LOW'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'LOW'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'TRUE'}) & (ERRLOWOUTPUT << {'TRUE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'NORMAL'}) & (HISTORY << {'TRUE'}) & (HR << {'LOW'}) & (HRBP << {'LOW'}) & (HREKG << {'HIGH'}) & (HRSAT << {'HIGH'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'ONESIDED'}) & (KINKEDTUBE << {'TRUE'}) & (LVEDVOLUME << {'HIGH'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'HIGH'}) & (MINVOLSET << {'NORMAL'}) & (PAP << {'NORMAL'}) & (PCWP << {'HIGH'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'FALSE'}) & (PVSAT << {'NORMAL'}) & (SAO2 << {'LOW'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'NORMAL'}) & (TPR << {'NORMAL'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'LOW'}) & (VENTMACH << {'NORMAL'}) & (VENTTUBE << {'ZERO'}),(ANAPHYLAXIS << {'FALSE'}) & (ARTCO2 << {'NORMAL'}) & (BP << {'NORMAL'}) & (CATECHOL << {'HIGH'}) & (CO << {'HIGH'}) & (CVP << {'NORMAL'}) & (DISCONNECT << {'TRUE'}) & (ERRCAUTER << {'FALSE'}) & (ERRLOWOUTPUT << {'FALSE'}) & (EXPCO2 << {'ZERO'}) & (FIO2 << {'LOW'}) & (HISTORY << {'FALSE'}) & (HR << {'LOW'}) & (HRBP << {'HIGH'}) & (HREKG << {'HIGH'}) & (HRSAT << {'NORMAL'}) & (HYPOVOLEMIA << {'FALSE'}) & (INSUFFANESTH << {'TRUE'}) & (INTUBATION << {'NORMAL'}) & (KINKEDTUBE << {'FALSE'}) & (LVEDVOLUME << {'NORMAL'}) & (LVFAILURE << {'TRUE'}) & (MINVOL << {'NORMAL'}) & (MINVOLSET << {'HIGH'}) & (PAP << {'HIGH'}) & (PCWP << {'LOW'}) & (PRESS << {'ZERO'}) & (PULMEMBOLUS << {'TRUE'}) & (PVSAT << {'HIGH'}) & (SAO2 << {'HIGH'}) & (SHUNT << {'HIGH'}) & (STROKEVOLUME << {'LOW'}) & (TPR << {'HIGH'}) & (VENTALV << {'HIGH'}) & (VENTLUNG << {'NORMAL'}) & (VENTMACH << {'LOW'}) & (VENTTUBE << {'LOW'})]
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
