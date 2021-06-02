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
AMDewptCalPl ~= choice({'Instability' : 0.3,'Neutral' : 0.25,'Stability' : 0.45})


AMInstabMt ~= choice({'None' : 0.333333,'Weak' : 0.333333,'Strong' : 0.333334})


Date ~= choice({'May15_Jun14' : 0.254098,'Jun15_Jul1' : 0.131148,'Jul2_Jul15' : 0.106557,'Jul16_Aug10' : 0.213115,'Aug11_Aug20' : 0.07377,'Aug20_Sep15' : 0.221312})


IRCloudCover ~= choice({'Cloudy' : 0.15,'PC' : 0.45,'Clear' : 0.4})


LIfr12ZDENSd ~= choice({'LIGt0' : 0.1,'N1GtLIGt_4' : 0.52,'N5GtLIGt_8' : 0.3,'LILt_8' : 0.08})


LLIW ~= choice({'Unfavorable' : 0.12,'Weak' : 0.32,'Moderate' : 0.38,'Strong' : 0.18})


LatestCIN ~= choice({'None' : 0.4,'PartInhibit' : 0.4,'Stifling' : 0.15,'TotalInhibit' : 0.05})


LoLevMoistAd ~= choice({'StrongPos' : 0.12,'WeakPos' : 0.28,'Neutral' : 0.3,'Negative' : 0.3})


MorningBound ~= choice({'None' : 0.5,'Weak' : 0.35,'Strong' : 0.15})


MorningCIN ~= choice({'None' : 0.15,'PartInhibit' : 0.57,'Stifling' : 0.2,'TotalInhibit' : 0.08})


N0_7muVerMo ~= choice({'StrongUp' : 0.25,'WeakUp' : 0.25,'Neutral' : 0.25,'Down' : 0.25})


QGVertMotion ~= choice({'StrongUp' : 0.15,'WeakUp' : 0.15,'Neutral' : 0.5,'Down' : 0.2})


RaoContMoist ~= choice({'VeryWet' : 0.15,'Wet' : 0.2,'Neutral' : 0.4,'Dry' : 0.25})


SatContMoist ~= choice({'VeryWet' : 0.15,'Wet' : 0.2,'Neutral' : 0.4,'Dry' : 0.25})


if (Date == 'May15_Jun14'):
	Scenario ~= choice({'A' : 0.1, 'B' : 0.16, 'C' : 0.1, 'D' : 0.08, 'E' : 0.08, 'F' : 0.01, 'G' : 0.08, 'H' : 0.1, 'I' : 0.09, 'J' : 0.03, 'K' : 0.17000000000000004})
elif(Date == 'Jun15_Jul1'):
	Scenario ~= choice({'A' : 0.05, 'B' : 0.16, 'C' : 0.09, 'D' : 0.09, 'E' : 0.12, 'F' : 0.02, 'G' : 0.13, 'H' : 0.06, 'I' : 0.07, 'J' : 0.11, 'K' : 0.09999999999999998})
elif(Date == 'Jul2_Jul15'):
	Scenario ~= choice({'A' : 0.04, 'B' : 0.13, 'C' : 0.1, 'D' : 0.08, 'E' : 0.15, 'F' : 0.03, 'G' : 0.14, 'H' : 0.04, 'I' : 0.06, 'J' : 0.15, 'K' : 0.07999999999999996})
elif(Date == 'Jul16_Aug10'):
	Scenario ~= choice({'A' : 0.04, 'B' : 0.13, 'C' : 0.09, 'D' : 0.07, 'E' : 0.2, 'F' : 0.08, 'G' : 0.06, 'H' : 0.05, 'I' : 0.07, 'J' : 0.13, 'K' : 0.07999999999999996})
elif(Date == 'Aug11_Aug20'):
	Scenario ~= choice({'A' : 0.04, 'B' : 0.11, 'C' : 0.1, 'D' : 0.07, 'E' : 0.17, 'F' : 0.05, 'G' : 0.1, 'H' : 0.05, 'I' : 0.07, 'J' : 0.14, 'K' : 0.09999999999999998})
else:
	Scenario ~= choice({'A' : 0.05, 'B' : 0.11, 'C' : 0.1, 'D' : 0.08, 'E' : 0.11, 'F' : 0.02, 'G' : 0.11, 'H' : 0.06, 'I' : 0.08, 'J' : 0.11, 'K' : 0.16999999999999993})


if (Scenario == 'A'):
	ScnRelPlFcst ~= choice({'A' : 1.0, 'B' : 0.0, 'C' : 0.0, 'D' : 0.0, 'E' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'I' : 0.0, 'J' : 0.0, 'K' : 0.0})
elif(Scenario == 'B'):
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 1.0, 'C' : 0.0, 'D' : 0.0, 'E' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'I' : 0.0, 'J' : 0.0, 'K' : 0.0})
elif(Scenario == 'C'):
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 0.0, 'C' : 1.0, 'D' : 0.0, 'E' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'I' : 0.0, 'J' : 0.0, 'K' : 0.0})
elif(Scenario == 'D'):
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 0.0, 'C' : 0.0, 'D' : 1.0, 'E' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'I' : 0.0, 'J' : 0.0, 'K' : 0.0})
elif(Scenario == 'E'):
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 0.0, 'C' : 0.0, 'D' : 0.0, 'E' : 1.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'I' : 0.0, 'J' : 0.0, 'K' : 0.0})
elif(Scenario == 'F'):
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 0.0, 'C' : 0.0, 'D' : 0.0, 'E' : 0.0, 'F' : 1.0, 'G' : 0.0, 'H' : 0.0, 'I' : 0.0, 'J' : 0.0, 'K' : 0.0})
elif(Scenario == 'G'):
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 0.0, 'C' : 0.0, 'D' : 0.0, 'E' : 0.0, 'F' : 0.0, 'G' : 1.0, 'H' : 0.0, 'I' : 0.0, 'J' : 0.0, 'K' : 0.0})
elif(Scenario == 'H'):
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 0.0, 'C' : 0.0, 'D' : 0.0, 'E' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 1.0, 'I' : 0.0, 'J' : 0.0, 'K' : 0.0})
elif(Scenario == 'I'):
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 0.0, 'C' : 0.0, 'D' : 0.0, 'E' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'I' : 1.0, 'J' : 0.0, 'K' : 0.0})
elif(Scenario == 'J'):
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 0.0, 'C' : 0.0, 'D' : 0.0, 'E' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'I' : 0.0, 'J' : 1.0, 'K' : 0.0})
else:
	ScnRelPlFcst ~= choice({'A' : 0.0, 'B' : 0.0, 'C' : 0.0, 'D' : 0.0, 'E' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'I' : 0.0, 'J' : 0.0, 'K' : 1.0})


if (Scenario == 'A'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.65, 'E_W_N' : 0.05, 'E_W_S' : 0.1, 'MovingFtorOt' : 0.08, 'DryLine' : 0.04, 'None' : 0.07, 'Other' : 0.010000000000000009})
elif(Scenario == 'B'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.65, 'E_W_N' : 0.05, 'E_W_S' : 0.1, 'MovingFtorOt' : 0.1, 'DryLine' : 0.02, 'None' : 0.07, 'Other' : 0.010000000000000009})
elif(Scenario == 'C'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.0, 'E_W_N' : 0.65, 'E_W_S' : 0.2, 'MovingFtorOt' : 0.02, 'DryLine' : 0.06, 'None' : 0.05, 'Other' : 0.019999999999999796})
elif(Scenario == 'D'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.12, 'E_W_N' : 0.02, 'E_W_S' : 0.02, 'MovingFtorOt' : 0.02, 'DryLine' : 0.45, 'None' : 0.27, 'Other' : 0.09999999999999998})
elif(Scenario == 'E'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.06, 'E_W_N' : 0.14, 'E_W_S' : 0.04, 'MovingFtorOt' : 0.04, 'DryLine' : 0.25, 'None' : 0.4, 'Other' : 0.06999999999999995})
elif(Scenario == 'F'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.1, 'E_W_N' : 0.1, 'E_W_S' : 0.1, 'MovingFtorOt' : 0.02, 'DryLine' : 0.0, 'None' : 0.56, 'Other' : 0.11999999999999988})
elif(Scenario == 'G'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.02, 'E_W_N' : 0.05, 'E_W_S' : 0.05, 'MovingFtorOt' : 0.0, 'DryLine' : 0.35, 'None' : 0.33, 'Other' : 0.19999999999999996})
elif(Scenario == 'H'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.01, 'E_W_N' : 0.1, 'E_W_S' : 0.15, 'MovingFtorOt' : 0.4, 'DryLine' : 0.0, 'None' : 0.23, 'Other' : 0.10999999999999999})
elif(Scenario == 'I'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.02, 'E_W_N' : 0.1, 'E_W_S' : 0.5, 'MovingFtorOt' : 0.3, 'DryLine' : 0.01, 'None' : 0.02, 'Other' : 0.050000000000000044})
elif(Scenario == 'J'):
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.06, 'E_W_N' : 0.08, 'E_W_S' : 0.04, 'MovingFtorOt' : 0.02, 'DryLine' : 0.6, 'None' : 0.14, 'Other' : 0.05999999999999994})
else:
	SfcWndShfDis ~= choice({'DenvCyclone' : 0.05, 'E_W_N' : 0.13, 'E_W_S' : 0.05, 'MovingFtorOt' : 0.39, 'DryLine' : 0.13, 'None' : 0.15, 'Other' : 0.09999999999999998})


SubjVertMo ~= choice({'StronUp' : 0.15,'WeakUp' : 0.15,'Neutral' : 0.5,'Down' : 0.2})


if (Scenario == 'A'):
	SynForcng ~= choice({'SigNegative' : 0.35, 'NegToPos' : 0.25, 'SigPositive' : 0.0, 'PosToNeg' : 0.35, 'LittleChange' : 0.050000000000000044})
elif(Scenario == 'B'):
	SynForcng ~= choice({'SigNegative' : 0.06, 'NegToPos' : 0.1, 'SigPositive' : 0.06, 'PosToNeg' : 0.3, 'LittleChange' : 0.48})
elif(Scenario == 'C'):
	SynForcng ~= choice({'SigNegative' : 0.1, 'NegToPos' : 0.27, 'SigPositive' : 0.4, 'PosToNeg' : 0.08, 'LittleChange' : 0.15000000000000002})
elif(Scenario == 'D'):
	SynForcng ~= choice({'SigNegative' : 0.35, 'NegToPos' : 0.2, 'SigPositive' : 0.1, 'PosToNeg' : 0.25, 'LittleChange' : 0.09999999999999998})
elif(Scenario == 'E'):
	SynForcng ~= choice({'SigNegative' : 0.15, 'NegToPos' : 0.15, 'SigPositive' : 0.1, 'PosToNeg' : 0.15, 'LittleChange' : 0.44999999999999996})
elif(Scenario == 'F'):
	SynForcng ~= choice({'SigNegative' : 0.15, 'NegToPos' : 0.1, 'SigPositive' : 0.05, 'PosToNeg' : 0.15, 'LittleChange' : 0.55})
elif(Scenario == 'G'):
	SynForcng ~= choice({'SigNegative' : 0.15, 'NegToPos' : 0.1, 'SigPositive' : 0.1, 'PosToNeg' : 0.25, 'LittleChange' : 0.4})
elif(Scenario == 'H'):
	SynForcng ~= choice({'SigNegative' : 0.25, 'NegToPos' : 0.25, 'SigPositive' : 0.25, 'PosToNeg' : 0.15, 'LittleChange' : 0.09999999999999998})
elif(Scenario == 'I'):
	SynForcng ~= choice({'SigNegative' : 0.25, 'NegToPos' : 0.2, 'SigPositive' : 0.15, 'PosToNeg' : 0.2, 'LittleChange' : 0.19999999999999996})
elif(Scenario == 'J'):
	SynForcng ~= choice({'SigNegative' : 0.01, 'NegToPos' : 0.05, 'SigPositive' : 0.01, 'PosToNeg' : 0.05, 'LittleChange' : 0.88})
else:
	SynForcng ~= choice({'SigNegative' : 0.2, 'NegToPos' : 0.2, 'SigPositive' : 0.35, 'PosToNeg' : 0.15, 'LittleChange' : 0.09999999999999998})


if (Scenario == 'A'):
	TempDis ~= choice({'QStationary' : 0.13, 'Moving' : 0.15, 'None' : 0.1, 'Other' : 0.62})
elif(Scenario == 'B'):
	TempDis ~= choice({'QStationary' : 0.15, 'Moving' : 0.15, 'None' : 0.25, 'Other' : 0.44999999999999996})
elif(Scenario == 'C'):
	TempDis ~= choice({'QStationary' : 0.12, 'Moving' : 0.1, 'None' : 0.35, 'Other' : 0.43000000000000005})
elif(Scenario == 'D'):
	TempDis ~= choice({'QStationary' : 0.1, 'Moving' : 0.15, 'None' : 0.4, 'Other' : 0.35})
elif(Scenario == 'E'):
	TempDis ~= choice({'QStationary' : 0.04, 'Moving' : 0.04, 'None' : 0.82, 'Other' : 0.10000000000000009})
elif(Scenario == 'F'):
	TempDis ~= choice({'QStationary' : 0.05, 'Moving' : 0.12, 'None' : 0.75, 'Other' : 0.08000000000000007})
elif(Scenario == 'G'):
	TempDis ~= choice({'QStationary' : 0.03, 'Moving' : 0.03, 'None' : 0.84, 'Other' : 0.10000000000000009})
elif(Scenario == 'H'):
	TempDis ~= choice({'QStationary' : 0.05, 'Moving' : 0.4, 'None' : 0.5, 'Other' : 0.050000000000000044})
elif(Scenario == 'I'):
	TempDis ~= choice({'QStationary' : 0.8, 'Moving' : 0.19, 'None' : 0.0, 'Other' : 0.010000000000000009})
elif(Scenario == 'J'):
	TempDis ~= choice({'QStationary' : 0.1, 'Moving' : 0.05, 'None' : 0.4, 'Other' : 0.44999999999999996})
else:
	TempDis ~= choice({'QStationary' : 0.2, 'Moving' : 0.3, 'None' : 0.3, 'Other' : 0.19999999999999996})


VISCloudCov ~= choice({'Cloudy' : 0.1,'PC' : 0.5,'Clear' : 0.4})


if (Scenario == 'A'):
	WindAloft ~= choice({'LV' : 0.0, 'SWQuad' : 0.95, 'NWQuad' : 0.01, 'AllElse' : 0.040000000000000036})
elif(Scenario == 'B'):
	WindAloft ~= choice({'LV' : 0.2, 'SWQuad' : 0.3, 'NWQuad' : 0.2, 'AllElse' : 0.30000000000000004})
elif(Scenario == 'C'):
	WindAloft ~= choice({'LV' : 0.05, 'SWQuad' : 0.09, 'NWQuad' : 0.59, 'AllElse' : 0.27})
elif(Scenario == 'D'):
	WindAloft ~= choice({'LV' : 0.03, 'SWQuad' : 0.32, 'NWQuad' : 0.42, 'AllElse' : 0.22999999999999998})
elif(Scenario == 'E'):
	WindAloft ~= choice({'LV' : 0.07, 'SWQuad' : 0.66, 'NWQuad' : 0.02, 'AllElse' : 0.25})
elif(Scenario == 'F'):
	WindAloft ~= choice({'LV' : 0.5, 'SWQuad' : 0.0, 'NWQuad' : 0.0, 'AllElse' : 0.5})
elif(Scenario == 'G'):
	WindAloft ~= choice({'LV' : 0.25, 'SWQuad' : 0.3, 'NWQuad' : 0.25, 'AllElse' : 0.19999999999999996})
elif(Scenario == 'H'):
	WindAloft ~= choice({'LV' : 0.2, 'SWQuad' : 0.14, 'NWQuad' : 0.43, 'AllElse' : 0.22999999999999998})
elif(Scenario == 'I'):
	WindAloft ~= choice({'LV' : 0.2, 'SWQuad' : 0.41, 'NWQuad' : 0.1, 'AllElse' : 0.29000000000000004})
elif(Scenario == 'J'):
	WindAloft ~= choice({'LV' : 0.96, 'SWQuad' : 0.0, 'NWQuad' : 0.0, 'AllElse' : 0.040000000000000036})
else:
	WindAloft ~= choice({'LV' : 0.03, 'SWQuad' : 0.08, 'NWQuad' : 0.33, 'AllElse' : 0.56})


if (Scenario == 'A'):
	WindFieldMt ~= choice({'Westerly' : 0.8, 'LVorOther' : 0.19999999999999996})
elif(Scenario == 'B'):
	WindFieldMt ~= choice({'Westerly' : 0.35, 'LVorOther' : 0.65})
elif(Scenario == 'C'):
	WindFieldMt ~= choice({'Westerly' : 0.75, 'LVorOther' : 0.25})
elif(Scenario == 'D'):
	WindFieldMt ~= choice({'Westerly' : 0.7, 'LVorOther' : 0.30000000000000004})
elif(Scenario == 'E'):
	WindFieldMt ~= choice({'Westerly' : 0.65, 'LVorOther' : 0.35})
elif(Scenario == 'F'):
	WindFieldMt ~= choice({'Westerly' : 0.15, 'LVorOther' : 0.85})
elif(Scenario == 'G'):
	WindFieldMt ~= choice({'Westerly' : 0.7, 'LVorOther' : 0.30000000000000004})
elif(Scenario == 'H'):
	WindFieldMt ~= choice({'Westerly' : 0.3, 'LVorOther' : 0.7})
elif(Scenario == 'I'):
	WindFieldMt ~= choice({'Westerly' : 0.5, 'LVorOther' : 0.5})
elif(Scenario == 'J'):
	WindFieldMt ~= choice({'Westerly' : 0.01, 'LVorOther' : 0.99})
else:
	WindFieldMt ~= choice({'Westerly' : 0.7, 'LVorOther' : 0.30000000000000004})


if (Scenario == 'A'):
	WindFieldPln ~= choice({'LV' : 0.05, 'DenvCyclone' : 0.6, 'LongAnticyc' : 0.02, 'E_NE' : 0.1, 'SEQuad' : 0.23, 'WidespdDnsl' : 0.0})
elif(Scenario == 'B'):
	WindFieldPln ~= choice({'LV' : 0.08, 'DenvCyclone' : 0.6, 'LongAnticyc' : 0.02, 'E_NE' : 0.1, 'SEQuad' : 0.2, 'WidespdDnsl' : 0.0})
elif(Scenario == 'C'):
	WindFieldPln ~= choice({'LV' : 0.1, 'DenvCyclone' : 0.0, 'LongAnticyc' : 0.75, 'E_NE' : 0.0, 'SEQuad' : 0.0, 'WidespdDnsl' : 0.15000000000000002})
elif(Scenario == 'D'):
	WindFieldPln ~= choice({'LV' : 0.1, 'DenvCyclone' : 0.15, 'LongAnticyc' : 0.2, 'E_NE' : 0.05, 'SEQuad' : 0.3, 'WidespdDnsl' : 0.19999999999999996})
elif(Scenario == 'E'):
	WindFieldPln ~= choice({'LV' : 0.43, 'DenvCyclone' : 0.1, 'LongAnticyc' : 0.15, 'E_NE' : 0.06, 'SEQuad' : 0.06, 'WidespdDnsl' : 0.19999999999999996})
elif(Scenario == 'F'):
	WindFieldPln ~= choice({'LV' : 0.6, 'DenvCyclone' : 0.07, 'LongAnticyc' : 0.01, 'E_NE' : 0.12, 'SEQuad' : 0.2, 'WidespdDnsl' : 0.0})
elif(Scenario == 'G'):
	WindFieldPln ~= choice({'LV' : 0.25, 'DenvCyclone' : 0.01, 'LongAnticyc' : 0.3, 'E_NE' : 0.01, 'SEQuad' : 0.03, 'WidespdDnsl' : 0.3999999999999999})
elif(Scenario == 'H'):
	WindFieldPln ~= choice({'LV' : 0.04, 'DenvCyclone' : 0.02, 'LongAnticyc' : 0.04, 'E_NE' : 0.8, 'SEQuad' : 0.1, 'WidespdDnsl' : 0.0})
elif(Scenario == 'I'):
	WindFieldPln ~= choice({'LV' : 0.2, 'DenvCyclone' : 0.3, 'LongAnticyc' : 0.05, 'E_NE' : 0.37, 'SEQuad' : 0.07, 'WidespdDnsl' : 0.010000000000000009})
elif(Scenario == 'J'):
	WindFieldPln ~= choice({'LV' : 0.6, 'DenvCyclone' : 0.08, 'LongAnticyc' : 0.07, 'E_NE' : 0.03, 'SEQuad' : 0.2, 'WidespdDnsl' : 0.020000000000000018})
else:
	WindFieldPln ~= choice({'LV' : 0.1, 'DenvCyclone' : 0.05, 'LongAnticyc' : 0.1, 'E_NE' : 0.05, 'SEQuad' : 0.2, 'WidespdDnsl' : 0.5})


WndHodograph ~= choice({'DCVZFavor' : 0.3,'StrongWest' : 0.25,'Westerly' : 0.25,'Other' : 0.2})


if (VISCloudCov == 'Cloudy'):
	if (IRCloudCover == 'Cloudy'):
		CombClouds ~= choice({'Cloudy' : 0.95, 'PC' : 0.04, 'Clear' : 0.010000000000000009})
	elif(IRCloudCover == 'PC'):
		CombClouds ~= choice({'Cloudy' : 0.85, 'PC' : 0.13, 'Clear' : 0.020000000000000018})
	else:
		CombClouds ~= choice({'Cloudy' : 0.8, 'PC' : 0.1, 'Clear' : 0.09999999999999998})
elif(VISCloudCov == 'PC'):
	if(IRCloudCover == 'Cloudy'):
		CombClouds ~= choice({'Cloudy' : 0.45, 'PC' : 0.52, 'Clear' : 0.030000000000000027})
	elif(IRCloudCover == 'PC'):
		CombClouds ~= choice({'Cloudy' : 0.1, 'PC' : 0.8, 'Clear' : 0.09999999999999998})
	else:
		CombClouds ~= choice({'Cloudy' : 0.05, 'PC' : 0.45, 'Clear' : 0.5})
else:
	if(IRCloudCover == 'Cloudy'):
		CombClouds ~= choice({'Cloudy' : 0.1, 'PC' : 0.4, 'Clear' : 0.5})
	elif(IRCloudCover == 'PC'):
		CombClouds ~= choice({'Cloudy' : 0.02, 'PC' : 0.28, 'Clear' : 0.7})
	else:
		CombClouds ~= choice({'Cloudy' : 0.0, 'PC' : 0.02, 'Clear' : 0.98})


if (SatContMoist == 'VeryWet'):
	if (RaoContMoist == 'VeryWet'):
		CombMoisture ~= choice({'VeryWet' : 0.9, 'Wet' : 0.1, 'Neutral' : 0.0, 'Dry' : 0.0})
	elif(RaoContMoist == 'Wet'):
		CombMoisture ~= choice({'VeryWet' : 0.6, 'Wet' : 0.35, 'Neutral' : 0.05, 'Dry' : 0.0})
	elif(RaoContMoist == 'Neutral'):
		CombMoisture ~= choice({'VeryWet' : 0.3, 'Wet' : 0.5, 'Neutral' : 0.2, 'Dry' : 0.0})
	else:
		CombMoisture ~= choice({'VeryWet' : 0.25, 'Wet' : 0.35, 'Neutral' : 0.25, 'Dry' : 0.15000000000000002})
elif(SatContMoist == 'Wet'):
	if(RaoContMoist == 'VeryWet'):
		CombMoisture ~= choice({'VeryWet' : 0.55, 'Wet' : 0.4, 'Neutral' : 0.05, 'Dry' : 0.0})
	elif(RaoContMoist == 'Wet'):
		CombMoisture ~= choice({'VeryWet' : 0.15, 'Wet' : 0.6, 'Neutral' : 0.2, 'Dry' : 0.050000000000000044})
	elif(RaoContMoist == 'Neutral'):
		CombMoisture ~= choice({'VeryWet' : 0.05, 'Wet' : 0.4, 'Neutral' : 0.45, 'Dry' : 0.09999999999999998})
	else:
		CombMoisture ~= choice({'VeryWet' : 0.1, 'Wet' : 0.3, 'Neutral' : 0.3, 'Dry' : 0.30000000000000004})
elif(SatContMoist == 'Neutral'):
	if(RaoContMoist == 'VeryWet'):
		CombMoisture ~= choice({'VeryWet' : 0.25, 'Wet' : 0.3, 'Neutral' : 0.35, 'Dry' : 0.09999999999999998})
	elif(RaoContMoist == 'Wet'):
		CombMoisture ~= choice({'VeryWet' : 0.1, 'Wet' : 0.35, 'Neutral' : 0.5, 'Dry' : 0.050000000000000044})
	elif(RaoContMoist == 'Neutral'):
		CombMoisture ~= choice({'VeryWet' : 0.0, 'Wet' : 0.15, 'Neutral' : 0.7, 'Dry' : 0.15000000000000002})
	else:
		CombMoisture ~= choice({'VeryWet' : 0.0, 'Wet' : 0.1, 'Neutral' : 0.4, 'Dry' : 0.5})
else:
	if(RaoContMoist == 'VeryWet'):
		CombMoisture ~= choice({'VeryWet' : 0.25, 'Wet' : 0.25, 'Neutral' : 0.25, 'Dry' : 0.25})
	elif(RaoContMoist == 'Wet'):
		CombMoisture ~= choice({'VeryWet' : 0.25, 'Wet' : 0.25, 'Neutral' : 0.25, 'Dry' : 0.25})
	elif(RaoContMoist == 'Neutral'):
		CombMoisture ~= choice({'VeryWet' : 0.25, 'Wet' : 0.25, 'Neutral' : 0.25, 'Dry' : 0.25})
	else:
		CombMoisture ~= choice({'VeryWet' : 0.25, 'Wet' : 0.25, 'Neutral' : 0.25, 'Dry' : 0.25})


if (N0_7muVerMo == 'StrongUp'):
	if (SubjVertMo == 'StronUp'):
		if (QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 1.0, 'WeakUp' : 0.0, 'Neutral' : 0.0, 'Down' : 0.0})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.9, 'WeakUp' : 0.1, 'Neutral' : 0.0, 'Down' : 0.0})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.7, 'WeakUp' : 0.2, 'Neutral' : 0.1, 'Down' : 1.1102230246251565e-16})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.2, 'WeakUp' : 0.5, 'Neutral' : 0.2, 'Down' : 0.10000000000000009})
	elif(SubjVertMo == 'WeakUp'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.9, 'WeakUp' : 0.1, 'Neutral' : 0.0, 'Down' : 0.0})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.7, 'WeakUp' : 0.3, 'Neutral' : 0.0, 'Down' : 0.0})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.15, 'WeakUp' : 0.7, 'Neutral' : 0.15, 'Down' : 0.0})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.35, 'Neutral' : 0.45, 'Down' : 0.10000000000000009})
	elif(SubjVertMo == 'Neutral'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.7, 'WeakUp' : 0.2, 'Neutral' : 0.1, 'Down' : 1.1102230246251565e-16})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.15, 'WeakUp' : 0.7, 'Neutral' : 0.15, 'Down' : 0.0})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.2, 'WeakUp' : 0.6, 'Neutral' : 0.2, 'Down' : 0.0})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.2, 'Neutral' : 0.6, 'Down' : 0.09999999999999998})
	else:
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.2, 'WeakUp' : 0.5, 'Neutral' : 0.2, 'Down' : 0.10000000000000009})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.35, 'Neutral' : 0.45, 'Down' : 0.10000000000000009})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.2, 'Neutral' : 0.6, 'Down' : 0.09999999999999998})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.1, 'Neutral' : 0.2, 'Down' : 0.6})
elif(N0_7muVerMo == 'WeakUp'):
	if(SubjVertMo == 'StronUp'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.9, 'WeakUp' : 0.1, 'Neutral' : 0.0, 'Down' : 0.0})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.7, 'WeakUp' : 0.3, 'Neutral' : 0.0, 'Down' : 0.0})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.15, 'WeakUp' : 0.7, 'Neutral' : 0.15, 'Down' : 0.0})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.35, 'Neutral' : 0.45, 'Down' : 0.10000000000000009})
	elif(SubjVertMo == 'WeakUp'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.7, 'WeakUp' : 0.3, 'Neutral' : 0.0, 'Down' : 0.0})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 1.0, 'Neutral' : 0.0, 'Down' : 0.0})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.7, 'Neutral' : 0.3, 'Down' : 0.0})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.2, 'Neutral' : 0.7, 'Down' : 0.10000000000000009})
	elif(SubjVertMo == 'Neutral'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.15, 'WeakUp' : 0.7, 'Neutral' : 0.15, 'Down' : 0.0})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.7, 'Neutral' : 0.3, 'Down' : 0.0})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.3, 'Neutral' : 0.7, 'Down' : 0.0})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.15, 'Neutral' : 0.5, 'Down' : 0.35})
	else:
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.35, 'Neutral' : 0.45, 'Down' : 0.10000000000000009})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.2, 'Neutral' : 0.7, 'Down' : 0.10000000000000009})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.15, 'Neutral' : 0.5, 'Down' : 0.35})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.1, 'Neutral' : 0.2, 'Down' : 0.7})
elif(N0_7muVerMo == 'Neutral'):
	if(SubjVertMo == 'StronUp'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.7, 'WeakUp' : 0.2, 'Neutral' : 0.1, 'Down' : 1.1102230246251565e-16})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.15, 'WeakUp' : 0.7, 'Neutral' : 0.15, 'Down' : 0.0})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.2, 'WeakUp' : 0.6, 'Neutral' : 0.2, 'Down' : 0.0})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.2, 'Neutral' : 0.6, 'Down' : 0.09999999999999998})
	elif(SubjVertMo == 'WeakUp'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.15, 'WeakUp' : 0.7, 'Neutral' : 0.15, 'Down' : 0.0})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.7, 'Neutral' : 0.3, 'Down' : 0.0})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.3, 'Neutral' : 0.7, 'Down' : 0.0})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.15, 'Neutral' : 0.5, 'Down' : 0.35})
	elif(SubjVertMo == 'Neutral'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.2, 'WeakUp' : 0.6, 'Neutral' : 0.2, 'Down' : 0.0})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.3, 'Neutral' : 0.7, 'Down' : 0.0})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 1.0, 'Down' : 0.0})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 0.7, 'Down' : 0.30000000000000004})
	else:
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.2, 'Neutral' : 0.6, 'Down' : 0.09999999999999998})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.15, 'Neutral' : 0.5, 'Down' : 0.35})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 0.7, 'Down' : 0.30000000000000004})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 0.3, 'Down' : 0.7})
else:
	if(SubjVertMo == 'StronUp'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.2, 'WeakUp' : 0.5, 'Neutral' : 0.2, 'Down' : 0.10000000000000009})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.35, 'Neutral' : 0.45, 'Down' : 0.10000000000000009})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.2, 'Neutral' : 0.6, 'Down' : 0.09999999999999998})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.1, 'Neutral' : 0.2, 'Down' : 0.6})
	elif(SubjVertMo == 'WeakUp'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.35, 'Neutral' : 0.45, 'Down' : 0.10000000000000009})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.2, 'Neutral' : 0.7, 'Down' : 0.10000000000000009})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.15, 'Neutral' : 0.5, 'Down' : 0.35})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.1, 'Neutral' : 0.2, 'Down' : 0.7})
	elif(SubjVertMo == 'Neutral'):
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.2, 'Neutral' : 0.6, 'Down' : 0.09999999999999998})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.15, 'Neutral' : 0.5, 'Down' : 0.35})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 0.7, 'Down' : 0.30000000000000004})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 0.3, 'Down' : 0.7})
	else:
		if(QGVertMotion == 'StrongUp'):
			CombVerMo ~= choice({'StrongUp' : 0.1, 'WeakUp' : 0.1, 'Neutral' : 0.2, 'Down' : 0.6})
		elif(QGVertMotion == 'WeakUp'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.1, 'Neutral' : 0.2, 'Down' : 0.7})
		elif(QGVertMotion == 'Neutral'):
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 0.3, 'Down' : 0.7})
		else:
			CombVerMo ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 0.0, 'Down' : 1.0})


if (LatestCIN == 'None'):
	if (LLIW == 'Unfavorable'):
		CurPropConv ~= choice({'None' : 0.7, 'Slight' : 0.28, 'Moderate' : 0.02, 'Strong' : 0.0})
	elif(LLIW == 'Weak'):
		CurPropConv ~= choice({'None' : 0.1, 'Slight' : 0.5, 'Moderate' : 0.3, 'Strong' : 0.10000000000000009})
	elif(LLIW == 'Moderate'):
		CurPropConv ~= choice({'None' : 0.01, 'Slight' : 0.14, 'Moderate' : 0.35, 'Strong' : 0.5})
	else:
		CurPropConv ~= choice({'None' : 0.0, 'Slight' : 0.02, 'Moderate' : 0.18, 'Strong' : 0.8})
elif(LatestCIN == 'PartInhibit'):
	if(LLIW == 'Unfavorable'):
		CurPropConv ~= choice({'None' : 0.9, 'Slight' : 0.09, 'Moderate' : 0.01, 'Strong' : 0.0})
	elif(LLIW == 'Weak'):
		CurPropConv ~= choice({'None' : 0.65, 'Slight' : 0.25, 'Moderate' : 0.09, 'Strong' : 0.010000000000000009})
	elif(LLIW == 'Moderate'):
		CurPropConv ~= choice({'None' : 0.25, 'Slight' : 0.35, 'Moderate' : 0.3, 'Strong' : 0.10000000000000009})
	else:
		CurPropConv ~= choice({'None' : 0.01, 'Slight' : 0.15, 'Moderate' : 0.33, 'Strong' : 0.51})
elif(LatestCIN == 'Stifling'):
	if(LLIW == 'Unfavorable'):
		CurPropConv ~= choice({'None' : 0.95, 'Slight' : 0.05, 'Moderate' : 0.0, 'Strong' : 0.0})
	elif(LLIW == 'Weak'):
		CurPropConv ~= choice({'None' : 0.75, 'Slight' : 0.23, 'Moderate' : 0.02, 'Strong' : 0.0})
	elif(LLIW == 'Moderate'):
		CurPropConv ~= choice({'None' : 0.4, 'Slight' : 0.4, 'Moderate' : 0.18, 'Strong' : 0.020000000000000018})
	else:
		CurPropConv ~= choice({'None' : 0.2, 'Slight' : 0.3, 'Moderate' : 0.35, 'Strong' : 0.15000000000000002})
else:
	if(LLIW == 'Unfavorable'):
		CurPropConv ~= choice({'None' : 1.0, 'Slight' : 0.0, 'Moderate' : 0.0, 'Strong' : 0.0})
	elif(LLIW == 'Weak'):
		CurPropConv ~= choice({'None' : 0.95, 'Slight' : 0.05, 'Moderate' : 0.0, 'Strong' : 0.0})
	elif(LLIW == 'Moderate'):
		CurPropConv ~= choice({'None' : 0.75, 'Slight' : 0.2, 'Moderate' : 0.05, 'Strong' : 0.0})
	else:
		CurPropConv ~= choice({'None' : 0.5, 'Slight' : 0.35, 'Moderate' : 0.1, 'Strong' : 0.050000000000000044})


if (Scenario == 'A'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.04, 'LowAtStation' : 0.05, 'LowSHighN' : 0.15, 'LowNHighS' : 0.05, 'LowMtsHighPl' : 0.19, 'HighEvrywher' : 0.3, 'Other' : 0.21999999999999997})
elif(Scenario == 'B'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.05, 'LowAtStation' : 0.07, 'LowSHighN' : 0.15, 'LowNHighS' : 0.1, 'LowMtsHighPl' : 0.3, 'HighEvrywher' : 0.27, 'Other' : 0.06000000000000005})
elif(Scenario == 'C'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.4, 'LowAtStation' : 0.25, 'LowSHighN' : 0.0, 'LowNHighS' : 0.15, 'LowMtsHighPl' : 0.05, 'HighEvrywher' : 0.02, 'Other' : 0.1299999999999999})
elif(Scenario == 'D'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.13, 'LowAtStation' : 0.22, 'LowSHighN' : 0.18, 'LowNHighS' : 0.07, 'LowMtsHighPl' : 0.34, 'HighEvrywher' : 0.03, 'Other' : 0.029999999999999805})
elif(Scenario == 'E'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.15, 'LowAtStation' : 0.2, 'LowSHighN' : 0.2, 'LowNHighS' : 0.18, 'LowMtsHighPl' : 0.11, 'HighEvrywher' : 0.11, 'Other' : 0.050000000000000044})
elif(Scenario == 'F'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.0, 'LowAtStation' : 0.0, 'LowSHighN' : 0.0, 'LowNHighS' : 0.0, 'LowMtsHighPl' : 0.0, 'HighEvrywher' : 0.98, 'Other' : 0.020000000000000018})
elif(Scenario == 'G'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.5, 'LowAtStation' : 0.27, 'LowSHighN' : 0.15, 'LowNHighS' : 0.02, 'LowMtsHighPl' : 0.02, 'HighEvrywher' : 0.0, 'Other' : 0.039999999999999925})
elif(Scenario == 'H'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.0, 'LowAtStation' : 0.02, 'LowSHighN' : 0.1, 'LowNHighS' : 0.05, 'LowMtsHighPl' : 0.5, 'HighEvrywher' : 0.2, 'Other' : 0.1299999999999999})
elif(Scenario == 'I'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.0, 'LowAtStation' : 0.02, 'LowSHighN' : 0.7, 'LowNHighS' : 0.0, 'LowMtsHighPl' : 0.2, 'HighEvrywher' : 0.04, 'Other' : 0.040000000000000036})
elif(Scenario == 'J'):
	Dewpoints ~= choice({'LowEvrywhere' : 0.1, 'LowAtStation' : 0.45, 'LowSHighN' : 0.1, 'LowNHighS' : 0.05, 'LowMtsHighPl' : 0.26, 'HighEvrywher' : 0.02, 'Other' : 0.019999999999999907})
else:
	Dewpoints ~= choice({'LowEvrywhere' : 0.1, 'LowAtStation' : 0.1, 'LowSHighN' : 0.1, 'LowNHighS' : 0.2, 'LowMtsHighPl' : 0.05, 'HighEvrywher' : 0.1, 'Other' : 0.35})


if (Scenario == 'A'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.04, 'Steep' : 0.25, 'ModerateOrLe' : 0.35, 'Stable' : 0.3600000000000001})
elif(Scenario == 'B'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.07, 'Steep' : 0.31, 'ModerateOrLe' : 0.31, 'Stable' : 0.31000000000000005})
elif(Scenario == 'C'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.35, 'Steep' : 0.47, 'ModerateOrLe' : 0.14, 'Stable' : 0.040000000000000036})
elif(Scenario == 'D'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.4, 'Steep' : 0.4, 'ModerateOrLe' : 0.13, 'Stable' : 0.06999999999999995})
elif(Scenario == 'E'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.45, 'Steep' : 0.35, 'ModerateOrLe' : 0.15, 'Stable' : 0.04999999999999993})
elif(Scenario == 'F'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.01, 'Steep' : 0.35, 'ModerateOrLe' : 0.45, 'Stable' : 0.18999999999999995})
elif(Scenario == 'G'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.78, 'Steep' : 0.19, 'ModerateOrLe' : 0.03, 'Stable' : 0.0})
elif(Scenario == 'H'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.0, 'Steep' : 0.02, 'ModerateOrLe' : 0.33, 'Stable' : 0.6499999999999999})
elif(Scenario == 'I'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.22, 'Steep' : 0.4, 'ModerateOrLe' : 0.3, 'Stable' : 0.08000000000000007})
elif(Scenario == 'J'):
	LowLLapse ~= choice({'CloseToDryAd' : 0.13, 'Steep' : 0.4, 'ModerateOrLe' : 0.35, 'Stable' : 0.12})
else:
	LowLLapse ~= choice({'CloseToDryAd' : 0.09, 'Steep' : 0.4, 'ModerateOrLe' : 0.33, 'Stable' : 0.17999999999999994})


if (Scenario == 'A'):
	MeanRH ~= choice({'VeryMoist' : 0.33, 'Average' : 0.5, 'Dry' : 0.16999999999999993})
elif(Scenario == 'B'):
	MeanRH ~= choice({'VeryMoist' : 0.4, 'Average' : 0.4, 'Dry' : 0.19999999999999996})
elif(Scenario == 'C'):
	MeanRH ~= choice({'VeryMoist' : 0.05, 'Average' : 0.45, 'Dry' : 0.5})
elif(Scenario == 'D'):
	MeanRH ~= choice({'VeryMoist' : 0.1, 'Average' : 0.5, 'Dry' : 0.4})
elif(Scenario == 'E'):
	MeanRH ~= choice({'VeryMoist' : 0.05, 'Average' : 0.65, 'Dry' : 0.29999999999999993})
elif(Scenario == 'F'):
	MeanRH ~= choice({'VeryMoist' : 1.0, 'Average' : 0.0, 'Dry' : 0.0})
elif(Scenario == 'G'):
	MeanRH ~= choice({'VeryMoist' : 0.0, 'Average' : 0.07, 'Dry' : 0.9299999999999999})
elif(Scenario == 'H'):
	MeanRH ~= choice({'VeryMoist' : 0.4, 'Average' : 0.55, 'Dry' : 0.04999999999999993})
elif(Scenario == 'I'):
	MeanRH ~= choice({'VeryMoist' : 0.2, 'Average' : 0.45, 'Dry' : 0.35})
elif(Scenario == 'J'):
	MeanRH ~= choice({'VeryMoist' : 0.05, 'Average' : 0.55, 'Dry' : 0.3999999999999999})
else:
	MeanRH ~= choice({'VeryMoist' : 0.2, 'Average' : 0.4, 'Dry' : 0.3999999999999999})


if (Scenario == 'A'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.25, 'Steep' : 0.55, 'ModerateOrLe' : 0.19999999999999996})
elif(Scenario == 'B'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.25, 'Steep' : 0.5, 'ModerateOrLe' : 0.25})
elif(Scenario == 'C'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.4, 'Steep' : 0.38, 'ModerateOrLe' : 0.21999999999999997})
elif(Scenario == 'D'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.43, 'Steep' : 0.37, 'ModerateOrLe' : 0.19999999999999996})
elif(Scenario == 'E'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.02, 'Steep' : 0.38, 'ModerateOrLe' : 0.6})
elif(Scenario == 'F'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.0, 'Steep' : 0.1, 'ModerateOrLe' : 0.9})
elif(Scenario == 'G'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.84, 'Steep' : 0.16, 'ModerateOrLe' : 0.0})
elif(Scenario == 'H'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.25, 'Steep' : 0.31, 'ModerateOrLe' : 0.43999999999999995})
elif(Scenario == 'I'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.41, 'Steep' : 0.29, 'ModerateOrLe' : 0.30000000000000004})
elif(Scenario == 'J'):
	MidLLapse ~= choice({'CloseToDryAd' : 0.23, 'Steep' : 0.42, 'ModerateOrLe' : 0.35})
else:
	MidLLapse ~= choice({'CloseToDryAd' : 0.16, 'Steep' : 0.28, 'ModerateOrLe' : 0.5599999999999999})


if (Scenario == 'A'):
	MvmtFeatures ~= choice({'StrongFront' : 0.25, 'MarkedUpper' : 0.55, 'OtherRapid' : 0.2, 'NoMajor' : 0.0})
elif(Scenario == 'B'):
	MvmtFeatures ~= choice({'StrongFront' : 0.05, 'MarkedUpper' : 0.1, 'OtherRapid' : 0.1, 'NoMajor' : 0.75})
elif(Scenario == 'C'):
	MvmtFeatures ~= choice({'StrongFront' : 0.1, 'MarkedUpper' : 0.3, 'OtherRapid' : 0.3, 'NoMajor' : 0.30000000000000004})
elif(Scenario == 'D'):
	MvmtFeatures ~= choice({'StrongFront' : 0.18, 'MarkedUpper' : 0.38, 'OtherRapid' : 0.34, 'NoMajor' : 0.09999999999999987})
elif(Scenario == 'E'):
	MvmtFeatures ~= choice({'StrongFront' : 0.02, 'MarkedUpper' : 0.02, 'OtherRapid' : 0.26, 'NoMajor' : 0.7})
elif(Scenario == 'F'):
	MvmtFeatures ~= choice({'StrongFront' : 0.05, 'MarkedUpper' : 0.07, 'OtherRapid' : 0.05, 'NoMajor' : 0.83})
elif(Scenario == 'G'):
	MvmtFeatures ~= choice({'StrongFront' : 0.1, 'MarkedUpper' : 0.25, 'OtherRapid' : 0.15, 'NoMajor' : 0.5})
elif(Scenario == 'H'):
	MvmtFeatures ~= choice({'StrongFront' : 0.0, 'MarkedUpper' : 0.6, 'OtherRapid' : 0.1, 'NoMajor' : 0.30000000000000004})
elif(Scenario == 'I'):
	MvmtFeatures ~= choice({'StrongFront' : 0.2, 'MarkedUpper' : 0.1, 'OtherRapid' : 0.2, 'NoMajor' : 0.5})
elif(Scenario == 'J'):
	MvmtFeatures ~= choice({'StrongFront' : 0.04, 'MarkedUpper' : 0.0, 'OtherRapid' : 0.04, 'NoMajor' : 0.92})
else:
	MvmtFeatures ~= choice({'StrongFront' : 0.5, 'MarkedUpper' : 0.35, 'OtherRapid' : 0.09, 'NoMajor' : 0.06000000000000005})


if (Scenario == 'A'):
	RHRatio ~= choice({'MoistMDryL' : 0.05, 'DryMMoistL' : 0.5, 'Other' : 0.44999999999999996})
elif(Scenario == 'B'):
	RHRatio ~= choice({'MoistMDryL' : 0.1, 'DryMMoistL' : 0.5, 'Other' : 0.4})
elif(Scenario == 'C'):
	RHRatio ~= choice({'MoistMDryL' : 0.4, 'DryMMoistL' : 0.15, 'Other' : 0.44999999999999996})
elif(Scenario == 'D'):
	RHRatio ~= choice({'MoistMDryL' : 0.2, 'DryMMoistL' : 0.45, 'Other' : 0.35})
elif(Scenario == 'E'):
	RHRatio ~= choice({'MoistMDryL' : 0.8, 'DryMMoistL' : 0.05, 'Other' : 0.1499999999999999})
elif(Scenario == 'F'):
	RHRatio ~= choice({'MoistMDryL' : 0.0, 'DryMMoistL' : 0.0, 'Other' : 1.0})
elif(Scenario == 'G'):
	RHRatio ~= choice({'MoistMDryL' : 0.6, 'DryMMoistL' : 0.0, 'Other' : 0.4})
elif(Scenario == 'H'):
	RHRatio ~= choice({'MoistMDryL' : 0.0, 'DryMMoistL' : 0.7, 'Other' : 0.30000000000000004})
elif(Scenario == 'I'):
	RHRatio ~= choice({'MoistMDryL' : 0.1, 'DryMMoistL' : 0.7, 'Other' : 0.20000000000000007})
elif(Scenario == 'J'):
	RHRatio ~= choice({'MoistMDryL' : 0.4, 'DryMMoistL' : 0.4, 'Other' : 0.19999999999999996})
else:
	RHRatio ~= choice({'MoistMDryL' : 0.15, 'DryMMoistL' : 0.45, 'Other' : 0.4})


if (Scenario == 'A'):
	ScenRel3_4 ~= choice({'ACEFK' : 1.0, 'B' : 0.0, 'D' : 0.0, 'GJ' : 0.0, 'HI' : 0.0})
elif(Scenario == 'B'):
	ScenRel3_4 ~= choice({'ACEFK' : 0.0, 'B' : 1.0, 'D' : 0.0, 'GJ' : 0.0, 'HI' : 0.0})
elif(Scenario == 'C'):
	ScenRel3_4 ~= choice({'ACEFK' : 1.0, 'B' : 0.0, 'D' : 0.0, 'GJ' : 0.0, 'HI' : 0.0})
elif(Scenario == 'D'):
	ScenRel3_4 ~= choice({'ACEFK' : 0.0, 'B' : 0.0, 'D' : 1.0, 'GJ' : 0.0, 'HI' : 0.0})
elif(Scenario == 'E'):
	ScenRel3_4 ~= choice({'ACEFK' : 1.0, 'B' : 0.0, 'D' : 0.0, 'GJ' : 0.0, 'HI' : 0.0})
elif(Scenario == 'F'):
	ScenRel3_4 ~= choice({'ACEFK' : 1.0, 'B' : 0.0, 'D' : 0.0, 'GJ' : 0.0, 'HI' : 0.0})
elif(Scenario == 'G'):
	ScenRel3_4 ~= choice({'ACEFK' : 0.0, 'B' : 0.0, 'D' : 0.0, 'GJ' : 1.0, 'HI' : 0.0})
elif(Scenario == 'H'):
	ScenRel3_4 ~= choice({'ACEFK' : 0.0, 'B' : 0.0, 'D' : 0.0, 'GJ' : 0.0, 'HI' : 1.0})
elif(Scenario == 'I'):
	ScenRel3_4 ~= choice({'ACEFK' : 0.0, 'B' : 0.0, 'D' : 0.0, 'GJ' : 0.0, 'HI' : 1.0})
elif(Scenario == 'J'):
	ScenRel3_4 ~= choice({'ACEFK' : 0.0, 'B' : 0.0, 'D' : 0.0, 'GJ' : 1.0, 'HI' : 0.0})
else:
	ScenRel3_4 ~= choice({'ACEFK' : 1.0, 'B' : 0.0, 'D' : 0.0, 'GJ' : 0.0, 'HI' : 0.0})


if (Scenario == 'A'):
	ScenRelAMCIN ~= choice({'AB' : 1.0, 'CThruK' : 0.0})
elif(Scenario == 'B'):
	ScenRelAMCIN ~= choice({'AB' : 1.0, 'CThruK' : 0.0})
elif(Scenario == 'C'):
	ScenRelAMCIN ~= choice({'AB' : 0.0, 'CThruK' : 1.0})
elif(Scenario == 'D'):
	ScenRelAMCIN ~= choice({'AB' : 0.0, 'CThruK' : 1.0})
elif(Scenario == 'E'):
	ScenRelAMCIN ~= choice({'AB' : 0.0, 'CThruK' : 1.0})
elif(Scenario == 'F'):
	ScenRelAMCIN ~= choice({'AB' : 0.0, 'CThruK' : 1.0})
elif(Scenario == 'G'):
	ScenRelAMCIN ~= choice({'AB' : 0.0, 'CThruK' : 1.0})
elif(Scenario == 'H'):
	ScenRelAMCIN ~= choice({'AB' : 0.0, 'CThruK' : 1.0})
elif(Scenario == 'I'):
	ScenRelAMCIN ~= choice({'AB' : 0.0, 'CThruK' : 1.0})
elif(Scenario == 'J'):
	ScenRelAMCIN ~= choice({'AB' : 0.0, 'CThruK' : 1.0})
else:
	ScenRelAMCIN ~= choice({'AB' : 0.0, 'CThruK' : 1.0})


if (Scenario == 'A'):
	ScenRelAMIns ~= choice({'ABI' : 1.0, 'CDEJ' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'K' : 0.0})
elif(Scenario == 'B'):
	ScenRelAMIns ~= choice({'ABI' : 1.0, 'CDEJ' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'K' : 0.0})
elif(Scenario == 'C'):
	ScenRelAMIns ~= choice({'ABI' : 0.0, 'CDEJ' : 1.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'K' : 0.0})
elif(Scenario == 'D'):
	ScenRelAMIns ~= choice({'ABI' : 0.0, 'CDEJ' : 1.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'K' : 0.0})
elif(Scenario == 'E'):
	ScenRelAMIns ~= choice({'ABI' : 0.0, 'CDEJ' : 1.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'K' : 0.0})
elif(Scenario == 'F'):
	ScenRelAMIns ~= choice({'ABI' : 0.0, 'CDEJ' : 0.0, 'F' : 1.0, 'G' : 0.0, 'H' : 0.0, 'K' : 0.0})
elif(Scenario == 'G'):
	ScenRelAMIns ~= choice({'ABI' : 0.0, 'CDEJ' : 0.0, 'F' : 0.0, 'G' : 1.0, 'H' : 0.0, 'K' : 0.0})
elif(Scenario == 'H'):
	ScenRelAMIns ~= choice({'ABI' : 0.0, 'CDEJ' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 1.0, 'K' : 0.0})
elif(Scenario == 'I'):
	ScenRelAMIns ~= choice({'ABI' : 1.0, 'CDEJ' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'K' : 0.0})
elif(Scenario == 'J'):
	ScenRelAMIns ~= choice({'ABI' : 0.0, 'CDEJ' : 1.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'K' : 0.0})
else:
	ScenRelAMIns ~= choice({'ABI' : 0.0, 'CDEJ' : 0.0, 'F' : 0.0, 'G' : 0.0, 'H' : 0.0, 'K' : 1.0})


if (ScenRelAMCIN == 'AB'):
	if (MorningCIN == 'None'):
		AMCINInScen ~= choice({'LessThanAve' : 1.0, 'Average' : 0.0, 'MoreThanAve' : 0.0})
	elif(MorningCIN == 'PartInhibit'):
		AMCINInScen ~= choice({'LessThanAve' : 0.6, 'Average' : 0.37, 'MoreThanAve' : 0.030000000000000027})
	elif(MorningCIN == 'Stifling'):
		AMCINInScen ~= choice({'LessThanAve' : 0.25, 'Average' : 0.45, 'MoreThanAve' : 0.30000000000000004})
	else:
		AMCINInScen ~= choice({'LessThanAve' : 0.0, 'Average' : 0.1, 'MoreThanAve' : 0.9})
else:
	if(MorningCIN == 'None'):
		AMCINInScen ~= choice({'LessThanAve' : 0.75, 'Average' : 0.25, 'MoreThanAve' : 0.0})
	elif(MorningCIN == 'PartInhibit'):
		AMCINInScen ~= choice({'LessThanAve' : 0.3, 'Average' : 0.6, 'MoreThanAve' : 0.10000000000000009})
	elif(MorningCIN == 'Stifling'):
		AMCINInScen ~= choice({'LessThanAve' : 0.01, 'Average' : 0.4, 'MoreThanAve' : 0.59})
	else:
		AMCINInScen ~= choice({'LessThanAve' : 0.0, 'Average' : 0.03, 'MoreThanAve' : 0.97})


if (ScenRelAMIns == 'ABI'):
	if (LIfr12ZDENSd == 'LIGt0'):
		if (AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.6, 'Average' : 0.3, 'MoreUnstable' : 0.10000000000000009})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.85, 'Average' : 0.13, 'MoreUnstable' : 0.020000000000000018})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.95, 'Average' : 0.04, 'MoreUnstable' : 0.010000000000000009})
	elif(LIfr12ZDENSd == 'N1GtLIGt_4'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.3, 'Average' : 0.3, 'MoreUnstable' : 0.4})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.5, 'Average' : 0.3, 'MoreUnstable' : 0.19999999999999996})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.75, 'Average' : 0.2, 'MoreUnstable' : 0.050000000000000044})
	elif(LIfr12ZDENSd == 'N5GtLIGt_8'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.06, 'Average' : 0.21, 'MoreUnstable' : 0.73})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.2, 'Average' : 0.4, 'MoreUnstable' : 0.3999999999999999})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.5, 'Average' : 0.4, 'MoreUnstable' : 0.09999999999999998})
	else:
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.01, 'Average' : 0.04, 'MoreUnstable' : 0.95})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.05, 'Average' : 0.2, 'MoreUnstable' : 0.75})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.35, 'Average' : 0.35, 'MoreUnstable' : 0.30000000000000004})
elif(ScenRelAMIns == 'CDEJ'):
	if(LIfr12ZDENSd == 'LIGt0'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.4, 'Average' : 0.3, 'MoreUnstable' : 0.30000000000000004})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.7, 'Average' : 0.2, 'MoreUnstable' : 0.10000000000000009})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.9, 'Average' : 0.08, 'MoreUnstable' : 0.020000000000000018})
	elif(LIfr12ZDENSd == 'N1GtLIGt_4'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.15, 'Average' : 0.3, 'MoreUnstable' : 0.55})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.25, 'Average' : 0.5, 'MoreUnstable' : 0.25})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.6, 'Average' : 0.3, 'MoreUnstable' : 0.10000000000000009})
	elif(LIfr12ZDENSd == 'N5GtLIGt_8'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.03, 'Average' : 0.17, 'MoreUnstable' : 0.8})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.2, 'Average' : 0.3, 'MoreUnstable' : 0.5})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.45, 'Average' : 0.4, 'MoreUnstable' : 0.1499999999999999})
	else:
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.01, 'Average' : 0.04, 'MoreUnstable' : 0.95})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.05, 'Average' : 0.18, 'MoreUnstable' : 0.77})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.25, 'Average' : 0.4, 'MoreUnstable' : 0.35})
elif(ScenRelAMIns == 'F'):
	if(LIfr12ZDENSd == 'LIGt0'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.35, 'Average' : 0.35, 'MoreUnstable' : 0.30000000000000004})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.55, 'Average' : 0.4, 'MoreUnstable' : 0.04999999999999993})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.85, 'Average' : 0.13, 'MoreUnstable' : 0.020000000000000018})
	elif(LIfr12ZDENSd == 'N1GtLIGt_4'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.07, 'Average' : 0.38, 'MoreUnstable' : 0.55})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.2, 'Average' : 0.6, 'MoreUnstable' : 0.19999999999999996})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.5, 'Average' : 0.43, 'MoreUnstable' : 0.07000000000000006})
	elif(LIfr12ZDENSd == 'N5GtLIGt_8'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.0, 'Average' : 0.05, 'MoreUnstable' : 0.95})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.05, 'Average' : 0.35, 'MoreUnstable' : 0.6000000000000001})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.25, 'Average' : 0.5, 'MoreUnstable' : 0.25})
	else:
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.0, 'Average' : 0.02, 'MoreUnstable' : 0.98})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.0, 'Average' : 0.05, 'MoreUnstable' : 0.95})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.04, 'Average' : 0.16, 'MoreUnstable' : 0.8})
elif(ScenRelAMIns == 'G'):
	if(LIfr12ZDENSd == 'LIGt0'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.3, 'Average' : 0.4, 'MoreUnstable' : 0.30000000000000004})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.5, 'Average' : 0.3, 'MoreUnstable' : 0.19999999999999996})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.75, 'Average' : 0.2, 'MoreUnstable' : 0.050000000000000044})
	elif(LIfr12ZDENSd == 'N1GtLIGt_4'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.15, 'Average' : 0.35, 'MoreUnstable' : 0.5})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.2, 'Average' : 0.6, 'MoreUnstable' : 0.19999999999999996})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.15, 'Average' : 0.7, 'MoreUnstable' : 0.15000000000000002})
	elif(LIfr12ZDENSd == 'N5GtLIGt_8'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.07, 'Average' : 0.23, 'MoreUnstable' : 0.7})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.13, 'Average' : 0.47, 'MoreUnstable' : 0.4})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.1, 'Average' : 0.75, 'MoreUnstable' : 0.15000000000000002})
	else:
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.02, 'Average' : 0.18, 'MoreUnstable' : 0.8})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.04, 'Average' : 0.26, 'MoreUnstable' : 0.7})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.07, 'Average' : 0.3, 'MoreUnstable' : 0.63})
elif(ScenRelAMIns == 'H'):
	if(LIfr12ZDENSd == 'LIGt0'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.35, 'Average' : 0.45, 'MoreUnstable' : 0.19999999999999996})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.4, 'Average' : 0.5, 'MoreUnstable' : 0.09999999999999998})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.58, 'Average' : 0.4, 'MoreUnstable' : 0.020000000000000018})
	elif(LIfr12ZDENSd == 'N1GtLIGt_4'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.1, 'Average' : 0.25, 'MoreUnstable' : 0.65})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.15, 'Average' : 0.45, 'MoreUnstable' : 0.4})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.4, 'Average' : 0.45, 'MoreUnstable' : 0.1499999999999999})
	elif(LIfr12ZDENSd == 'N5GtLIGt_8'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.02, 'Average' : 0.18, 'MoreUnstable' : 0.8})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.05, 'Average' : 0.25, 'MoreUnstable' : 0.7})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.15, 'Average' : 0.35, 'MoreUnstable' : 0.5})
	else:
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.01, 'Average' : 0.09, 'MoreUnstable' : 0.9})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.03, 'Average' : 0.17, 'MoreUnstable' : 0.8})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.08, 'Average' : 0.32, 'MoreUnstable' : 0.6})
else:
	if(LIfr12ZDENSd == 'LIGt0'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.3, 'Average' : 0.55, 'MoreUnstable' : 0.1499999999999999})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.4, 'Average' : 0.5, 'MoreUnstable' : 0.09999999999999998})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.5, 'Average' : 0.43, 'MoreUnstable' : 0.07000000000000006})
	elif(LIfr12ZDENSd == 'N1GtLIGt_4'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.1, 'Average' : 0.35, 'MoreUnstable' : 0.55})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.25, 'Average' : 0.5, 'MoreUnstable' : 0.25})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.3, 'Average' : 0.5, 'MoreUnstable' : 0.19999999999999996})
	elif(LIfr12ZDENSd == 'N5GtLIGt_8'):
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.05, 'Average' : 0.22, 'MoreUnstable' : 0.73})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.1, 'Average' : 0.35, 'MoreUnstable' : 0.55})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.15, 'Average' : 0.35, 'MoreUnstable' : 0.5})
	else:
		if(AMDewptCalPl == 'Instability'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.02, 'Average' : 0.1, 'MoreUnstable' : 0.88})
		elif(AMDewptCalPl == 'Neutral'):
			AMInsWliScen ~= choice({'LessUnstable' : 0.04, 'Average' : 0.16, 'MoreUnstable' : 0.8})
		else:
			AMInsWliScen ~= choice({'LessUnstable' : 0.1, 'Average' : 0.25, 'MoreUnstable' : 0.65})


if (CombVerMo == 'StrongUp'):
	AreaMeso_ALS ~= choice({'StrongUp' : 1.0, 'WeakUp' : 0.0, 'Neutral' : 0.0, 'Down' : 0.0})
elif(CombVerMo == 'WeakUp'):
	AreaMeso_ALS ~= choice({'StrongUp' : 0.0, 'WeakUp' : 1.0, 'Neutral' : 0.0, 'Down' : 0.0})
elif(CombVerMo == 'Neutral'):
	AreaMeso_ALS ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 1.0, 'Down' : 0.0})
else:
	AreaMeso_ALS ~= choice({'StrongUp' : 0.0, 'WeakUp' : 0.0, 'Neutral' : 0.0, 'Down' : 1.0})


if (AreaMeso_ALS == 'StrongUp'):
	if (CombMoisture == 'VeryWet'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.99, 'Wet' : 0.01, 'Neutral' : 0.0, 'Dry' : 0.0})
	elif(CombMoisture == 'Wet'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.7, 'Wet' : 0.29, 'Neutral' : 0.01, 'Dry' : 0.0})
	elif(CombMoisture == 'Neutral'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.2, 'Wet' : 0.55, 'Neutral' : 0.24, 'Dry' : 0.010000000000000009})
	else:
		AreaMoDryAir ~= choice({'VeryWet' : 0.0, 'Wet' : 0.25, 'Neutral' : 0.55, 'Dry' : 0.19999999999999996})
elif(AreaMeso_ALS == 'WeakUp'):
	if(CombMoisture == 'VeryWet'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.8, 'Wet' : 0.2, 'Neutral' : 0.0, 'Dry' : 0.0})
	elif(CombMoisture == 'Wet'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.35, 'Wet' : 0.55, 'Neutral' : 0.1, 'Dry' : 0.0})
	elif(CombMoisture == 'Neutral'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.01, 'Wet' : 0.39, 'Neutral' : 0.55, 'Dry' : 0.04999999999999993})
	else:
		AreaMoDryAir ~= choice({'VeryWet' : 0.0, 'Wet' : 0.02, 'Neutral' : 0.43, 'Dry' : 0.55})
elif(AreaMeso_ALS == 'Neutral'):
	if(CombMoisture == 'VeryWet'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.7, 'Wet' : 0.29, 'Neutral' : 0.01, 'Dry' : 0.0})
	elif(CombMoisture == 'Wet'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.2, 'Wet' : 0.6, 'Neutral' : 0.2, 'Dry' : 0.0})
	elif(CombMoisture == 'Neutral'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.01, 'Wet' : 0.09, 'Neutral' : 0.8, 'Dry' : 0.09999999999999998})
	else:
		AreaMoDryAir ~= choice({'VeryWet' : 0.0, 'Wet' : 0.0, 'Neutral' : 0.3, 'Dry' : 0.7})
else:
	if(CombMoisture == 'VeryWet'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.2, 'Wet' : 0.74, 'Neutral' : 0.06, 'Dry' : 0.0})
	elif(CombMoisture == 'Wet'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.05, 'Wet' : 0.4, 'Neutral' : 0.45, 'Dry' : 0.09999999999999998})
	elif(CombMoisture == 'Neutral'):
		AreaMoDryAir ~= choice({'VeryWet' : 0.0, 'Wet' : 0.05, 'Neutral' : 0.5, 'Dry' : 0.44999999999999996})
	else:
		AreaMoDryAir ~= choice({'VeryWet' : 0.0, 'Wet' : 0.0, 'Neutral' : 0.01, 'Dry' : 0.99})


if (AreaMoDryAir == 'VeryWet'):
	if (AreaMeso_ALS == 'StrongUp'):
		if (CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 1.0, 'PC' : 0.0, 'Clear' : 0.0})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.85, 'PC' : 0.15, 'Clear' : 0.0})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.25, 'PC' : 0.35, 'Clear' : 0.4})
	elif(AreaMeso_ALS == 'WeakUp'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.95, 'PC' : 0.05, 'Clear' : 0.0})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.4, 'PC' : 0.55, 'Clear' : 0.04999999999999993})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.05, 'PC' : 0.45, 'Clear' : 0.5})
	elif(AreaMeso_ALS == 'Neutral'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.93, 'PC' : 0.07, 'Clear' : 0.0})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.2, 'PC' : 0.78, 'Clear' : 0.020000000000000018})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.01, 'PC' : 0.29, 'Clear' : 0.7})
	else:
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.74, 'PC' : 0.25, 'Clear' : 0.010000000000000009})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.5, 'Clear' : 0.5})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.1, 'Clear' : 0.9})
elif(AreaMoDryAir == 'Wet'):
	if(AreaMeso_ALS == 'StrongUp'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.92, 'PC' : 0.08, 'Clear' : 0.0})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.7, 'PC' : 0.29, 'Clear' : 0.010000000000000009})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.15, 'PC' : 0.4, 'Clear' : 0.44999999999999996})
	elif(AreaMeso_ALS == 'WeakUp'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.9, 'PC' : 0.09, 'Clear' : 0.010000000000000009})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.25, 'PC' : 0.6, 'Clear' : 0.15000000000000002})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.01, 'PC' : 0.3, 'Clear' : 0.69})
	elif(AreaMeso_ALS == 'Neutral'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.8, 'PC' : 0.2, 'Clear' : 0.0})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.01, 'PC' : 0.89, 'Clear' : 0.09999999999999998})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.1, 'Clear' : 0.9})
	else:
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.65, 'PC' : 0.34, 'Clear' : 0.010000000000000009})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.4, 'Clear' : 0.6})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.02, 'Clear' : 0.98})
elif(AreaMoDryAir == 'Neutral'):
	if(AreaMeso_ALS == 'StrongUp'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.88, 'PC' : 0.12, 'Clear' : 0.0})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.4, 'PC' : 0.5, 'Clear' : 0.09999999999999998})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.1, 'PC' : 0.4, 'Clear' : 0.5})
	elif(AreaMeso_ALS == 'WeakUp'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.85, 'PC' : 0.15, 'Clear' : 0.0})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.15, 'PC' : 0.75, 'Clear' : 0.09999999999999998})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.2, 'Clear' : 0.8})
	elif(AreaMeso_ALS == 'Neutral'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.8, 'PC' : 0.18, 'Clear' : 0.020000000000000018})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.03, 'PC' : 0.85, 'Clear' : 0.12})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.05, 'Clear' : 0.95})
	else:
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.5, 'PC' : 0.48, 'Clear' : 0.020000000000000018})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.01, 'PC' : 0.74, 'Clear' : 0.25})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.01, 'Clear' : 0.99})
else:
	if(AreaMeso_ALS == 'StrongUp'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.85, 'PC' : 0.14, 'Clear' : 0.010000000000000009})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.55, 'PC' : 0.43, 'Clear' : 0.020000000000000018})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.1, 'PC' : 0.25, 'Clear' : 0.65})
	elif(AreaMeso_ALS == 'WeakUp'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.6, 'PC' : 0.39, 'Clear' : 0.010000000000000009})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.01, 'PC' : 0.9, 'Clear' : 0.08999999999999997})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.15, 'Clear' : 0.85})
	elif(AreaMeso_ALS == 'Neutral'):
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.78, 'PC' : 0.2, 'Clear' : 0.020000000000000018})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.01, 'PC' : 0.74, 'Clear' : 0.25})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.04, 'Clear' : 0.96})
	else:
		if(CombClouds == 'Cloudy'):
			CldShadeOth ~= choice({'Cloudy' : 0.42, 'PC' : 0.55, 'Clear' : 0.030000000000000027})
		elif(CombClouds == 'PC'):
			CldShadeOth ~= choice({'Cloudy' : 0.05, 'PC' : 0.65, 'Clear' : 0.29999999999999993})
		else:
			CldShadeOth ~= choice({'Cloudy' : 0.0, 'PC' : 0.0, 'Clear' : 1.0})


if (CldShadeOth == 'Cloudy'):
	if (AMInstabMt == 'None'):
		InsInMt ~= choice({'None' : 0.9, 'Weak' : 0.1, 'Strong' : 0.0})
	elif(AMInstabMt == 'Weak'):
		InsInMt ~= choice({'None' : 0.01, 'Weak' : 0.4, 'Strong' : 0.59})
	else:
		InsInMt ~= choice({'None' : 0.0, 'Weak' : 0.05, 'Strong' : 0.95})
elif(CldShadeOth == 'PC'):
	if(AMInstabMt == 'None'):
		InsInMt ~= choice({'None' : 0.6, 'Weak' : 0.39, 'Strong' : 0.010000000000000009})
	elif(AMInstabMt == 'Weak'):
		InsInMt ~= choice({'None' : 0.0, 'Weak' : 0.4, 'Strong' : 0.6})
	else:
		InsInMt ~= choice({'None' : 0.0, 'Weak' : 0.0, 'Strong' : 1.0})
else:
	if(AMInstabMt == 'None'):
		InsInMt ~= choice({'None' : 0.5, 'Weak' : 0.35, 'Strong' : 0.15000000000000002})
	elif(AMInstabMt == 'Weak'):
		InsInMt ~= choice({'None' : 0.0, 'Weak' : 0.15, 'Strong' : 0.85})
	else:
		InsInMt ~= choice({'None' : 0.0, 'Weak' : 0.0, 'Strong' : 1.0})


if (InsInMt == 'None'):
	MountainFcst ~= choice({'XNIL' : 1.0, 'SIG' : 0.0, 'SVR' : 0.0})
elif(InsInMt == 'Weak'):
	MountainFcst ~= choice({'XNIL' : 0.48, 'SIG' : 0.5, 'SVR' : 0.020000000000000018})
else:
	MountainFcst ~= choice({'XNIL' : 0.2, 'SIG' : 0.5, 'SVR' : 0.30000000000000004})


if (InsInMt == 'None'):
	if (WndHodograph == 'DCVZFavor'):
		OutflowFrMt ~= choice({'None' : 1.0, 'Weak' : 0.0, 'Strong' : 0.0})
	elif(WndHodograph == 'StrongWest'):
		OutflowFrMt ~= choice({'None' : 1.0, 'Weak' : 0.0, 'Strong' : 0.0})
	elif(WndHodograph == 'Westerly'):
		OutflowFrMt ~= choice({'None' : 1.0, 'Weak' : 0.0, 'Strong' : 0.0})
	else:
		OutflowFrMt ~= choice({'None' : 1.0, 'Weak' : 0.0, 'Strong' : 0.0})
elif(InsInMt == 'Weak'):
	if(WndHodograph == 'DCVZFavor'):
		OutflowFrMt ~= choice({'None' : 0.5, 'Weak' : 0.4, 'Strong' : 0.09999999999999998})
	elif(WndHodograph == 'StrongWest'):
		OutflowFrMt ~= choice({'None' : 0.15, 'Weak' : 0.4, 'Strong' : 0.44999999999999996})
	elif(WndHodograph == 'Westerly'):
		OutflowFrMt ~= choice({'None' : 0.35, 'Weak' : 0.6, 'Strong' : 0.050000000000000044})
	else:
		OutflowFrMt ~= choice({'None' : 0.8, 'Weak' : 0.19, 'Strong' : 0.010000000000000009})
else:
	if(WndHodograph == 'DCVZFavor'):
		OutflowFrMt ~= choice({'None' : 0.05, 'Weak' : 0.45, 'Strong' : 0.5})
	elif(WndHodograph == 'StrongWest'):
		OutflowFrMt ~= choice({'None' : 0.01, 'Weak' : 0.15, 'Strong' : 0.84})
	elif(WndHodograph == 'Westerly'):
		OutflowFrMt ~= choice({'None' : 0.1, 'Weak' : 0.25, 'Strong' : 0.65})
	else:
		OutflowFrMt ~= choice({'None' : 0.6, 'Weak' : 0.3, 'Strong' : 0.10000000000000009})


if (OutflowFrMt == 'None'):
	if (WndHodograph == 'DCVZFavor'):
		if (MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.5, 'Weak' : 0.48, 'Strong' : 0.020000000000000018})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.3, 'Weak' : 0.5, 'Strong' : 0.19999999999999996})
		else:
			Boundaries ~= choice({'None' : 0.1, 'Weak' : 0.25, 'Strong' : 0.65})
	elif(WndHodograph == 'StrongWest'):
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.75, 'Weak' : 0.22, 'Strong' : 0.030000000000000027})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.45, 'Weak' : 0.45, 'Strong' : 0.09999999999999998})
		else:
			Boundaries ~= choice({'None' : 0.25, 'Weak' : 0.4, 'Strong' : 0.35})
	elif(WndHodograph == 'Westerly'):
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.8, 'Weak' : 0.18, 'Strong' : 0.020000000000000018})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.35, 'Weak' : 0.5, 'Strong' : 0.15000000000000002})
		else:
			Boundaries ~= choice({'None' : 0.25, 'Weak' : 0.35, 'Strong' : 0.4})
	else:
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.7, 'Weak' : 0.28, 'Strong' : 0.020000000000000018})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.25, 'Weak' : 0.6, 'Strong' : 0.15000000000000002})
		else:
			Boundaries ~= choice({'None' : 0.05, 'Weak' : 0.35, 'Strong' : 0.6000000000000001})
elif(OutflowFrMt == 'Weak'):
	if(WndHodograph == 'DCVZFavor'):
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.3, 'Weak' : 0.63, 'Strong' : 0.07000000000000006})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.1, 'Weak' : 0.5, 'Strong' : 0.4})
		else:
			Boundaries ~= choice({'None' : 0.05, 'Weak' : 0.2, 'Strong' : 0.75})
	elif(WndHodograph == 'StrongWest'):
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.15, 'Weak' : 0.7, 'Strong' : 0.15000000000000002})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.1, 'Weak' : 0.75, 'Strong' : 0.15000000000000002})
		else:
			Boundaries ~= choice({'None' : 0.05, 'Weak' : 0.5, 'Strong' : 0.44999999999999996})
	elif(WndHodograph == 'Westerly'):
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.15, 'Weak' : 0.7, 'Strong' : 0.15000000000000002})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.05, 'Weak' : 0.8, 'Strong' : 0.1499999999999999})
		else:
			Boundaries ~= choice({'None' : 0.05, 'Weak' : 0.45, 'Strong' : 0.5})
	else:
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.4, 'Weak' : 0.55, 'Strong' : 0.04999999999999993})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.2, 'Weak' : 0.65, 'Strong' : 0.1499999999999999})
		else:
			Boundaries ~= choice({'None' : 0.05, 'Weak' : 0.3, 'Strong' : 0.65})
else:
	if(WndHodograph == 'DCVZFavor'):
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.0, 'Weak' : 0.55, 'Strong' : 0.44999999999999996})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.0, 'Weak' : 0.4, 'Strong' : 0.6})
		else:
			Boundaries ~= choice({'None' : 0.0, 'Weak' : 0.15, 'Strong' : 0.85})
	elif(WndHodograph == 'StrongWest'):
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.0, 'Weak' : 0.5, 'Strong' : 0.5})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.0, 'Weak' : 0.4, 'Strong' : 0.6})
		else:
			Boundaries ~= choice({'None' : 0.0, 'Weak' : 0.2, 'Strong' : 0.8})
	elif(WndHodograph == 'Westerly'):
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.0, 'Weak' : 0.7, 'Strong' : 0.30000000000000004})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.0, 'Weak' : 0.5, 'Strong' : 0.5})
		else:
			Boundaries ~= choice({'None' : 0.0, 'Weak' : 0.2, 'Strong' : 0.8})
	else:
		if(MorningBound == 'None'):
			Boundaries ~= choice({'None' : 0.02, 'Weak' : 0.73, 'Strong' : 0.25})
		elif(MorningBound == 'Weak'):
			Boundaries ~= choice({'None' : 0.01, 'Weak' : 0.5, 'Strong' : 0.49})
		else:
			Boundaries ~= choice({'None' : 0.01, 'Weak' : 0.2, 'Strong' : 0.79})


if (InsInMt == 'None'):
	if (WndHodograph == 'DCVZFavor'):
		CldShadeConv ~= choice({'None' : 1.0, 'Some' : 0.0, 'Marked' : 0.0})
	elif(WndHodograph == 'StrongWest'):
		CldShadeConv ~= choice({'None' : 1.0, 'Some' : 0.0, 'Marked' : 0.0})
	elif(WndHodograph == 'Westerly'):
		CldShadeConv ~= choice({'None' : 1.0, 'Some' : 0.0, 'Marked' : 0.0})
	else:
		CldShadeConv ~= choice({'None' : 1.0, 'Some' : 0.0, 'Marked' : 0.0})
elif(InsInMt == 'Weak'):
	if(WndHodograph == 'DCVZFavor'):
		CldShadeConv ~= choice({'None' : 0.3, 'Some' : 0.6, 'Marked' : 0.10000000000000009})
	elif(WndHodograph == 'StrongWest'):
		CldShadeConv ~= choice({'None' : 0.2, 'Some' : 0.7, 'Marked' : 0.10000000000000009})
	elif(WndHodograph == 'Westerly'):
		CldShadeConv ~= choice({'None' : 0.5, 'Some' : 0.46, 'Marked' : 0.040000000000000036})
	else:
		CldShadeConv ~= choice({'None' : 0.8, 'Some' : 0.19, 'Marked' : 0.010000000000000009})
else:
	if(WndHodograph == 'DCVZFavor'):
		CldShadeConv ~= choice({'None' : 0.0, 'Some' : 0.3, 'Marked' : 0.7})
	elif(WndHodograph == 'StrongWest'):
		CldShadeConv ~= choice({'None' : 0.0, 'Some' : 0.2, 'Marked' : 0.8})
	elif(WndHodograph == 'Westerly'):
		CldShadeConv ~= choice({'None' : 0.1, 'Some' : 0.5, 'Marked' : 0.4})
	else:
		CldShadeConv ~= choice({'None' : 0.5, 'Some' : 0.38, 'Marked' : 0.12})


if (Boundaries == 'None'):
	if (CldShadeConv == 'None'):
		if (AreaMeso_ALS == 'StrongUp'):
			if (CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.4, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.25})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.1, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.55})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.05, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.65})
		elif(AreaMeso_ALS == 'WeakUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.6, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.15000000000000002})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.4, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.30000000000000004})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.2, 'LittleChange' : 0.5, 'DecCapIncIns' : 0.30000000000000004})
		elif(AreaMeso_ALS == 'Neutral'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.6, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.050000000000000044})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.45, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.1499999999999999})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.25, 'LittleChange' : 0.45, 'DecCapIncIns' : 0.30000000000000004})
		else:
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.7, 'LittleChange' : 0.27, 'DecCapIncIns' : 0.030000000000000027})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.65, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.050000000000000044})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.6, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.050000000000000044})
	elif(CldShadeConv == 'Some'):
		if(AreaMeso_ALS == 'StrongUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.4, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.25})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.25, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.44999999999999996})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.15, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.5})
		elif(AreaMeso_ALS == 'WeakUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.65, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.09999999999999998})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.45, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.25})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.25, 'LittleChange' : 0.5, 'DecCapIncIns' : 0.25})
		elif(AreaMeso_ALS == 'Neutral'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.65, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.050000000000000044})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.5, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.09999999999999998})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.3, 'LittleChange' : 0.45, 'DecCapIncIns' : 0.25})
		else:
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.75, 'LittleChange' : 0.23, 'DecCapIncIns' : 0.020000000000000018})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.7, 'LittleChange' : 0.26, 'DecCapIncIns' : 0.040000000000000036})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.65, 'LittleChange' : 0.32, 'DecCapIncIns' : 0.030000000000000027})
	else:
		if(AreaMeso_ALS == 'StrongUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.45, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.25})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.4, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.30000000000000004})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.35, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.3500000000000001})
		elif(AreaMeso_ALS == 'WeakUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.7, 'LittleChange' : 0.22, 'DecCapIncIns' : 0.08000000000000007})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.55, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.1499999999999999})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.4, 'LittleChange' : 0.45, 'DecCapIncIns' : 0.1499999999999999})
		elif(AreaMeso_ALS == 'Neutral'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.7, 'LittleChange' : 0.27, 'DecCapIncIns' : 0.030000000000000027})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.6, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.10000000000000009})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.55, 'LittleChange' : 0.33, 'DecCapIncIns' : 0.11999999999999988})
		else:
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.85, 'LittleChange' : 0.14, 'DecCapIncIns' : 0.010000000000000009})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.8, 'LittleChange' : 0.17, 'DecCapIncIns' : 0.029999999999999916})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.75, 'LittleChange' : 0.23, 'DecCapIncIns' : 0.020000000000000018})
elif(Boundaries == 'Weak'):
	if(CldShadeConv == 'None'):
		if(AreaMeso_ALS == 'StrongUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.35, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.30000000000000004})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.05, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.6000000000000001})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.03, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.72})
		elif(AreaMeso_ALS == 'WeakUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.5, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.25})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.3, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.3500000000000001})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.15, 'LittleChange' : 0.45, 'DecCapIncIns' : 0.4})
		elif(AreaMeso_ALS == 'Neutral'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.55, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.1499999999999999})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.4, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.19999999999999996})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.2, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.3999999999999999})
		else:
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.6, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.050000000000000044})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.6, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.10000000000000009})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.55, 'LittleChange' : 0.33, 'DecCapIncIns' : 0.11999999999999988})
	elif(CldShadeConv == 'Some'):
		if(AreaMeso_ALS == 'StrongUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.35, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.30000000000000004})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.1, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.55})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.05, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.65})
		elif(AreaMeso_ALS == 'WeakUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.55, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.19999999999999996})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.35, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.30000000000000004})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.2, 'LittleChange' : 0.5, 'DecCapIncIns' : 0.30000000000000004})
		elif(AreaMeso_ALS == 'Neutral'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.6, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.10000000000000009})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.45, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.1499999999999999})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.25, 'LittleChange' : 0.5, 'DecCapIncIns' : 0.25})
		else:
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.65, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.050000000000000044})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.65, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.050000000000000044})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.6, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.050000000000000044})
	else:
		if(AreaMeso_ALS == 'StrongUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.4, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.25})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.25, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.35})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.2, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.3999999999999999})
		elif(AreaMeso_ALS == 'WeakUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.65, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.09999999999999998})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.45, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.19999999999999996})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.3, 'LittleChange' : 0.5, 'DecCapIncIns' : 0.19999999999999996})
		elif(AreaMeso_ALS == 'Neutral'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.65, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.050000000000000044})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.55, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.1499999999999999})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.5, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.19999999999999996})
		else:
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.78, 'LittleChange' : 0.18, 'DecCapIncIns' : 0.040000000000000036})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.75, 'LittleChange' : 0.2, 'DecCapIncIns' : 0.050000000000000044})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.7, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.050000000000000044})
else:
	if(CldShadeConv == 'None'):
		if(AreaMeso_ALS == 'StrongUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.3, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.4})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.01, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.74})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.01, 'LittleChange' : 0.2, 'DecCapIncIns' : 0.79})
		elif(AreaMeso_ALS == 'WeakUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.35, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.4})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.15, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.44999999999999996})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.1, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.55})
		elif(AreaMeso_ALS == 'Neutral'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.45, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.25})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.3, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.30000000000000004})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.15, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.44999999999999996})
		else:
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.5, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.15000000000000002})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.48, 'LittleChange' : 0.32, 'DecCapIncIns' : 0.19999999999999996})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.45, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.19999999999999996})
	elif(CldShadeConv == 'Some'):
		if(AreaMeso_ALS == 'StrongUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.3, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.4})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.05, 'LittleChange' : 0.6, 'DecCapIncIns' : 0.35})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.04, 'LittleChange' : 0.27, 'DecCapIncIns' : 0.69})
		elif(AreaMeso_ALS == 'WeakUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.4, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.35})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.2, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.3999999999999999})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.12, 'LittleChange' : 0.43, 'DecCapIncIns' : 0.44999999999999996})
		elif(AreaMeso_ALS == 'Neutral'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.5, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.19999999999999996})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.35, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.25})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.2, 'LittleChange' : 0.45, 'DecCapIncIns' : 0.35})
		else:
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.55, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.09999999999999998})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.55, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.1499999999999999})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.5, 'LittleChange' : 0.4, 'DecCapIncIns' : 0.09999999999999998})
	else:
		if(AreaMeso_ALS == 'StrongUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.3, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.3500000000000001})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.15, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.5})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.13, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.52})
		elif(AreaMeso_ALS == 'WeakUp'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.5, 'LittleChange' : 0.25, 'DecCapIncIns' : 0.25})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.35, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.30000000000000004})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.2, 'LittleChange' : 0.45, 'DecCapIncIns' : 0.35})
		elif(AreaMeso_ALS == 'Neutral'):
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.55, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.09999999999999998})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.45, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.19999999999999996})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.4, 'LittleChange' : 0.35, 'DecCapIncIns' : 0.25})
		else:
			if(CldShadeOth == 'Cloudy'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.7, 'LittleChange' : 0.24, 'DecCapIncIns' : 0.06000000000000005})
			elif(CldShadeOth == 'PC'):
				CompPlFcst ~= choice({'IncCapDecIns' : 0.65, 'LittleChange' : 0.28, 'DecCapIncIns' : 0.06999999999999995})
			else:
				CompPlFcst ~= choice({'IncCapDecIns' : 0.6, 'LittleChange' : 0.3, 'DecCapIncIns' : 0.10000000000000009})


if (LoLevMoistAd == 'StrongPos'):
	if (CompPlFcst == 'IncCapDecIns'):
		InsChange ~= choice({'Decreasing' : 0.0, 'LittleChange' : 0.05, 'Increasing' : 0.95})
	elif(CompPlFcst == 'LittleChange'):
		InsChange ~= choice({'Decreasing' : 0.0, 'LittleChange' : 0.12, 'Increasing' : 0.88})
	else:
		InsChange ~= choice({'Decreasing' : 0.05, 'LittleChange' : 0.15, 'Increasing' : 0.8})
elif(LoLevMoistAd == 'WeakPos'):
	if(CompPlFcst == 'IncCapDecIns'):
		InsChange ~= choice({'Decreasing' : 0.05, 'LittleChange' : 0.15, 'Increasing' : 0.8})
	elif(CompPlFcst == 'LittleChange'):
		InsChange ~= choice({'Decreasing' : 0.1, 'LittleChange' : 0.4, 'Increasing' : 0.5})
	else:
		InsChange ~= choice({'Decreasing' : 0.25, 'LittleChange' : 0.5, 'Increasing' : 0.25})
elif(LoLevMoistAd == 'Neutral'):
	if(CompPlFcst == 'IncCapDecIns'):
		InsChange ~= choice({'Decreasing' : 0.15, 'LittleChange' : 0.5, 'Increasing' : 0.35})
	elif(CompPlFcst == 'LittleChange'):
		InsChange ~= choice({'Decreasing' : 0.2, 'LittleChange' : 0.6, 'Increasing' : 0.19999999999999996})
	else:
		InsChange ~= choice({'Decreasing' : 0.35, 'LittleChange' : 0.5, 'Increasing' : 0.15000000000000002})
else:
	if(CompPlFcst == 'IncCapDecIns'):
		InsChange ~= choice({'Decreasing' : 0.5, 'LittleChange' : 0.4, 'Increasing' : 0.09999999999999998})
	elif(CompPlFcst == 'LittleChange'):
		InsChange ~= choice({'Decreasing' : 0.8, 'LittleChange' : 0.16, 'Increasing' : 0.039999999999999925})
	else:
		InsChange ~= choice({'Decreasing' : 0.9, 'LittleChange' : 0.09, 'Increasing' : 0.010000000000000009})


if (AMInsWliScen == 'LessUnstable'):
	if (InsChange == 'Decreasing'):
		InsSclInScen ~= choice({'LessUnstable' : 1.0, 'Average' : 0.0, 'MoreUnstable' : 0.0})
	elif(InsChange == 'LittleChange'):
		InsSclInScen ~= choice({'LessUnstable' : 0.9, 'Average' : 0.1, 'MoreUnstable' : 0.0})
	else:
		InsSclInScen ~= choice({'LessUnstable' : 0.4, 'Average' : 0.35, 'MoreUnstable' : 0.25})
elif(AMInsWliScen == 'Average'):
	if(InsChange == 'Decreasing'):
		InsSclInScen ~= choice({'LessUnstable' : 0.6, 'Average' : 0.4, 'MoreUnstable' : 0.0})
	elif(InsChange == 'LittleChange'):
		InsSclInScen ~= choice({'LessUnstable' : 0.15, 'Average' : 0.7, 'MoreUnstable' : 0.15000000000000002})
	else:
		InsSclInScen ~= choice({'LessUnstable' : 0.0, 'Average' : 0.4, 'MoreUnstable' : 0.6})
else:
	if(InsChange == 'Decreasing'):
		InsSclInScen ~= choice({'LessUnstable' : 0.25, 'Average' : 0.35, 'MoreUnstable' : 0.4})
	elif(InsChange == 'LittleChange'):
		InsSclInScen ~= choice({'LessUnstable' : 0.0, 'Average' : 0.1, 'MoreUnstable' : 0.9})
	else:
		InsSclInScen ~= choice({'LessUnstable' : 0.0, 'Average' : 0.0, 'MoreUnstable' : 1.0})


if (CompPlFcst == 'IncCapDecIns'):
	CapChange ~= choice({'Decreasing' : 0.0, 'LittleChange' : 0.0, 'Increasing' : 1.0})
elif(CompPlFcst == 'LittleChange'):
	CapChange ~= choice({'Decreasing' : 0.0, 'LittleChange' : 1.0, 'Increasing' : 0.0})
else:
	CapChange ~= choice({'Decreasing' : 1.0, 'LittleChange' : 0.0, 'Increasing' : 0.0})


if (AMCINInScen == 'LessThanAve'):
	if (CapChange == 'Decreasing'):
		CapInScen ~= choice({'LessThanAve' : 1.0, 'Average' : 0.0, 'MoreThanAve' : 0.0})
	elif(CapChange == 'LittleChange'):
		CapInScen ~= choice({'LessThanAve' : 0.98, 'Average' : 0.02, 'MoreThanAve' : 0.0})
	else:
		CapInScen ~= choice({'LessThanAve' : 0.35, 'Average' : 0.35, 'MoreThanAve' : 0.30000000000000004})
elif(AMCINInScen == 'Average'):
	if(CapChange == 'Decreasing'):
		CapInScen ~= choice({'LessThanAve' : 0.75, 'Average' : 0.25, 'MoreThanAve' : 0.0})
	elif(CapChange == 'LittleChange'):
		CapInScen ~= choice({'LessThanAve' : 0.03, 'Average' : 0.94, 'MoreThanAve' : 0.030000000000000027})
	else:
		CapInScen ~= choice({'LessThanAve' : 0.0, 'Average' : 0.25, 'MoreThanAve' : 0.75})
else:
	if(CapChange == 'Decreasing'):
		CapInScen ~= choice({'LessThanAve' : 0.3, 'Average' : 0.35, 'MoreThanAve' : 0.3500000000000001})
	elif(CapChange == 'LittleChange'):
		CapInScen ~= choice({'LessThanAve' : 0.0, 'Average' : 0.02, 'MoreThanAve' : 0.98})
	else:
		CapInScen ~= choice({'LessThanAve' : 0.0, 'Average' : 0.0, 'MoreThanAve' : 1.0})


if (CurPropConv == 'None'):
	if (InsSclInScen == 'LessUnstable'):
		if (CapInScen == 'LessThanAve'):
			if (ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.08, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.06, 'SVR' : 0.040000000000000036})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.88, 'SIG' : 0.1, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.92, 'SIG' : 0.08, 'SVR' : 0.0})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.13, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 1.0, 'SIG' : 0.0, 'SVR' : 0.0})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.08, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.08, 'SVR' : 0.020000000000000018})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.95, 'SIG' : 0.04, 'SVR' : 0.010000000000000009})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.3, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.08, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.91, 'SIG' : 0.05, 'SVR' : 0.039999999999999925})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.13, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.1, 'SVR' : 0.0})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.84, 'SIG' : 0.12, 'SVR' : 0.040000000000000036})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.99, 'SIG' : 0.01, 'SVR' : 0.0})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.88, 'SIG' : 0.1, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.92, 'SIG' : 0.06, 'SVR' : 0.020000000000000018})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.96, 'SIG' : 0.03, 'SVR' : 0.010000000000000009})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.95, 'SIG' : 0.04, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.93, 'SIG' : 0.04, 'SVR' : 0.029999999999999916})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.92, 'SIG' : 0.06, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.87, 'SIG' : 0.13, 'SVR' : 0.0})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.06, 'SVR' : 0.040000000000000036})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.98, 'SIG' : 0.02, 'SVR' : 0.0})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.92, 'SIG' : 0.06, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.95, 'SIG' : 0.04, 'SVR' : 0.010000000000000009})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.97, 'SIG' : 0.02, 'SVR' : 0.010000000000000009})
	elif(InsSclInScen == 'Average'):
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.3, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.3, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.14, 'SVR' : 0.05999999999999994})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.09, 'SVR' : 0.06000000000000005})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.1, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.88, 'SIG' : 0.11, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.17, 'SVR' : 0.029999999999999916})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.92, 'SIG' : 0.06, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.12, 'SVR' : 0.07999999999999996})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.22, 'SVR' : 0.030000000000000027})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.08, 'SVR' : 0.020000000000000018})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.3, 'SVR' : 0.3500000000000001})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.3, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.82, 'SIG' : 0.13, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.82, 'SIG' : 0.1, 'SVR' : 0.08000000000000007})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.18, 'SVR' : 0.07000000000000006})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.88, 'SIG' : 0.11, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.07, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.2, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.15, 'SVR' : 0.04999999999999993})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.08, 'SVR' : 0.020000000000000018})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.2, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.25, 'SVR' : 0.15000000000000002})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.1, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.07, 'SVR' : 0.08000000000000007})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.15, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.14, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.94, 'SIG' : 0.05, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.22, 'SVR' : 0.13})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.83, 'SIG' : 0.1, 'SVR' : 0.07000000000000006})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.93, 'SIG' : 0.06, 'SVR' : 0.010000000000000009})
	else:
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.2, 'SVR' : 0.44999999999999996})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.35, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.1, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.72, 'SIG' : 0.14, 'SVR' : 0.14})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.78, 'SIG' : 0.15, 'SVR' : 0.06999999999999995})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.86, 'SIG' : 0.12, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.25, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.1, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.2, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.72, 'SIG' : 0.2, 'SVR' : 0.08000000000000007})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.1, 'SVR' : 0.050000000000000044})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.15, 'SVR' : 0.6})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.35, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.2, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.2, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.25, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.81, 'SIG' : 0.17, 'SVR' : 0.019999999999999907})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.28, 'SVR' : 0.12})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.13, 'SVR' : 0.06999999999999995})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.2, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.15, 'SVR' : 0.09999999999999998})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.88, 'SIG' : 0.08, 'SVR' : 0.040000000000000036})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.08, 'SVR' : 0.52})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.25, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.1, 'SVR' : 0.15000000000000002})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.15, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.52, 'SIG' : 0.25, 'SVR' : 0.22999999999999998})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.82, 'SIG' : 0.16, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.27, 'SVR' : 0.07999999999999996})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.09, 'SVR' : 0.06000000000000005})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.2, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.77, 'SIG' : 0.1, 'SVR' : 0.13})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.07, 'SVR' : 0.030000000000000027})
elif(CurPropConv == 'Slight'):
	if(InsSclInScen == 'LessUnstable'):
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.25, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.33, 'SVR' : 0.07000000000000006})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.82, 'SIG' : 0.13, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.1, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.82, 'SIG' : 0.15, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.14, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.17, 'SVR' : 0.029999999999999916})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.97, 'SIG' : 0.02, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.88, 'SIG' : 0.1, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.86, 'SIG' : 0.1, 'SVR' : 0.040000000000000036})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.88, 'SIG' : 0.1, 'SVR' : 0.020000000000000018})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.25, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.58, 'SIG' : 0.32, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.15, 'SVR' : 0.04999999999999993})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.1, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.16, 'SVR' : 0.039999999999999925})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.83, 'SIG' : 0.16, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.77, 'SIG' : 0.17, 'SVR' : 0.05999999999999994})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.93, 'SIG' : 0.06, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.12, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.1, 'SVR' : 0.050000000000000044})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.08, 'SVR' : 0.020000000000000018})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.3, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.28, 'SVR' : 0.06999999999999995})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.08, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.1, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.82, 'SIG' : 0.13, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.19, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.13, 'SVR' : 0.06999999999999995})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.91, 'SIG' : 0.08, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.12, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.9, 'SIG' : 0.08, 'SVR' : 0.020000000000000018})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.93, 'SIG' : 0.06, 'SVR' : 0.010000000000000009})
	elif(InsSclInScen == 'Average'):
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.4, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.34, 'SVR' : 0.10999999999999988})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.2, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.15, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.62, 'SIG' : 0.28, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.14, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.82, 'SIG' : 0.14, 'SVR' : 0.040000000000000036})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.25, 'SVR' : 0.15000000000000002})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.68, 'SIG' : 0.22, 'SVR' : 0.09999999999999998})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.82, 'SIG' : 0.15, 'SVR' : 0.030000000000000027})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.28, 'SIG' : 0.37, 'SVR' : 0.35})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.48, 'SIG' : 0.35, 'SVR' : 0.17000000000000004})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.2, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.17, 'SVR' : 0.13})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.29, 'SVR' : 0.1100000000000001})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.82, 'SIG' : 0.16, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.63, 'SIG' : 0.3, 'SVR' : 0.07000000000000006})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.15, 'SVR' : 0.04999999999999993})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.3, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.2, 'SVR' : 0.10000000000000009})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.16, 'SVR' : 0.039999999999999925})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.28, 'SVR' : 0.31999999999999995})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.25, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.72, 'SIG' : 0.18, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.2, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.3, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.78, 'SIG' : 0.2, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.35, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.12, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.3, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.73, 'SIG' : 0.15, 'SVR' : 0.12})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.12, 'SVR' : 0.030000000000000027})
	else:
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.25, 'SVR' : 0.44999999999999996})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.36, 'SVR' : 0.24})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.2, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.2, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.28, 'SVR' : 0.12})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.83, 'SIG' : 0.14, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.4, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.18, 'SVR' : 0.1200000000000001})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.25, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.25, 'SVR' : 0.15000000000000002})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.72, 'SIG' : 0.2, 'SVR' : 0.08000000000000007})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.22, 'SIG' : 0.17, 'SVR' : 0.61})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.37, 'SVR' : 0.28})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.3, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.25, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.48, 'SIG' : 0.29, 'SVR' : 0.22999999999999998})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.72, 'SIG' : 0.25, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.43, 'SIG' : 0.4, 'SVR' : 0.16999999999999993})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.68, 'SIG' : 0.2, 'SVR' : 0.11999999999999988})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.3, 'SVR' : 0.3500000000000001})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.2, 'SVR' : 0.19999999999999996})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.74, 'SIG' : 0.16, 'SVR' : 0.09999999999999998})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.27, 'SIG' : 0.1, 'SVR' : 0.63})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.3, 'SVR' : 0.3500000000000001})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.22, 'SVR' : 0.22999999999999998})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.25, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.42, 'SIG' : 0.3, 'SVR' : 0.28})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.74, 'SIG' : 0.22, 'SVR' : 0.040000000000000036})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.4, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.77, 'SIG' : 0.13, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.25, 'SVR' : 0.44999999999999996})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.68, 'SIG' : 0.15, 'SVR' : 0.16999999999999993})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.15, 'SVR' : 0.09999999999999998})
elif(CurPropConv == 'Moderate'):
	if(InsSclInScen == 'LessUnstable'):
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.4, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.42, 'SVR' : 0.13})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.18, 'SVR' : 0.07000000000000006})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.15, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.72, 'SIG' : 0.22, 'SVR' : 0.06000000000000005})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.78, 'SIG' : 0.21, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.66, 'SIG' : 0.27, 'SVR' : 0.06999999999999995})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.88, 'SIG' : 0.1, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.22, 'SVR' : 0.08000000000000007})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.78, 'SIG' : 0.16, 'SVR' : 0.05999999999999994})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.16, 'SVR' : 0.039999999999999925})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.35, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.35, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.2, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.72, 'SIG' : 0.17, 'SVR' : 0.10999999999999999})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.22, 'SVR' : 0.08000000000000007})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.24, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.62, 'SIG' : 0.3, 'SVR' : 0.08000000000000007})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.85, 'SIG' : 0.12, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.15, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.76, 'SIG' : 0.17, 'SVR' : 0.06999999999999995})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.16, 'SVR' : 0.039999999999999925})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.4, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.4, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.19, 'SVR' : 0.06000000000000005})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.2, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.3, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.72, 'SIG' : 0.27, 'SVR' : 0.010000000000000009})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.3, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.16, 'SVR' : 0.039999999999999925})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.17, 'SVR' : 0.07999999999999996})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.88, 'SIG' : 0.1, 'SVR' : 0.020000000000000018})
	elif(InsSclInScen == 'Average'):
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.2, 'SIG' : 0.45, 'SVR' : 0.35})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.4, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.2, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.22, 'SVR' : 0.13})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.34, 'SVR' : 0.15999999999999992})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.74, 'SIG' : 0.24, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.3, 'SVR' : 0.10000000000000009})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.67, 'SIG' : 0.24, 'SVR' : 0.08999999999999997})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.4, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.25, 'SVR' : 0.15000000000000002})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.23, 'SIG' : 0.4, 'SVR' : 0.37})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.38, 'SIG' : 0.35, 'SVR' : 0.27})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.58, 'SIG' : 0.25, 'SVR' : 0.17000000000000004})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.25, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.53, 'SIG' : 0.32, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.73, 'SIG' : 0.25, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.53, 'SVR' : 0.12})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.24, 'SVR' : 0.10999999999999999})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.4, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.24, 'SVR' : 0.16000000000000003})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.68, 'SIG' : 0.24, 'SVR' : 0.07999999999999996})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.34, 'SVR' : 0.36})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.35, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.25, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.27, 'SVR' : 0.22999999999999998})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.38, 'SVR' : 0.21999999999999997})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.28, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.5, 'SVR' : 0.15000000000000002})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.25, 'SVR' : 0.15000000000000002})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.35, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.62, 'SIG' : 0.22, 'SVR' : 0.16000000000000003})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.22, 'SVR' : 0.08000000000000007})
	else:
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.28, 'SVR' : 0.47})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.38, 'SVR' : 0.32000000000000006})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.3, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.25, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.35, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.72, 'SIG' : 0.24, 'SVR' : 0.040000000000000036})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.57, 'SVR' : 0.18000000000000005})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.57, 'SIG' : 0.28, 'SVR' : 0.15000000000000002})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.35, 'SVR' : 0.4})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.48, 'SIG' : 0.26, 'SVR' : 0.26})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.26, 'SVR' : 0.14})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.19, 'SIG' : 0.18, 'SVR' : 0.63})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.4, 'SVR' : 0.35})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.3, 'SVR' : 0.3500000000000001})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.3, 'SVR' : 0.3500000000000001})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.35, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.3, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.22, 'SIG' : 0.58, 'SVR' : 0.20000000000000007})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.35, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.34, 'SVR' : 0.4099999999999999})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.48, 'SIG' : 0.26, 'SVR' : 0.26})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.58, 'SIG' : 0.25, 'SVR' : 0.17000000000000004})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.15, 'SIG' : 0.16, 'SVR' : 0.69})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.3, 'SVR' : 0.44999999999999996})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.3, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.3, 'SVR' : 0.4})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.4, 'SVR' : 0.35})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.34, 'SVR' : 0.06000000000000005})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.18, 'SIG' : 0.62, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.47, 'SIG' : 0.3, 'SVR' : 0.22999999999999998})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.3, 'SVR' : 0.44999999999999996})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.22, 'SVR' : 0.28})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.27, 'SVR' : 0.22999999999999998})
else:
	if(InsSclInScen == 'LessUnstable'):
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.45, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.45, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.27, 'SVR' : 0.13})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.22, 'SVR' : 0.18000000000000005})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.32, 'SVR' : 0.1299999999999999})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.69, 'SIG' : 0.29, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.54, 'SIG' : 0.36, 'SVR' : 0.09999999999999998})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.75, 'SIG' : 0.2, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.3, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.22, 'SVR' : 0.08000000000000007})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.25, 'SVR' : 0.050000000000000044})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.4, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.4, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.3, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.27, 'SVR' : 0.17999999999999994})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.35, 'SVR' : 0.15000000000000002})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.33, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.38, 'SIG' : 0.5, 'SVR' : 0.12})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.24, 'SVR' : 0.06000000000000005})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.2, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.67, 'SIG' : 0.23, 'SVR' : 0.09999999999999998})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.7, 'SIG' : 0.25, 'SVR' : 0.050000000000000044})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.2, 'SIG' : 0.45, 'SVR' : 0.35})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.45, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.3, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.3, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.38, 'SVR' : 0.16999999999999993})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.6, 'SIG' : 0.38, 'SVR' : 0.020000000000000018})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.28, 'SIG' : 0.57, 'SVR' : 0.15000000000000002})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.28, 'SVR' : 0.06999999999999995})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.63, 'SIG' : 0.25, 'SVR' : 0.12})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.62, 'SIG' : 0.28, 'SVR' : 0.09999999999999998})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.8, 'SIG' : 0.17, 'SVR' : 0.029999999999999916})
	elif(InsSclInScen == 'Average'):
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.16, 'SIG' : 0.47, 'SVR' : 0.37})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.45, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.32, 'SVR' : 0.22999999999999998})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.52, 'SIG' : 0.26, 'SVR' : 0.21999999999999997})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.45, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.32, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.48, 'SIG' : 0.39, 'SVR' : 0.13})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.58, 'SIG' : 0.3, 'SVR' : 0.1200000000000001})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.45, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.28, 'SVR' : 0.21999999999999997})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.65, 'SIG' : 0.27, 'SVR' : 0.07999999999999996})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.18, 'SIG' : 0.45, 'SVR' : 0.37})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.35, 'SVR' : 0.3500000000000001})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.3, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.45, 'SIG' : 0.3, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.43, 'SVR' : 0.21999999999999997})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.62, 'SIG' : 0.35, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.2, 'SIG' : 0.65, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.52, 'SIG' : 0.33, 'SVR' : 0.1499999999999999})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.23, 'SIG' : 0.42, 'SVR' : 0.35})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.47, 'SIG' : 0.3, 'SVR' : 0.22999999999999998})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.3, 'SVR' : 0.1499999999999999})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.23, 'SIG' : 0.4, 'SVR' : 0.37})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.4, 'SVR' : 0.35})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.3, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.3, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.45, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.57, 'SIG' : 0.4, 'SVR' : 0.030000000000000027})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.15, 'SIG' : 0.65, 'SVR' : 0.19999999999999996})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.33, 'SVR' : 0.16999999999999993})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.36, 'SVR' : 0.39})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.28, 'SVR' : 0.21999999999999997})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.55, 'SIG' : 0.3, 'SVR' : 0.1499999999999999})
	else:
		if(CapInScen == 'LessThanAve'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.18, 'SIG' : 0.3, 'SVR' : 0.52})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.2, 'SIG' : 0.4, 'SVR' : 0.3999999999999999})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.3, 'SVR' : 0.4})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.3, 'SVR' : 0.30000000000000004})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.48, 'SVR' : 0.27})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.63, 'SIG' : 0.32, 'SVR' : 0.050000000000000044})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.15, 'SIG' : 0.63, 'SVR' : 0.21999999999999997})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.4, 'SIG' : 0.38, 'SVR' : 0.21999999999999997})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.2, 'SIG' : 0.37, 'SVR' : 0.42999999999999994})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.35, 'SVR' : 0.3500000000000001})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.32, 'SVR' : 0.17999999999999994})
		elif(CapInScen == 'Average'):
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.15, 'SIG' : 0.2, 'SVR' : 0.65})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.18, 'SIG' : 0.4, 'SVR' : 0.41999999999999993})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.35, 'SVR' : 0.4})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.35, 'SVR' : 0.4})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.25, 'SIG' : 0.42, 'SVR' : 0.33000000000000007})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.58, 'SIG' : 0.36, 'SVR' : 0.06000000000000005})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.13, 'SIG' : 0.62, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.45, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.22, 'SIG' : 0.35, 'SVR' : 0.43000000000000005})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.35, 'SIG' : 0.32, 'SVR' : 0.33000000000000007})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.3, 'SVR' : 0.19999999999999996})
		else:
			if(ScnRelPlFcst == 'A'):
				PlainsFcst ~= choice({'XNIL' : 0.1, 'SIG' : 0.2, 'SVR' : 0.7})
			elif(ScnRelPlFcst == 'B'):
				PlainsFcst ~= choice({'XNIL' : 0.2, 'SIG' : 0.3, 'SVR' : 0.5})
			elif(ScnRelPlFcst == 'C'):
				PlainsFcst ~= choice({'XNIL' : 0.2, 'SIG' : 0.4, 'SVR' : 0.3999999999999999})
			elif(ScnRelPlFcst == 'D'):
				PlainsFcst ~= choice({'XNIL' : 0.23, 'SIG' : 0.3, 'SVR' : 0.47})
			elif(ScnRelPlFcst == 'E'):
				PlainsFcst ~= choice({'XNIL' : 0.15, 'SIG' : 0.45, 'SVR' : 0.4})
			elif(ScnRelPlFcst == 'F'):
				PlainsFcst ~= choice({'XNIL' : 0.5, 'SIG' : 0.42, 'SVR' : 0.08000000000000007})
			elif(ScnRelPlFcst == 'G'):
				PlainsFcst ~= choice({'XNIL' : 0.1, 'SIG' : 0.65, 'SVR' : 0.25})
			elif(ScnRelPlFcst == 'H'):
				PlainsFcst ~= choice({'XNIL' : 0.28, 'SIG' : 0.4, 'SVR' : 0.31999999999999995})
			elif(ScnRelPlFcst == 'I'):
				PlainsFcst ~= choice({'XNIL' : 0.2, 'SIG' : 0.32, 'SVR' : 0.48})
			elif(ScnRelPlFcst == 'J'):
				PlainsFcst ~= choice({'XNIL' : 0.3, 'SIG' : 0.28, 'SVR' : 0.41999999999999993})
			else:
				PlainsFcst ~= choice({'XNIL' : 0.38, 'SIG' : 0.32, 'SVR' : 0.30000000000000004})


if (ScenRel3_4 == 'ACEFK'):
	if (PlainsFcst == 'XNIL'):
		N34StarFcst ~= choice({'XNIL' : 0.94, 'SIG' : 0.05, 'SVR' : 0.010000000000000009})
	elif(PlainsFcst == 'SIG'):
		N34StarFcst ~= choice({'XNIL' : 0.06, 'SIG' : 0.89, 'SVR' : 0.050000000000000044})
	else:
		N34StarFcst ~= choice({'XNIL' : 0.01, 'SIG' : 0.05, 'SVR' : 0.94})
elif(ScenRel3_4 == 'B'):
	if(PlainsFcst == 'XNIL'):
		N34StarFcst ~= choice({'XNIL' : 0.98, 'SIG' : 0.02, 'SVR' : 0.0})
	elif(PlainsFcst == 'SIG'):
		N34StarFcst ~= choice({'XNIL' : 0.04, 'SIG' : 0.94, 'SVR' : 0.020000000000000018})
	else:
		N34StarFcst ~= choice({'XNIL' : 0.0, 'SIG' : 0.03, 'SVR' : 0.97})
elif(ScenRel3_4 == 'D'):
	if(PlainsFcst == 'XNIL'):
		N34StarFcst ~= choice({'XNIL' : 0.92, 'SIG' : 0.06, 'SVR' : 0.020000000000000018})
	elif(PlainsFcst == 'SIG'):
		N34StarFcst ~= choice({'XNIL' : 0.01, 'SIG' : 0.89, 'SVR' : 0.09999999999999998})
	else:
		N34StarFcst ~= choice({'XNIL' : 0.0, 'SIG' : 0.01, 'SVR' : 0.99})
elif(ScenRel3_4 == 'GJ'):
	if(PlainsFcst == 'XNIL'):
		N34StarFcst ~= choice({'XNIL' : 0.92, 'SIG' : 0.06, 'SVR' : 0.020000000000000018})
	elif(PlainsFcst == 'SIG'):
		N34StarFcst ~= choice({'XNIL' : 0.03, 'SIG' : 0.92, 'SVR' : 0.04999999999999993})
	else:
		N34StarFcst ~= choice({'XNIL' : 0.01, 'SIG' : 0.04, 'SVR' : 0.95})
else:
	if(PlainsFcst == 'XNIL'):
		N34StarFcst ~= choice({'XNIL' : 0.99, 'SIG' : 0.01, 'SVR' : 0.0})
	elif(PlainsFcst == 'SIG'):
		N34StarFcst ~= choice({'XNIL' : 0.09, 'SIG' : 0.9, 'SVR' : 0.010000000000000009})
	else:
		N34StarFcst ~= choice({'XNIL' : 0.03, 'SIG' : 0.12, 'SVR' : 0.85})


if (MountainFcst == 'XNIL'):
	if (N34StarFcst == 'XNIL'):
		R5Fcst ~= choice({'XNIL' : 1.0, 'SIG' : 0.0, 'SVR' : 0.0})
	elif(N34StarFcst == 'SIG'):
		R5Fcst ~= choice({'XNIL' : 0.0, 'SIG' : 1.0, 'SVR' : 0.0})
	else:
		R5Fcst ~= choice({'XNIL' : 0.0, 'SIG' : 0.0, 'SVR' : 1.0})
elif(MountainFcst == 'SIG'):
	if(N34StarFcst == 'XNIL'):
		R5Fcst ~= choice({'XNIL' : 0.0, 'SIG' : 1.0, 'SVR' : 0.0})
	elif(N34StarFcst == 'SIG'):
		R5Fcst ~= choice({'XNIL' : 0.0, 'SIG' : 1.0, 'SVR' : 0.0})
	else:
		R5Fcst ~= choice({'XNIL' : 0.0, 'SIG' : 0.0, 'SVR' : 1.0})
else:
	if(N34StarFcst == 'XNIL'):
		R5Fcst ~= choice({'XNIL' : 0.0, 'SIG' : 0.0, 'SVR' : 1.0})
	elif(N34StarFcst == 'SIG'):
		R5Fcst ~= choice({'XNIL' : 0.0, 'SIG' : 0.0, 'SVR' : 1.0})
	else:
		R5Fcst ~= choice({'XNIL' : 0.0, 'SIG' : 0.0, 'SVR' : 1.0})


'''
compiler = SPPL_Compiler(data)
namespace = compiler.execute_module()
model=namespace.model
AMCINInScen = Id('AMCINInScen')
AMDewptCalPl = Id('AMDewptCalPl')
AMInsWliScen = Id('AMInsWliScen')
AMInstabMt = Id('AMInstabMt')
AreaMeso_ALS = Id('AreaMeso_ALS')
AreaMoDryAir = Id('AreaMoDryAir')
Boundaries = Id('Boundaries')
CapChange = Id('CapChange')
CapInScen = Id('CapInScen')
CldShadeConv = Id('CldShadeConv')
CldShadeOth = Id('CldShadeOth')
CombClouds = Id('CombClouds')
CombMoisture = Id('CombMoisture')
CombVerMo = Id('CombVerMo')
CompPlFcst = Id('CompPlFcst')
CurPropConv = Id('CurPropConv')
Date = Id('Date')
Dewpoints = Id('Dewpoints')
IRCloudCover = Id('IRCloudCover')
InsChange = Id('InsChange')
InsInMt = Id('InsInMt')
InsSclInScen = Id('InsSclInScen')
LIfr12ZDENSd = Id('LIfr12ZDENSd')
LLIW = Id('LLIW')
LatestCIN = Id('LatestCIN')
LoLevMoistAd = Id('LoLevMoistAd')
LowLLapse = Id('LowLLapse')
MeanRH = Id('MeanRH')
MidLLapse = Id('MidLLapse')
MorningBound = Id('MorningBound')
MorningCIN = Id('MorningCIN')
MountainFcst = Id('MountainFcst')
MvmtFeatures = Id('MvmtFeatures')
N0_7muVerMo = Id('N0_7muVerMo')
N34StarFcst = Id('N34StarFcst')
OutflowFrMt = Id('OutflowFrMt')
PlainsFcst = Id('PlainsFcst')
QGVertMotion = Id('QGVertMotion')
R5Fcst = Id('R5Fcst')
RHRatio = Id('RHRatio')
RaoContMoist = Id('RaoContMoist')
SatContMoist = Id('SatContMoist')
ScenRel3_4 = Id('ScenRel3_4')
ScenRelAMCIN = Id('ScenRelAMCIN')
ScenRelAMIns = Id('ScenRelAMIns')
Scenario = Id('Scenario')
ScnRelPlFcst = Id('ScnRelPlFcst')
SfcWndShfDis = Id('SfcWndShfDis')
SubjVertMo = Id('SubjVertMo')
SynForcng = Id('SynForcng')
TempDis = Id('TempDis')
VISCloudCov = Id('VISCloudCov')
WindAloft = Id('WindAloft')
WindFieldMt = Id('WindFieldMt')
WindFieldPln = Id('WindFieldPln')
WndHodograph = Id('WndHodograph')
events = [InsSclInScen << {'LessUnstable'},LoLevMoistAd << {'Neutral'},WindAloft << {'SWQuad'},ScenRelAMCIN << {'CThruK'},AreaMoDryAir << {'VeryWet'},SfcWndShfDis << {'DenvCyclone'},WndHodograph << {'DCVZFavor'},Date << {'Jul2_Jul15'},LIfr12ZDENSd << {'LILt_8'},MvmtFeatures << {'MarkedUpper'},ScenRel3_4 << {'B'},LowLLapse << {'Steep'},CombClouds << {'Cloudy'},LIfr12ZDENSd << {'N5GtLIGt_8'},WindFieldPln << {'DenvCyclone'},CombMoisture << {'VeryWet'},LIfr12ZDENSd << {'LILt_8'},ScnRelPlFcst << {'K'},CompPlFcst << {'DecCapIncIns'},MeanRH << {'VeryMoist'},QGVertMotion << {'Down'},AreaMoDryAir << {'Neutral'},LLIW << {'Weak'},N34StarFcst << {'SIG'},ScenRelAMCIN << {'AB'},RaoContMoist << {'Wet'},QGVertMotion << {'WeakUp'},MorningBound << {'None'},RHRatio << {'Other'},SynForcng << {'SigNegative'},SatContMoist << {'Neutral'},AMInsWliScen << {'Average'},LatestCIN << {'None'},N34StarFcst << {'XNIL'},InsChange << {'LittleChange'},ScenRelAMIns << {'G'},LoLevMoistAd << {'Neutral'},ScenRelAMCIN << {'CThruK'},CldShadeOth << {'Cloudy'},InsInMt << {'Weak'},OutflowFrMt << {'Strong'},CombVerMo << {'Down'},OutflowFrMt << {'Weak'},LowLLapse << {'ModerateOrLe'},AMInstabMt << {'Weak'},Dewpoints << {'LowSHighN'},R5Fcst << {'XNIL'},MidLLapse << {'CloseToDryAd'},SfcWndShfDis << {'DenvCyclone'},MvmtFeatures << {'OtherRapid'},(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Strong'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Increasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Strong'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Strong'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Neutral'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'H'}) & (Scenario << {'G'}) & (ScnRelPlFcst << {'E'}) & (SfcWndShfDis << {'Other'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'LittleChange'}) & (TempDis << {'None'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'LongAnticyc'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'Neutral'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'None'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'LowSHighN'}) & (IRCloudCover << {'PC'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'Average'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Strong'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'StrongUp'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'ACEFK'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'G'}) & (Scenario << {'H'}) & (ScnRelPlFcst << {'E'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'SigPositive'}) & (TempDis << {'QStationary'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'Westerly'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'Weak'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'None'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'LowNHighS'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'Stifling'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'G'}) & (Scenario << {'H'}) & (ScnRelPlFcst << {'G'}) & (SfcWndShfDis << {'None'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'PosToNeg'}) & (TempDis << {'None'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'LV'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Strong'}) & (Date << {'Jul16_Aug10'}) & (Dewpoints << {'LowMtsHighPl'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Moderate'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'None'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'ACEFK'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'K'}) & (Scenario << {'F'}) & (ScnRelPlFcst << {'A'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'Moving'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'LongAnticyc'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'None'}) & (CapChange << {'Increasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Moderate'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'LowNHighS'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Strong'}) & (LatestCIN << {'Stifling'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'K'}) & (ScnRelPlFcst << {'B'}) & (SfcWndShfDis << {'None'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'LongAnticyc'}) & (WndHodograph << {'Other'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Slight'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'N1GtLIGt_4'}) & (LLIW << {'Weak'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Average'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'None'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'StrongUp'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'B'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'I'}) & (ScnRelPlFcst << {'C'}) & (SfcWndShfDis << {'E_W_S'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'PosToNeg'}) & (TempDis << {'None'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'E_NE'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Slight'}) & (Date << {'Aug11_Aug20'}) & (Dewpoints << {'HighEvrywher'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Strong'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'Average'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'None'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'B'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'F'}) & (Scenario << {'B'}) & (ScnRelPlFcst << {'G'}) & (SfcWndShfDis << {'None'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'None'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'E_NE'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'Weak'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Strong'}) & (Date << {'Aug11_Aug20'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Moderate'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Strong'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'G'}) & (Scenario << {'I'}) & (ScnRelPlFcst << {'I'}) & (SfcWndShfDis << {'DenvCyclone'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'None'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'E_NE'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Weak'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'PC'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Moderate'}) & (Date << {'Aug11_Aug20'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Weak'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Strong'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'H'}) & (Scenario << {'A'}) & (ScnRelPlFcst << {'I'}) & (SfcWndShfDis << {'Other'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'SigPositive'}) & (TempDis << {'QStationary'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'WidespdDnsl'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'Weak'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'None'}) & (Date << {'Jul2_Jul15'}) & (Dewpoints << {'LowAtStation'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'Average'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'Neutral'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Wet'}) & (ScenRel3_4 << {'ACEFK'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'C'}) & (ScnRelPlFcst << {'J'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'PosToNeg'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Weak'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'Neutral'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Moderate'}) & (Date << {'Aug11_Aug20'}) & (Dewpoints << {'LowMtsHighPl'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'None'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'Stifling'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'None'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'B'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'F'}) & (Scenario << {'G'}) & (ScnRelPlFcst << {'B'}) & (SfcWndShfDis << {'None'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'LittleChange'}) & (TempDis << {'None'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'WidespdDnsl'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Weak'}) & (CapChange << {'Increasing'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Moderate'}) & (Date << {'Aug11_Aug20'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Increasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Strong'}) & (LatestCIN << {'Stifling'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'None'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'StrongUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Neutral'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Wet'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'G'}) & (Scenario << {'E'}) & (ScnRelPlFcst << {'B'}) & (SfcWndShfDis << {'MovingFtorOt'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'Moving'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'LongAnticyc'}) & (WndHodograph << {'Westerly'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Moderate'}) & (Date << {'Jul16_Aug10'}) & (Dewpoints << {'HighEvrywher'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'None'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Wet'}) & (ScenRel3_4 << {'B'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'ABI'}) & (Scenario << {'C'}) & (ScnRelPlFcst << {'H'}) & (SfcWndShfDis << {'E_W_S'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'LittleChange'}) & (TempDis << {'Other'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Weak'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Moderate'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'LowMtsHighPl'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Moderate'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'F'}) & (Scenario << {'D'}) & (ScnRelPlFcst << {'B'}) & (SfcWndShfDis << {'Other'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'SigPositive'}) & (TempDis << {'Other'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'StrongUp'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'None'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Moderate'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'LowMtsHighPl'}) & (IRCloudCover << {'PC'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'None'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Strong'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'ABI'}) & (Scenario << {'B'}) & (ScnRelPlFcst << {'H'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'None'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'WidespdDnsl'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'None'}) & (CapChange << {'Increasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Slight'}) & (Date << {'May15_Jun14'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N1GtLIGt_4'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'G'}) & (Scenario << {'F'}) & (ScnRelPlFcst << {'C'}) & (SfcWndShfDis << {'MovingFtorOt'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'PosToNeg'}) & (TempDis << {'Moving'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'WidespdDnsl'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Slight'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N1GtLIGt_4'}) & (LLIW << {'Moderate'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'None'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'K'}) & (Scenario << {'C'}) & (ScnRelPlFcst << {'H'}) & (SfcWndShfDis << {'DryLine'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'SigPositive'}) & (TempDis << {'Moving'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'LV'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'Weak'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Slight'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'LowSHighN'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Increasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'Average'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'None'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'B'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'F'}) & (Scenario << {'D'}) & (ScnRelPlFcst << {'C'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'LittleChange'}) & (TempDis << {'None'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'Strong'}) & (CapChange << {'Increasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'None'}) & (Date << {'Jul2_Jul15'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Moderate'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'Strong'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Neutral'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'Wet'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'F'}) & (Scenario << {'I'}) & (ScnRelPlFcst << {'J'}) & (SfcWndShfDis << {'None'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'QStationary'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'LV'}) & (WndHodograph << {'Westerly'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'None'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'None'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'HighEvrywher'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'None'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Average'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'ABI'}) & (Scenario << {'C'}) & (ScnRelPlFcst << {'I'}) & (SfcWndShfDis << {'None'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'Other'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'Strong'}) & (CapChange << {'Increasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Slight'}) & (Date << {'Aug11_Aug20'}) & (Dewpoints << {'LowMtsHighPl'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Strong'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'None'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'HI'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'G'}) & (Scenario << {'I'}) & (ScnRelPlFcst << {'E'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'Other'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'LongAnticyc'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'StrongUp'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'Weak'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Slight'}) & (Date << {'May15_Jun14'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'None'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'None'}) & (MorningCIN << {'None'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'StrongUp'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'C'}) & (ScnRelPlFcst << {'E'}) & (SfcWndShfDis << {'Other'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'None'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'Westerly'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'Strong'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'None'}) & (Date << {'Jul16_Aug10'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Moderate'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'None'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'StrongUp'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'Neutral'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'H'}) & (Scenario << {'C'}) & (ScnRelPlFcst << {'G'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'None'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'Strong'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'None'}) & (Date << {'May15_Jun14'}) & (Dewpoints << {'LowSHighN'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'Average'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'ABI'}) & (Scenario << {'K'}) & (ScnRelPlFcst << {'C'}) & (SfcWndShfDis << {'E_W_S'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'QStationary'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'LV'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'None'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'None'}) & (Date << {'Aug11_Aug20'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'Average'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'I'}) & (ScnRelPlFcst << {'A'}) & (SfcWndShfDis << {'DenvCyclone'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'Moving'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'LV'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'None'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'Neutral'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Strong'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'HighEvrywher'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'Stifling'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'StrongUp'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'D'}) & (ScnRelPlFcst << {'J'}) & (SfcWndShfDis << {'DryLine'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'Moving'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'Other'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'None'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'Neutral'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Strong'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'LowMtsHighPl'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Increasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N1GtLIGt_4'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'Wet'}) & (ScenRel3_4 << {'HI'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'K'}) & (ScnRelPlFcst << {'I'}) & (SfcWndShfDis << {'DenvCyclone'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'None'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'Other'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'StrongUp'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'Weak'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Slight'}) & (Date << {'Jul2_Jul15'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'PC'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'N1GtLIGt_4'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Strong'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'F'}) & (Scenario << {'A'}) & (ScnRelPlFcst << {'C'}) & (SfcWndShfDis << {'DenvCyclone'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'Other'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'LV'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'StrongUp'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Weak'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Strong'}) & (Date << {'Jul16_Aug10'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Strong'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'None'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'ABI'}) & (Scenario << {'I'}) & (ScnRelPlFcst << {'F'}) & (SfcWndShfDis << {'E_W_S'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'QStationary'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'WidespdDnsl'}) & (WndHodograph << {'Other'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Moderate'}) & (Date << {'Jul16_Aug10'}) & (Dewpoints << {'LowSHighN'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'N1GtLIGt_4'}) & (LLIW << {'Strong'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Strong'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'ACEFK'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'F'}) & (Scenario << {'F'}) & (ScnRelPlFcst << {'D'}) & (SfcWndShfDis << {'MovingFtorOt'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'SigPositive'}) & (TempDis << {'Moving'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'Westerly'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'Weak'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Slight'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Moderate'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Wet'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'C'}) & (ScnRelPlFcst << {'F'}) & (SfcWndShfDis << {'DenvCyclone'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'Other'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Strong'}) & (CapChange << {'Increasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Moderate'}) & (Date << {'May15_Jun14'}) & (Dewpoints << {'LowNHighS'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N1GtLIGt_4'}) & (LLIW << {'Strong'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'None'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'G'}) & (Scenario << {'C'}) & (ScnRelPlFcst << {'K'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'LittleChange'}) & (TempDis << {'None'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'Westerly'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'Strong'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Moderate'}) & (Date << {'May15_Jun14'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'K'}) & (Scenario << {'J'}) & (ScnRelPlFcst << {'B'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'Other'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'E_NE'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'StrongUp'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'Weak'}) & (CapChange << {'Increasing'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Strong'}) & (Date << {'Jul16_Aug10'}) & (Dewpoints << {'LowMtsHighPl'}) & (IRCloudCover << {'PC'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N1GtLIGt_4'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'Stifling'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'Average'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'None'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'H'}) & (Scenario << {'D'}) & (ScnRelPlFcst << {'F'}) & (SfcWndShfDis << {'E_W_N'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'PosToNeg'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'LongAnticyc'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'None'}) & (CapChange << {'Increasing'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Slight'}) & (Date << {'Jul2_Jul15'}) & (Dewpoints << {'LowNHighS'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'StrongUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'H'}) & (ScnRelPlFcst << {'A'}) & (SfcWndShfDis << {'None'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'SigPositive'}) & (TempDis << {'Other'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'None'}) & (Date << {'Jul16_Aug10'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Strong'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'Average'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'B'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'E'}) & (ScnRelPlFcst << {'C'}) & (SfcWndShfDis << {'DryLine'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'LittleChange'}) & (TempDis << {'None'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'LongAnticyc'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'Strong'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'None'}) & (Date << {'Aug11_Aug20'}) & (Dewpoints << {'LowSHighN'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'WeakPos'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'H'}) & (Scenario << {'G'}) & (ScnRelPlFcst << {'F'}) & (SfcWndShfDis << {'DryLine'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'LittleChange'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'E_NE'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Strong'}) & (CapChange << {'Increasing'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'IncCapDecIns'}) & (CurPropConv << {'Slight'}) & (Date << {'Aug11_Aug20'}) & (Dewpoints << {'LowNHighS'}) & (IRCloudCover << {'PC'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'Average'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Strong'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'None'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Wet'}) & (ScenRel3_4 << {'B'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'G'}) & (Scenario << {'D'}) & (ScnRelPlFcst << {'I'}) & (SfcWndShfDis << {'DenvCyclone'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Strong'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'Neutral'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Slight'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Increasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'N1GtLIGt_4'}) & (LLIW << {'Moderate'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'None'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Strong'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'Neutral'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'ACEFK'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'K'}) & (Scenario << {'D'}) & (ScnRelPlFcst << {'D'}) & (SfcWndShfDis << {'MovingFtorOt'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'Moving'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'LV'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'StrongUp'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'None'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'PC'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'Neutral'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Slight'}) & (Date << {'Jul16_Aug10'}) & (Dewpoints << {'LowAtStation'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Strong'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'None'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'Neutral'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'HI'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'H'}) & (Scenario << {'K'}) & (ScnRelPlFcst << {'B'}) & (SfcWndShfDis << {'None'}) & (SubjVertMo << {'Neutral'}) & (SynForcng << {'LittleChange'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'LV'}) & (WndHodograph << {'Other'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'None'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'Down'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Strong'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'None'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'Average'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'None'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'ACEFK'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'F'}) & (Scenario << {'A'}) & (ScnRelPlFcst << {'J'}) & (SfcWndShfDis << {'MovingFtorOt'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'PosToNeg'}) & (TempDis << {'None'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'LongAnticyc'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'StrongUp'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'Weak'}) & (CapChange << {'Increasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Slight'}) & (Date << {'Jul16_Aug10'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'N5GtLIGt_8'}) & (LLIW << {'Strong'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'None'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'HI'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'G'}) & (Scenario << {'F'}) & (ScnRelPlFcst << {'B'}) & (SfcWndShfDis << {'DenvCyclone'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'SigPositive'}) & (TempDis << {'None'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'Other'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'VeryWet'}) & (Boundaries << {'Strong'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Strong'}) & (Date << {'May15_Jun14'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Weak'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'None'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'OtherRapid'}) & (N0_7muVerMo << {'StrongUp'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'H'}) & (Scenario << {'K'}) & (ScnRelPlFcst << {'A'}) & (SfcWndShfDis << {'None'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'SigPositive'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'Other'}),(AMCINInScen << {'MoreThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'Weak'}) & (AreaMeso_ALS << {'Neutral'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'Strong'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Strong'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'LowEvrywhere'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'LittleChange'}) & (InsInMt << {'Strong'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'LIGt0'}) & (LLIW << {'Strong'}) & (LatestCIN << {'PartInhibit'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'Average'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'XNIL'}) & (QGVertMotion << {'Neutral'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'VeryWet'}) & (SatContMoist << {'Dry'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'K'}) & (Scenario << {'E'}) & (ScnRelPlFcst << {'J'}) & (SfcWndShfDis << {'DryLine'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'PosToNeg'}) & (TempDis << {'Moving'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'Westerly'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'StrongUp'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'None'}) & (CapChange << {'Increasing'}) & (CapInScen << {'MoreThanAve'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'Moderate'}) & (Date << {'May15_Jun14'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'ModerateOrLe'}) & (MeanRH << {'Average'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'SIG'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'None'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'MoistMDryL'}) & (RaoContMoist << {'Wet'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'D'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'H'}) & (Scenario << {'F'}) & (ScnRelPlFcst << {'A'}) & (SfcWndShfDis << {'MovingFtorOt'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'PosToNeg'}) & (TempDis << {'QStationary'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'AllElse'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'LongAnticyc'}) & (WndHodograph << {'DCVZFavor'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Instability'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Neutral'}) & (Boundaries << {'Strong'}) & (CapChange << {'Increasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'PC'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'WeakUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Moderate'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Increasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'Stifling'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'Average'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'B'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'H'}) & (ScnRelPlFcst << {'C'}) & (SfcWndShfDis << {'MovingFtorOt'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'None'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'NWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'Other'}),(AMCINInScen << {'Average'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'StrongUp'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Weak'}) & (CapChange << {'Increasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Clear'}) & (CombClouds << {'PC'}) & (CombMoisture << {'Dry'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'LittleChange'}) & (CurPropConv << {'None'}) & (Date << {'Jun15_Jul1'}) & (Dewpoints << {'LowMtsHighPl'}) & (IRCloudCover << {'Clear'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'None'}) & (LoLevMoistAd << {'StrongPos'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'Average'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'Stifling'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'StrongFront'}) & (N0_7muVerMo << {'WeakUp'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'WeakUp'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'CDEJ'}) & (Scenario << {'F'}) & (ScnRelPlFcst << {'G'}) & (SfcWndShfDis << {'E_W_S'}) & (SubjVertMo << {'Down'}) & (SynForcng << {'SigNegative'}) & (TempDis << {'QStationary'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'E_NE'}) & (WndHodograph << {'Westerly'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'MoreUnstable'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Wet'}) & (Boundaries << {'Strong'}) & (CapChange << {'Increasing'}) & (CapInScen << {'Average'}) & (CldShadeConv << {'None'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Cloudy'}) & (CombMoisture << {'VeryWet'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Strong'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'Other'}) & (IRCloudCover << {'PC'}) & (InsChange << {'Increasing'}) & (InsInMt << {'None'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Unfavorable'}) & (LatestCIN << {'TotalInhibit'}) & (LoLevMoistAd << {'Neutral'}) & (LowLLapse << {'Stable'}) & (MeanRH << {'Dry'}) & (MidLLapse << {'CloseToDryAd'}) & (MorningBound << {'Weak'}) & (MorningCIN << {'None'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'NoMajor'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'SVR'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SVR'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'Neutral'}) & (ScenRel3_4 << {'B'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'F'}) & (Scenario << {'G'}) & (ScnRelPlFcst << {'F'}) & (SfcWndShfDis << {'DenvCyclone'}) & (SubjVertMo << {'WeakUp'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'QStationary'}) & (VISCloudCov << {'PC'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'WidespdDnsl'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Neutral'}) & (AMInsWliScen << {'Average'}) & (AMInstabMt << {'Strong'}) & (AreaMeso_ALS << {'Down'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Weak'}) & (CapChange << {'LittleChange'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Some'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Wet'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Slight'}) & (Date << {'May15_Jun14'}) & (Dewpoints << {'HighEvrywher'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'LessUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Strong'}) & (LatestCIN << {'Stifling'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'Steep'}) & (MeanRH << {'Average'}) & (MidLLapse << {'Steep'}) & (MorningBound << {'None'}) & (MorningCIN << {'TotalInhibit'}) & (MountainFcst << {'SVR'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'Neutral'}) & (N34StarFcst << {'XNIL'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SVR'}) & (QGVertMotion << {'StrongUp'}) & (R5Fcst << {'XNIL'}) & (RHRatio << {'DryMMoistL'}) & (RaoContMoist << {'Neutral'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'GJ'}) & (ScenRelAMCIN << {'CThruK'}) & (ScenRelAMIns << {'K'}) & (Scenario << {'G'}) & (ScnRelPlFcst << {'J'}) & (SfcWndShfDis << {'MovingFtorOt'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'NegToPos'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Cloudy'}) & (WindAloft << {'SWQuad'}) & (WindFieldMt << {'Westerly'}) & (WindFieldPln << {'SEQuad'}) & (WndHodograph << {'StrongWest'}),(AMCINInScen << {'LessThanAve'}) & (AMDewptCalPl << {'Stability'}) & (AMInsWliScen << {'LessUnstable'}) & (AMInstabMt << {'None'}) & (AreaMeso_ALS << {'WeakUp'}) & (AreaMoDryAir << {'Dry'}) & (Boundaries << {'Strong'}) & (CapChange << {'Decreasing'}) & (CapInScen << {'LessThanAve'}) & (CldShadeConv << {'Marked'}) & (CldShadeOth << {'Cloudy'}) & (CombClouds << {'Clear'}) & (CombMoisture << {'Neutral'}) & (CombVerMo << {'StrongUp'}) & (CompPlFcst << {'DecCapIncIns'}) & (CurPropConv << {'Moderate'}) & (Date << {'Aug20_Sep15'}) & (Dewpoints << {'HighEvrywher'}) & (IRCloudCover << {'Cloudy'}) & (InsChange << {'Decreasing'}) & (InsInMt << {'Weak'}) & (InsSclInScen << {'MoreUnstable'}) & (LIfr12ZDENSd << {'LILt_8'}) & (LLIW << {'Weak'}) & (LatestCIN << {'Stifling'}) & (LoLevMoistAd << {'Negative'}) & (LowLLapse << {'CloseToDryAd'}) & (MeanRH << {'VeryMoist'}) & (MidLLapse << {'ModerateOrLe'}) & (MorningBound << {'None'}) & (MorningCIN << {'PartInhibit'}) & (MountainFcst << {'XNIL'}) & (MvmtFeatures << {'MarkedUpper'}) & (N0_7muVerMo << {'Down'}) & (N34StarFcst << {'SIG'}) & (OutflowFrMt << {'Weak'}) & (PlainsFcst << {'SIG'}) & (QGVertMotion << {'Down'}) & (R5Fcst << {'SIG'}) & (RHRatio << {'Other'}) & (RaoContMoist << {'Dry'}) & (SatContMoist << {'VeryWet'}) & (ScenRel3_4 << {'ACEFK'}) & (ScenRelAMCIN << {'AB'}) & (ScenRelAMIns << {'ABI'}) & (Scenario << {'E'}) & (ScnRelPlFcst << {'D'}) & (SfcWndShfDis << {'Other'}) & (SubjVertMo << {'StronUp'}) & (SynForcng << {'PosToNeg'}) & (TempDis << {'Other'}) & (VISCloudCov << {'Clear'}) & (WindAloft << {'LV'}) & (WindFieldMt << {'LVorOther'}) & (WindFieldPln << {'DenvCyclone'}) & (WndHodograph << {'StrongWest'})]
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
