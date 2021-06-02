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
age ~= choice({'age65_100' : 0.07725322,'age51_65' : 0.38769671,'age31_50' : 0.39771102,'age0_30' : 0.13733906})


alcoholism ~= choice({'present' : 0.1359084,'absent' : 0.8640916})


diabetes ~= choice({'present' : 0.03576538,'absent' : 0.96423462})


gallstones ~= choice({'present' : 0.1530758,'absent' : 0.8469242})


hepatotoxic ~= choice({'present' : 0.08154506,'absent' : 0.91845494})


hospital ~= choice({'present' : 0.5350501,'absent' : 0.4649499})


if (diabetes == 'present'):
	obesity ~= choice({'present' : 0.24, 'absent' : 0.76})
else:
	obesity ~= choice({'present' : 0.06231454, 'absent' : 0.93768546})


sex ~= choice({'female' : 0.5979971,'male' : 0.4020029})


surgery ~= choice({'present' : 0.4234621,'absent' : 0.5765379})


if (gallstones == 'present'):
	upper_pain ~= choice({'present' : 0.411215, 'absent' : 0.588785})
else:
	upper_pain ~= choice({'present' : 0.3868243, 'absent' : 0.6131757})


vh_amn ~= choice({'present' : 0.1731044,'absent' : 0.8268956})


if (age == 'age65_100'):
	if (sex == 'female'):
		Hyperbilirubinemia ~= choice({'present' : 0.002849, 'absent' : 0.997151})
	else:
		Hyperbilirubinemia ~= choice({'present' : 0.0052356, 'absent' : 0.9947644})
elif(age == 'age51_65'):
	if(sex == 'female'):
		Hyperbilirubinemia ~= choice({'present' : 0.01129944, 'absent' : 0.98870056})
	else:
		Hyperbilirubinemia ~= choice({'present' : 0.0212766, 'absent' : 0.9787234})
elif(age == 'age31_50'):
	if(sex == 'female'):
		Hyperbilirubinemia ~= choice({'present' : 0.04597701, 'absent' : 0.95402299})
	else:
		Hyperbilirubinemia ~= choice({'present' : 0.07692308, 'absent' : 0.92307692})
else:
	if(sex == 'female'):
		Hyperbilirubinemia ~= choice({'present' : 0.21875, 'absent' : 0.78125})
	else:
		Hyperbilirubinemia ~= choice({'present' : 0.453125, 'absent' : 0.546875})


if (sex == 'female'):
	if (age == 'age65_100'):
		PBC ~= choice({'present' : 0.6571429, 'absent' : 0.3428571})
	elif(age == 'age51_65'):
		PBC ~= choice({'present' : 0.700565, 'absent' : 0.299435})
	elif(age == 'age31_50'):
		PBC ~= choice({'present' : 0.6149425, 'absent' : 0.38505750000000005})
	else:
		PBC ~= choice({'present' : 0.125, 'absent' : 0.875})
else:
	if(age == 'age65_100'):
		PBC ~= choice({'present' : 0.3684211, 'absent' : 0.6315789})
	elif(age == 'age51_65'):
		PBC ~= choice({'present' : 0.08510638, 'absent' : 0.91489362})
	elif(age == 'age31_50'):
		PBC ~= choice({'present' : 0.06730769, 'absent' : 0.93269231})
	else:
		PBC ~= choice({'present' : 0.00156006, 'absent' : 0.99843994})


if (hepatotoxic == 'present'):
	RHepatitis ~= choice({'present' : 0.01754386, 'absent' : 0.98245614})
else:
	RHepatitis ~= choice({'present' : 0.02492212, 'absent' : 0.97507788})


if (obesity == 'present'):
	if (alcoholism == 'present'):
		Steatosis ~= choice({'present' : 0.3636364, 'absent' : 0.6363635999999999})
	else:
		Steatosis ~= choice({'present' : 0.1891892, 'absent' : 0.8108108})
else:
	if(alcoholism == 'present'):
		Steatosis ~= choice({'present' : 0.2380952, 'absent' : 0.7619047999999999})
	else:
		Steatosis ~= choice({'present' : 0.06349206, 'absent' : 0.93650794})


if (hepatotoxic == 'present'):
	if (alcoholism == 'present'):
		THepatitis ~= choice({'present' : 0.2, 'absent' : 0.8})
	else:
		THepatitis ~= choice({'present' : 0.00191939, 'absent' : 0.99808061})
else:
	if(alcoholism == 'present'):
		THepatitis ~= choice({'present' : 0.08888889, 'absent' : 0.91111111})
	else:
		THepatitis ~= choice({'present' : 0.0326087, 'absent' : 0.9673913})


if (PBC == 'present'):
	ama ~= choice({'present' : 0.5678571, 'absent' : 0.4321429})
else:
	ama ~= choice({'present' : 0.01193317, 'absent' : 0.98806683})


if (gallstones == 'present'):
	amylase ~= choice({'a1400_500' : 0.01869159, 'a499_300' : 0.04672897, 'a299_0' : 0.93457944})
else:
	amylase ~= choice({'a1400_500' : 0.01013514, 'a499_300' : 0.01689189, 'a299_0' : 0.97297297})


if (RHepatitis == 'present'):
	if (THepatitis == 'present'):
		anorexia ~= choice({'present' : 0.1818182, 'absent' : 0.8181818})
	else:
		anorexia ~= choice({'present' : 0.1176471, 'absent' : 0.8823529})
else:
	if(THepatitis == 'present'):
		anorexia ~= choice({'present' : 0.2222222, 'absent' : 0.7777778})
	else:
		anorexia ~= choice({'present' : 0.280916, 'absent' : 0.7190840000000001})


if (gallstones == 'present'):
	choledocholithotomy ~= choice({'present' : 0.7102804, 'absent' : 0.28971959999999997})
else:
	choledocholithotomy ~= choice({'present' : 0.03716216, 'absent' : 0.96283784})


if (gallstones == 'present'):
	fat ~= choice({'present' : 0.1775701, 'absent' : 0.8224298999999999})
else:
	fat ~= choice({'present' : 0.2804054, 'absent' : 0.7195946})


if (gallstones == 'present'):
	flatulence ~= choice({'present' : 0.3925234, 'absent' : 0.6074766})
else:
	flatulence ~= choice({'present' : 0.4307432, 'absent' : 0.5692568})


if (RHepatitis == 'present'):
	if (THepatitis == 'present'):
		if (Steatosis == 'present'):
			if (Hyperbilirubinemia == 'present'):
				hepatomegaly ~= choice({'present' : 0.6097561, 'absent' : 0.3902439})
			else:
				hepatomegaly ~= choice({'present' : 0.68, 'absent' : 0.31999999999999995})
		else:
			if(Hyperbilirubinemia == 'present'):
				hepatomegaly ~= choice({'present' : 0.5918367, 'absent' : 0.4081633})
			else:
				hepatomegaly ~= choice({'present' : 0.673913, 'absent' : 0.326087})
	else:
		if(Steatosis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				hepatomegaly ~= choice({'present' : 0.5901639, 'absent' : 0.40983610000000004})
			else:
				hepatomegaly ~= choice({'present' : 0.6527778, 'absent' : 0.34722220000000004})
		else:
			if(Hyperbilirubinemia == 'present'):
				hepatomegaly ~= choice({'present' : 0.5555556, 'absent' : 0.44444439999999996})
			else:
				hepatomegaly ~= choice({'present' : 0.7058823, 'absent' : 0.29411770000000004})
else:
	if(THepatitis == 'present'):
		if(Steatosis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				hepatomegaly ~= choice({'present' : 0.6, 'absent' : 0.4})
			else:
				hepatomegaly ~= choice({'present' : 0.6756757, 'absent' : 0.3243243})
		else:
			if(Hyperbilirubinemia == 'present'):
				hepatomegaly ~= choice({'present' : 0.5897436, 'absent' : 0.41025639999999997})
			else:
				hepatomegaly ~= choice({'present' : 0.7777778, 'absent' : 0.22222220000000004})
	else:
		if(Steatosis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				hepatomegaly ~= choice({'present' : 0.5866667, 'absent' : 0.4133333})
			else:
				hepatomegaly ~= choice({'present' : 0.6865672, 'absent' : 0.31343279999999996})
		else:
			if(Hyperbilirubinemia == 'present'):
				hepatomegaly ~= choice({'present' : 0.375, 'absent' : 0.625})
			else:
				hepatomegaly ~= choice({'present' : 0.6973684, 'absent' : 0.3026316})


if (hospital == 'present'):
	if (surgery == 'present'):
		if (choledocholithotomy == 'present'):
			injections ~= choice({'present' : 0.8, 'absent' : 0.19999999999999996})
		else:
			injections ~= choice({'present' : 0.715847, 'absent' : 0.284153})
	else:
		if(choledocholithotomy == 'present'):
			injections ~= choice({'present' : 0.8333333, 'absent' : 0.16666669999999995})
		else:
			injections ~= choice({'present' : 0.4818182, 'absent' : 0.5181818})
else:
	if(surgery == 'present'):
		if(choledocholithotomy == 'present'):
			injections ~= choice({'present' : 0.375, 'absent' : 0.625})
		else:
			injections ~= choice({'present' : 0.2333333, 'absent' : 0.7666667})
	else:
		if(choledocholithotomy == 'present'):
			injections ~= choice({'present' : 0.01098901, 'absent' : 0.98901099})
		else:
			injections ~= choice({'present' : 0.0647482, 'absent' : 0.9352518})


if (PBC == 'present'):
	joints ~= choice({'present' : 0.1285714, 'absent' : 0.8714286})
else:
	joints ~= choice({'present' : 0.1002387, 'absent' : 0.8997613})


if (PBC == 'present'):
	le_cells ~= choice({'present' : 0.1214286, 'absent' : 0.8785714})
else:
	le_cells ~= choice({'present' : 0.04057279, 'absent' : 0.95942721})


if (RHepatitis == 'present'):
	if (THepatitis == 'present'):
		nausea ~= choice({'present' : 0.3636364, 'absent' : 0.6363635999999999})
	else:
		nausea ~= choice({'present' : 0.3529412, 'absent' : 0.6470587999999999})
else:
	if(THepatitis == 'present'):
		nausea ~= choice({'present' : 0.3703704, 'absent' : 0.6296296})
	else:
		nausea ~= choice({'present' : 0.2854962, 'absent' : 0.7145038})


if (PBC == 'present'):
	if (joints == 'present'):
		pain ~= choice({'present' : 0.3888889, 'absent' : 0.6111111})
	else:
		pain ~= choice({'present' : 0.147541, 'absent' : 0.852459})
else:
	if(joints == 'present'):
		pain ~= choice({'present' : 0.8095238, 'absent' : 0.19047619999999998})
	else:
		pain ~= choice({'present' : 0.1830239, 'absent' : 0.8169761})


if (Steatosis == 'present'):
	if (Hyperbilirubinemia == 'present'):
		pain_ruq ~= choice({'present' : 0.3934426, 'absent' : 0.6065574})
	else:
		pain_ruq ~= choice({'present' : 0.4776119, 'absent' : 0.5223881})
else:
	if(Hyperbilirubinemia == 'present'):
		pain_ruq ~= choice({'present' : 0.2857143, 'absent' : 0.7142857})
	else:
		pain_ruq ~= choice({'present' : 0.421875, 'absent' : 0.578125})


if (hospital == 'present'):
	if (surgery == 'present'):
		if (choledocholithotomy == 'present'):
			transfusion ~= choice({'present' : 0.3333333, 'absent' : 0.6666667})
		else:
			transfusion ~= choice({'present' : 0.2896175, 'absent' : 0.7103824999999999})
	else:
		if(choledocholithotomy == 'present'):
			transfusion ~= choice({'present' : 0.1666667, 'absent' : 0.8333333})
		else:
			transfusion ~= choice({'present' : 0.1181818, 'absent' : 0.8818182})
else:
	if(surgery == 'present'):
		if(choledocholithotomy == 'present'):
			transfusion ~= choice({'present' : 0.125, 'absent' : 0.875})
		else:
			transfusion ~= choice({'present' : 0.3, 'absent' : 0.7})
	else:
		if(choledocholithotomy == 'present'):
			transfusion ~= choice({'present' : 0.01098901, 'absent' : 0.98901099})
		else:
			transfusion ~= choice({'present' : 0.01079137, 'absent' : 0.98920863})


if (Steatosis == 'present'):
	triglycerides ~= choice({'a17_4' : 0.1791045, 'a3_2' : 0.1641791, 'a1_0' : 0.6567164})
else:
	triglycerides ~= choice({'a17_4' : 0.02373418, 'a3_2' : 0.03164557, 'a1_0' : 0.94462025})


if (transfusion == 'present'):
	if (vh_amn == 'present'):
		if (injections == 'present'):
			ChHepatitis ~= choice({'active' : 0.2094241, 'persistent' : 0.0052356, 'absent' : 0.7853403})
		else:
			ChHepatitis ~= choice({'active' : 0.4615385, 'persistent' : 0.3076923, 'absent' : 0.2307692})
	else:
		if(injections == 'present'):
			ChHepatitis ~= choice({'active' : 0.06, 'persistent' : 0.06, 'absent' : 0.88})
		else:
			ChHepatitis ~= choice({'active' : 0.13043478, 'persistent' : 0.04347826, 'absent' : 0.82608696})
else:
	if(vh_amn == 'present'):
		if(injections == 'present'):
			ChHepatitis ~= choice({'active' : 0.15384615, 'persistent' : 0.05128205, 'absent' : 0.7948718})
		else:
			ChHepatitis ~= choice({'active' : 0.24, 'persistent' : 0.14, 'absent' : 0.62})
	else:
		if(injections == 'present'):
			ChHepatitis ~= choice({'active' : 0.07692308, 'persistent' : 0.00591716, 'absent' : 0.91715976})
		else:
			ChHepatitis ~= choice({'active' : 0.13095238, 'persistent' : 0.05357143, 'absent' : 0.81547619})


if (PBC == 'present'):
	if (ChHepatitis == 'active'):
		if (Steatosis == 'present'):
			if (Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.2704918, 'a49_15' : 0.1721312, 'a14_0' : 0.557377})
			else:
				ESR ~= choice({'a200_50' : 0.2972973, 'a49_15' : 0.1837838, 'a14_0' : 0.5189189000000001})
		else:
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.2941177, 'a49_15' : 0.1911765, 'a14_0' : 0.5147058})
			else:
				ESR ~= choice({'a200_50' : 0.3205575, 'a49_15' : 0.2055749, 'a14_0' : 0.47386759999999994})
	elif(ChHepatitis == 'persistent'):
		if(Steatosis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.3093923, 'a49_15' : 0.1767956, 'a14_0' : 0.5138121})
			else:
				ESR ~= choice({'a200_50' : 0.3315508, 'a49_15' : 0.171123, 'a14_0' : 0.49732620000000005})
		else:
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.3333333, 'a49_15' : 0.172043, 'a14_0' : 0.4946237})
			else:
				ESR ~= choice({'a200_50' : 0.368, 'a49_15' : 0.184, 'a14_0' : 0.44799999999999995})
	else:
		if(Steatosis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.3425926, 'a49_15' : 0.1712963, 'a14_0' : 0.4861111})
			else:
				ESR ~= choice({'a200_50' : 0.3629893, 'a49_15' : 0.1779359, 'a14_0' : 0.4590748})
		else:
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.3636364, 'a49_15' : 0.1818182, 'a14_0' : 0.4545454})
			else:
				ESR ~= choice({'a200_50' : 0.4321429, 'a49_15' : 0.2107143, 'a14_0' : 0.3571428})
else:
	if(ChHepatitis == 'active'):
		if(Steatosis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.2682927, 'a49_15' : 0.1768293, 'a14_0' : 0.554878})
			else:
				ESR ~= choice({'a200_50' : 0.175, 'a49_15' : 0.1625, 'a14_0' : 0.6625})
		else:
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.1045752, 'a49_15' : 0.1633987, 'a14_0' : 0.7320261})
			else:
				ESR ~= choice({'a200_50' : 0.03296703, 'a49_15' : 0.21978022, 'a14_0' : 0.74725275})
	elif(ChHepatitis == 'persistent'):
		if(Steatosis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.06024096, 'a49_15' : 0.12048193, 'a14_0' : 0.81927711})
			else:
				ESR ~= choice({'a200_50' : 0.08602151, 'a49_15' : 0.08602151, 'a14_0' : 0.82795698})
		else:
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.05434783, 'a49_15' : 0.07608696, 'a14_0' : 0.86956521})
			else:
				ESR ~= choice({'a200_50' : 0.05555556, 'a49_15' : 0.05555556, 'a14_0' : 0.88888888})
	else:
		if(Steatosis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.07594937, 'a49_15' : 0.06329114, 'a14_0' : 0.86075949})
			else:
				ESR ~= choice({'a200_50' : 0.13432836, 'a49_15' : 0.05970149, 'a14_0' : 0.80597015})
		else:
			if(Hyperbilirubinemia == 'present'):
				ESR ~= choice({'a200_50' : 0.01785714, 'a49_15' : 0.07142857, 'a14_0' : 0.91071429})
			else:
				ESR ~= choice({'a200_50' : 0.04733728, 'a49_15' : 0.05325444, 'a14_0' : 0.8994082800000001})


if (PBC == 'present'):
	if (Steatosis == 'present'):
		if (ChHepatitis == 'active'):
			cholesterol ~= choice({'a999_350' : 0.08965517, 'a349_240' : 0.28275862, 'a239_0' : 0.62758621})
		elif(ChHepatitis == 'persistent'):
			cholesterol ~= choice({'a999_350' : 0.09659091, 'a349_240' : 0.30113636, 'a239_0' : 0.60227273})
		else:
			cholesterol ~= choice({'a999_350' : 0.1034483, 'a349_240' : 0.3256705, 'a239_0' : 0.5708812000000001})
	else:
		if(ChHepatitis == 'active'):
			cholesterol ~= choice({'a999_350' : 0.1015873, 'a349_240' : 0.3047619, 'a239_0' : 0.5936508})
		elif(ChHepatitis == 'persistent'):
			cholesterol ~= choice({'a999_350' : 0.1050955, 'a349_240' : 0.3152866, 'a239_0' : 0.5796179})
		else:
			cholesterol ~= choice({'a999_350' : 0.125, 'a349_240' : 0.3642857, 'a239_0' : 0.5107143000000001})
else:
	if(Steatosis == 'present'):
		if(ChHepatitis == 'active'):
			cholesterol ~= choice({'a999_350' : 0.09174312, 'a349_240' : 0.27981651, 'a239_0' : 0.62844037})
		elif(ChHepatitis == 'persistent'):
			cholesterol ~= choice({'a999_350' : 0.06918239, 'a349_240' : 0.23899371, 'a239_0' : 0.6918238999999999})
		else:
			cholesterol ~= choice({'a999_350' : 0.04477612, 'a349_240' : 0.2238806, 'a239_0' : 0.7313432799999999})
	else:
		if(ChHepatitis == 'active'):
			cholesterol ~= choice({'a999_350' : 0.03296703, 'a349_240' : 0.06593407, 'a239_0' : 0.9010989})
		elif(ChHepatitis == 'persistent'):
			cholesterol ~= choice({'a999_350' : 0.00277008, 'a349_240' : 0.02770083, 'a239_0' : 0.96952909})
		else:
			cholesterol ~= choice({'a999_350' : 0.00044425, 'a349_240' : 0.09773434, 'a239_0' : 0.90182141})


if (ChHepatitis == 'active'):
	if (THepatitis == 'present'):
		if (RHepatitis == 'present'):
			fatigue ~= choice({'present' : 0.6363636, 'absent' : 0.36363639999999997})
		else:
			fatigue ~= choice({'present' : 0.625, 'absent' : 0.375})
	else:
		if(RHepatitis == 'present'):
			fatigue ~= choice({'present' : 0.6236559, 'absent' : 0.37634409999999996})
		else:
			fatigue ~= choice({'present' : 0.6043956, 'absent' : 0.39560439999999997})
elif(ChHepatitis == 'persistent'):
	if(THepatitis == 'present'):
		if(RHepatitis == 'present'):
			fatigue ~= choice({'present' : 0.6071429, 'absent' : 0.39285709999999996})
		else:
			fatigue ~= choice({'present' : 0.5932203, 'absent' : 0.40677969999999997})
	else:
		if(RHepatitis == 'present'):
			fatigue ~= choice({'present' : 0.5892857, 'absent' : 0.4107143})
		else:
			fatigue ~= choice({'present' : 0.5277778, 'absent' : 0.47222220000000004})
else:
	if(THepatitis == 'present'):
		if(RHepatitis == 'present'):
			fatigue ~= choice({'present' : 0.6153846, 'absent' : 0.38461540000000005})
		else:
			fatigue ~= choice({'present' : 0.6666667, 'absent' : 0.33333330000000005})
	else:
		if(RHepatitis == 'present'):
			fatigue ~= choice({'present' : 0.7058823, 'absent' : 0.29411770000000004})
		else:
			fatigue ~= choice({'present' : 0.5359849, 'absent' : 0.4640151})


if (ChHepatitis == 'active'):
	fibrosis ~= choice({'present' : 0.3, 'absent' : 0.7})
elif(ChHepatitis == 'persistent'):
	fibrosis ~= choice({'present' : 0.05, 'absent' : 0.95})
else:
	fibrosis ~= choice({'present' : 0.001, 'absent' : 0.999})


if (PBC == 'present'):
	if (THepatitis == 'present'):
		if (RHepatitis == 'present'):
			if (Steatosis == 'present'):
				if (ChHepatitis == 'active'):
					if (Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1590909, 'a69_30' : 0.1477273, 'a29_10' : 0.1136364, 'a9_0' : 0.5795454})
					else:
						ggtp ~= choice({'a640_70' : 0.1696429, 'a69_30' : 0.1607143, 'a29_10' : 0.125, 'a9_0' : 0.5446428})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1546392, 'a69_30' : 0.1443299, 'a29_10' : 0.1134021, 'a9_0' : 0.5876288000000001})
					else:
						ggtp ~= choice({'a640_70' : 0.1730769, 'a69_30' : 0.1538462, 'a29_10' : 0.125, 'a9_0' : 0.5480769})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1666667, 'a69_30' : 0.1481482, 'a29_10' : 0.1111111, 'a9_0' : 0.574074})
					else:
						ggtp ~= choice({'a640_70' : 0.1854839, 'a69_30' : 0.1612903, 'a29_10' : 0.1209677, 'a9_0' : 0.5322581})
			else:
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1694915, 'a69_30' : 0.1610169, 'a29_10' : 0.1186441, 'a9_0' : 0.5508475})
					else:
						ggtp ~= choice({'a640_70' : 0.1832061, 'a69_30' : 0.1755725, 'a29_10' : 0.129771, 'a9_0' : 0.5114504})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1759259, 'a69_30' : 0.1574074, 'a29_10' : 0.1111111, 'a9_0' : 0.5555555999999999})
					else:
						ggtp ~= choice({'a640_70' : 0.1913044, 'a69_30' : 0.173913, 'a29_10' : 0.1217391, 'a9_0' : 0.5130435})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1885246, 'a69_30' : 0.1639344, 'a29_10' : 0.1065574, 'a9_0' : 0.5409836})
					else:
						ggtp ~= choice({'a640_70' : 0.2108843, 'a69_30' : 0.1836735, 'a29_10' : 0.1156463, 'a9_0' : 0.48979590000000006})
		else:
			if(Steatosis == 'present'):
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1666667, 'a69_30' : 0.1590909, 'a29_10' : 0.1212121, 'a9_0' : 0.5530303000000001})
					else:
						ggtp ~= choice({'a640_70' : 0.1768707, 'a69_30' : 0.1632653, 'a29_10' : 0.1292517, 'a9_0' : 0.5306123})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1652893, 'a69_30' : 0.1487603, 'a29_10' : 0.1157025, 'a9_0' : 0.5702479})
					else:
						ggtp ~= choice({'a640_70' : 0.1742424, 'a69_30' : 0.1590909, 'a29_10' : 0.1287879, 'a9_0' : 0.5378788})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1726619, 'a69_30' : 0.1510791, 'a29_10' : 0.1151079, 'a9_0' : 0.5611511})
					else:
						ggtp ~= choice({'a640_70' : 0.1893491, 'a69_30' : 0.1656805, 'a29_10' : 0.1242604, 'a9_0' : 0.52071})
			else:
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1741935, 'a69_30' : 0.1677419, 'a29_10' : 0.1225806, 'a9_0' : 0.5354840000000001})
					else:
						ggtp ~= choice({'a640_70' : 0.1857923, 'a69_30' : 0.1857923, 'a29_10' : 0.1311475, 'a9_0' : 0.4972679})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1748252, 'a69_30' : 0.1678322, 'a29_10' : 0.1188811, 'a9_0' : 0.5384614999999999})
					else:
						ggtp ~= choice({'a640_70' : 0.19375, 'a69_30' : 0.18125, 'a29_10' : 0.125, 'a9_0' : 0.5})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1918605, 'a69_30' : 0.1744186, 'a29_10' : 0.1104651, 'a9_0' : 0.5232558})
					else:
						ggtp ~= choice({'a640_70' : 0.2142857, 'a69_30' : 0.1932773, 'a29_10' : 0.1176471, 'a9_0' : 0.4747899})
	else:
		if(RHepatitis == 'present'):
			if(Steatosis == 'present'):
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1756757, 'a69_30' : 0.1621622, 'a29_10' : 0.1216216, 'a9_0' : 0.5405405})
					else:
						ggtp ~= choice({'a640_70' : 0.18, 'a69_30' : 0.1666667, 'a29_10' : 0.1333333, 'a9_0' : 0.52})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1666667, 'a69_30' : 0.15, 'a29_10' : 0.125, 'a9_0' : 0.5583333})
					else:
						ggtp ~= choice({'a640_70' : 0.1782946, 'a69_30' : 0.1627907, 'a29_10' : 0.131783, 'a9_0' : 0.5271317})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1764706, 'a69_30' : 0.1544118, 'a29_10' : 0.1176471, 'a9_0' : 0.5514705})
					else:
						ggtp ~= choice({'a640_70' : 0.1939394, 'a69_30' : 0.169697, 'a29_10' : 0.1272727, 'a9_0' : 0.5090909})
			else:
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1776316, 'a69_30' : 0.1710526, 'a29_10' : 0.125, 'a9_0' : 0.5263158})
					else:
						ggtp ~= choice({'a640_70' : 0.1899441, 'a69_30' : 0.1899441, 'a29_10' : 0.1340782, 'a9_0' : 0.48603359999999995})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1785714, 'a69_30' : 0.1714286, 'a29_10' : 0.1214286, 'a9_0' : 0.5285714})
					else:
						ggtp ~= choice({'a640_70' : 0.1987179, 'a69_30' : 0.1858974, 'a29_10' : 0.1282051, 'a9_0' : 0.48717960000000005})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1964286, 'a69_30' : 0.1785714, 'a29_10' : 0.1130952, 'a9_0' : 0.5119047999999999})
					else:
						ggtp ~= choice({'a640_70' : 0.2207792, 'a69_30' : 0.1991342, 'a29_10' : 0.1212121, 'a9_0' : 0.45887449999999996})
		else:
			if(Steatosis == 'present'):
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.178771, 'a69_30' : 0.1731844, 'a29_10' : 0.122905, 'a9_0' : 0.5251395999999999})
					else:
						ggtp ~= choice({'a640_70' : 0.1804878, 'a69_30' : 0.1756098, 'a29_10' : 0.1365854, 'a9_0' : 0.507317})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1698113, 'a69_30' : 0.1572327, 'a29_10' : 0.1257862, 'a9_0' : 0.5471698})
					else:
						ggtp ~= choice({'a640_70' : 0.1843575, 'a69_30' : 0.1675978, 'a29_10' : 0.1340782, 'a9_0' : 0.5139665})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1822917, 'a69_30' : 0.1614583, 'a29_10' : 0.1197917, 'a9_0' : 0.5364583})
					else:
						ggtp ~= choice({'a640_70' : 0.1977612, 'a69_30' : 0.1791045, 'a29_10' : 0.1268657, 'a9_0' : 0.49626860000000006})
			else:
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1826087, 'a69_30' : 0.1782609, 'a29_10' : 0.126087, 'a9_0' : 0.5130433999999999})
					else:
						ggtp ~= choice({'a640_70' : 0.1939799, 'a69_30' : 0.1939799, 'a29_10' : 0.1371237, 'a9_0' : 0.47491649999999996})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1846847, 'a69_30' : 0.1846847, 'a29_10' : 0.1261261, 'a9_0' : 0.5045045})
					else:
						ggtp ~= choice({'a640_70' : 0.2007435, 'a69_30' : 0.197026, 'a29_10' : 0.133829, 'a9_0' : 0.4684015})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1986755, 'a69_30' : 0.192053, 'a29_10' : 0.1225166, 'a9_0' : 0.4867549})
					else:
						ggtp ~= choice({'a640_70' : 0.2392857, 'a69_30' : 0.225, 'a29_10' : 0.1321429, 'a9_0' : 0.4035713999999999})
else:
	if(THepatitis == 'present'):
		if(RHepatitis == 'present'):
			if(Steatosis == 'present'):
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.1509434, 'a69_30' : 0.1415094, 'a29_10' : 0.1226415, 'a9_0' : 0.5849057})
					else:
						ggtp ~= choice({'a640_70' : 0.10526316, 'a69_30' : 0.09210526, 'a29_10' : 0.13157895, 'a9_0' : 0.67105263})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.05555556, 'a69_30' : 0.03703704, 'a29_10' : 0.09259259, 'a9_0' : 0.81481481})
					else:
						ggtp ~= choice({'a640_70' : 0.06122449, 'a69_30' : 0.02040816, 'a29_10' : 0.10204082, 'a9_0' : 0.81632653})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.05649718, 'a69_30' : 0.00188324, 'a29_10' : 0.07532957, 'a9_0' : 0.8662900099999999})
					else:
						ggtp ~= choice({'a640_70' : 0.07393715, 'a69_30' : 0.00184843, 'a29_10' : 0.09242144, 'a9_0' : 0.83179298})
			else:
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.06557377, 'a69_30' : 0.04918033, 'a29_10' : 0.09836066, 'a9_0' : 0.7868852399999999})
					else:
						ggtp ~= choice({'a640_70' : 0.078125, 'a69_30' : 0.078125, 'a29_10' : 0.125, 'a9_0' : 0.71875})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.04166667, 'a69_30' : 0.02083333, 'a29_10' : 0.08333333, 'a9_0' : 0.85416667})
					else:
						ggtp ~= choice({'a640_70' : 0.04761905, 'a69_30' : 0.02380952, 'a29_10' : 0.0952381, 'a9_0' : 0.83333333})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.04338395, 'a69_30' : 0.0021692, 'a29_10' : 0.04338395, 'a9_0' : 0.9110629})
					else:
						ggtp ~= choice({'a640_70' : 0.06651885, 'a69_30' : 0.00221729, 'a29_10' : 0.0443459, 'a9_0' : 0.88691796})
		else:
			if(Steatosis == 'present'):
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.05714286, 'a69_30' : 0.04285714, 'a29_10' : 0.1, 'a9_0' : 0.8})
					else:
						ggtp ~= choice({'a640_70' : 0.07142857, 'a69_30' : 0.07142857, 'a29_10' : 0.13095238, 'a9_0' : 0.72619048})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.04545455, 'a69_30' : 0.03030303, 'a29_10' : 0.10606061, 'a9_0' : 0.81818181})
					else:
						ggtp ~= choice({'a640_70' : 0.04615385, 'a69_30' : 0.03076923, 'a29_10' : 0.12307692, 'a9_0' : 0.8})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.04285714, 'a69_30' : 0.01428571, 'a29_10' : 0.08571429, 'a9_0' : 0.85714286})
					else:
						ggtp ~= choice({'a640_70' : 0.0617284, 'a69_30' : 0.01234568, 'a29_10' : 0.09876543, 'a9_0' : 0.82716049})
			else:
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.06024096, 'a69_30' : 0.04819277, 'a29_10' : 0.09638554, 'a9_0' : 0.79518073})
					else:
						ggtp ~= choice({'a640_70' : 0.07070707, 'a69_30' : 0.08080808, 'a29_10' : 0.12121212, 'a9_0' : 0.72727273})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.04166667, 'a69_30' : 0.04166667, 'a29_10' : 0.09722222, 'a9_0' : 0.81944444})
					else:
						ggtp ~= choice({'a640_70' : 0.03030303, 'a69_30' : 0.03030303, 'a29_10' : 0.10606061, 'a9_0' : 0.83333333})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.02702703, 'a69_30' : 0.01351351, 'a29_10' : 0.05405405, 'a9_0' : 0.90540541})
					else:
						ggtp ~= choice({'a640_70' : 0.07380074, 'a69_30' : 0.00369004, 'a29_10' : 0.03690037, 'a9_0' : 0.88560885})
	else:
		if(RHepatitis == 'present'):
			if(Steatosis == 'present'):
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.06349206, 'a69_30' : 0.04761905, 'a29_10' : 0.11111111, 'a9_0' : 0.77777778})
					else:
						ggtp ~= choice({'a640_70' : 0.07594937, 'a69_30' : 0.07594937, 'a29_10' : 0.13924051, 'a9_0' : 0.7088607499999999})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.0483871, 'a69_30' : 0.03225806, 'a29_10' : 0.11290323, 'a9_0' : 0.80645161})
					else:
						ggtp ~= choice({'a640_70' : 0.05, 'a69_30' : 0.03333333, 'a29_10' : 0.13333333, 'a9_0' : 0.78333334})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.04615385, 'a69_30' : 0.01538462, 'a29_10' : 0.09230769, 'a9_0' : 0.8461538399999999})
					else:
						ggtp ~= choice({'a640_70' : 0.06756757, 'a69_30' : 0.01351351, 'a29_10' : 0.10810811, 'a9_0' : 0.81081081})
			else:
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.06410256, 'a69_30' : 0.05128205, 'a29_10' : 0.1025641, 'a9_0' : 0.7820512900000001})
					else:
						ggtp ~= choice({'a640_70' : 0.07692308, 'a69_30' : 0.08791209, 'a29_10' : 0.13186813, 'a9_0' : 0.7032967})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.04545455, 'a69_30' : 0.04545455, 'a29_10' : 0.10606061, 'a9_0' : 0.8030302899999999})
					else:
						ggtp ~= choice({'a640_70' : 0.03448276, 'a69_30' : 0.03448276, 'a29_10' : 0.12068966, 'a9_0' : 0.81034482})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.03076923, 'a69_30' : 0.01538462, 'a29_10' : 0.06153846, 'a9_0' : 0.89230769})
					else:
						ggtp ~= choice({'a640_70' : 0.11695906, 'a69_30' : 0.00584795, 'a29_10' : 0.05847953, 'a9_0' : 0.81871346})
		else:
			if(Steatosis == 'present'):
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.06493506, 'a69_30' : 0.06493506, 'a29_10' : 0.11688312, 'a9_0' : 0.75324676})
					else:
						ggtp ~= choice({'a640_70' : 0.07692308, 'a69_30' : 0.08547009, 'a29_10' : 0.14529915, 'a9_0' : 0.69230768})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.04444444, 'a69_30' : 0.04444444, 'a29_10' : 0.12222222, 'a9_0' : 0.7888889})
					else:
						ggtp ~= choice({'a640_70' : 0.04210526, 'a69_30' : 0.04210526, 'a29_10' : 0.13684211, 'a9_0' : 0.77894737})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.03703704, 'a69_30' : 0.02777778, 'a29_10' : 0.10185185, 'a9_0' : 0.83333333})
					else:
						ggtp ~= choice({'a640_70' : 0.07462687, 'a69_30' : 0.02985075, 'a29_10' : 0.13432836, 'a9_0' : 0.76119402})
			else:
				if(ChHepatitis == 'active'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.05660377, 'a69_30' : 0.06603774, 'a29_10' : 0.12264151, 'a9_0' : 0.75471698})
					else:
						ggtp ~= choice({'a640_70' : 0.08791209, 'a69_30' : 0.14285714, 'a29_10' : 0.17582418, 'a9_0' : 0.5934065900000001})
				elif(ChHepatitis == 'persistent'):
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.04395604, 'a69_30' : 0.07692308, 'a29_10' : 0.13186813, 'a9_0' : 0.74725275})
					else:
						ggtp ~= choice({'a640_70' : 0.00277008, 'a69_30' : 0.05540166, 'a29_10' : 0.19390582, 'a9_0' : 0.74792244})
				else:
					if(Hyperbilirubinemia == 'present'):
						ggtp ~= choice({'a640_70' : 0.00177936, 'a69_30' : 0.00177936, 'a29_10' : 0.01779359, 'a9_0' : 0.97864769})
					else:
						ggtp ~= choice({'a640_70' : 0.08, 'a69_30' : 0.096, 'a29_10' : 0.144, 'a9_0' : 0.68})


if (vh_amn == 'present'):
	if (ChHepatitis == 'active'):
		hbc_anti ~= choice({'present' : 0.00355872, 'absent' : 0.99644128})
	elif(ChHepatitis == 'persistent'):
		hbc_anti ~= choice({'present' : 0.00763359, 'absent' : 0.99236641})
	else:
		hbc_anti ~= choice({'present' : 0.0875, 'absent' : 0.9125})
else:
	if(ChHepatitis == 'active'):
		hbc_anti ~= choice({'present' : 0.07936508, 'absent' : 0.92063492})
	elif(ChHepatitis == 'persistent'):
		hbc_anti ~= choice({'present' : 0.1304348, 'absent' : 0.8695652})
	else:
		hbc_anti ~= choice({'present' : 0.101626, 'absent' : 0.898374})


if (vh_amn == 'present'):
	if (ChHepatitis == 'active'):
		hbeag ~= choice({'present' : 0.00355872, 'absent' : 0.99644128})
	elif(ChHepatitis == 'persistent'):
		hbeag ~= choice({'present' : 0.00763359, 'absent' : 0.99236641})
	else:
		hbeag ~= choice({'present' : 0.00124844, 'absent' : 0.99875156})
else:
	if(ChHepatitis == 'active'):
		hbeag ~= choice({'present' : 0.00158479, 'absent' : 0.99841521})
	elif(ChHepatitis == 'persistent'):
		hbeag ~= choice({'present' : 0.04347826, 'absent' : 0.95652174})
	else:
		hbeag ~= choice({'present' : 0.00203252, 'absent' : 0.99796748})


if (vh_amn == 'present'):
	if (ChHepatitis == 'active'):
		hbsag ~= choice({'present' : 0.5, 'absent' : 0.5})
	elif(ChHepatitis == 'persistent'):
		hbsag ~= choice({'present' : 0.4615385, 'absent' : 0.5384614999999999})
	else:
		hbsag ~= choice({'present' : 0.1125, 'absent' : 0.8875})
else:
	if(ChHepatitis == 'active'):
		hbsag ~= choice({'present' : 0.1904762, 'absent' : 0.8095238})
	elif(ChHepatitis == 'persistent'):
		hbsag ~= choice({'present' : 0.04347826, 'absent' : 0.95652174})
	else:
		hbsag ~= choice({'present' : 0.04674797, 'absent' : 0.95325203})


if (vh_amn == 'present'):
	if (ChHepatitis == 'active'):
		if (hbsag == 'present'):
			hbsag_anti ~= choice({'present' : 0.0070922, 'absent' : 0.9929078})
		else:
			hbsag_anti ~= choice({'present' : 0.07142857, 'absent' : 0.92857143})
	elif(ChHepatitis == 'persistent'):
		if(hbsag == 'present'):
			hbsag_anti ~= choice({'present' : 0.01639344, 'absent' : 0.98360656})
		else:
			hbsag_anti ~= choice({'present' : 0.01408451, 'absent' : 0.98591549})
	else:
		if(hbsag == 'present'):
			hbsag_anti ~= choice({'present' : 0.01098901, 'absent' : 0.98901099})
		else:
			hbsag_anti ~= choice({'present' : 0.04225352, 'absent' : 0.95774648})
else:
	if(ChHepatitis == 'active'):
		if(hbsag == 'present'):
			hbsag_anti ~= choice({'present' : 0.08333333, 'absent' : 0.91666667})
		else:
			hbsag_anti ~= choice({'present' : 0.00195695, 'absent' : 0.99804305})
	elif(ChHepatitis == 'persistent'):
		if(hbsag == 'present'):
			hbsag_anti ~= choice({'present' : 0.09090909, 'absent' : 0.90909091})
		else:
			hbsag_anti ~= choice({'present' : 0.00452489, 'absent' : 0.99547511})
	else:
		if(hbsag == 'present'):
			hbsag_anti ~= choice({'present' : 0.004329, 'absent' : 0.995671})
		else:
			hbsag_anti ~= choice({'present' : 0.01492537, 'absent' : 0.98507463})


if (vh_amn == 'present'):
	if (ChHepatitis == 'active'):
		hcv_anti ~= choice({'present' : 0.00355872, 'absent' : 0.99644128})
	elif(ChHepatitis == 'persistent'):
		hcv_anti ~= choice({'present' : 0.00763359, 'absent' : 0.99236641})
	else:
		hcv_anti ~= choice({'present' : 0.00124844, 'absent' : 0.99875156})
else:
	if(ChHepatitis == 'active'):
		hcv_anti ~= choice({'present' : 0.00158479, 'absent' : 0.99841521})
	elif(ChHepatitis == 'persistent'):
		hcv_anti ~= choice({'present' : 0.004329, 'absent' : 0.995671})
	else:
		hcv_anti ~= choice({'present' : 0.00203252, 'absent' : 0.99796748})


if (hepatomegaly == 'present'):
	hepatalgia ~= choice({'present' : 0.3142251, 'absent' : 0.6857749})
else:
	hepatalgia ~= choice({'present' : 0.03070175, 'absent' : 0.96929825})


if (gallstones == 'present'):
	if (PBC == 'present'):
		if (ChHepatitis == 'active'):
			pressure_ruq ~= choice({'present' : 0.3333333, 'absent' : 0.6666667})
		elif(ChHepatitis == 'persistent'):
			pressure_ruq ~= choice({'present' : 0.328125, 'absent' : 0.671875})
		else:
			pressure_ruq ~= choice({'present' : 0.3292683, 'absent' : 0.6707316999999999})
	else:
		if(ChHepatitis == 'active'):
			pressure_ruq ~= choice({'present' : 0.4, 'absent' : 0.6})
		elif(ChHepatitis == 'persistent'):
			pressure_ruq ~= choice({'present' : 0.09090909, 'absent' : 0.90909091})
		else:
			pressure_ruq ~= choice({'present' : 0.2857143, 'absent' : 0.7142857})
else:
	if(PBC == 'present'):
		if(ChHepatitis == 'active'):
			pressure_ruq ~= choice({'present' : 0.3424658, 'absent' : 0.6575342})
		elif(ChHepatitis == 'persistent'):
			pressure_ruq ~= choice({'present' : 0.3227513, 'absent' : 0.6772487})
		else:
			pressure_ruq ~= choice({'present' : 0.2929293, 'absent' : 0.7070707})
	else:
		if(ChHepatitis == 'active'):
			pressure_ruq ~= choice({'present' : 0.4691358, 'absent' : 0.5308642})
		elif(ChHepatitis == 'persistent'):
			pressure_ruq ~= choice({'present' : 0.4285714, 'absent' : 0.5714286})
		else:
			pressure_ruq ~= choice({'present' : 0.4532374, 'absent' : 0.5467626})


if (fibrosis == 'present'):
	if (Steatosis == 'present'):
		Cirrhosis ~= choice({'decompensate' : 0.56, 'compensate' : 0.24, 'absent' : 0.19999999999999996})
	else:
		Cirrhosis ~= choice({'decompensate' : 0.49, 'compensate' : 0.21, 'absent' : 0.30000000000000004})
else:
	if(Steatosis == 'present'):
		Cirrhosis ~= choice({'decompensate' : 0.35, 'compensate' : 0.15, 'absent' : 0.5})
	else:
		Cirrhosis ~= choice({'decompensate' : 0.001, 'compensate' : 0.001, 'absent' : 0.998})


if (Cirrhosis == 'decompensate'):
	albumin ~= choice({'a70_50' : 0.91222031, 'a49_30' : 0.08605852, 'a29_0' : 0.0017211699999999386})
elif(Cirrhosis == 'compensate'):
	albumin ~= choice({'a70_50' : 0.96463023, 'a49_30' : 0.00321543, 'a29_0' : 0.03215433999999995})
else:
	albumin ~= choice({'a70_50' : 0.7393443, 'a49_30' : 0.1426229, 'a29_0' : 0.11803280000000005})


if (Cirrhosis == 'decompensate'):
	alcohol ~= choice({'present' : 0.2068966, 'absent' : 0.7931034})
elif(Cirrhosis == 'compensate'):
	alcohol ~= choice({'present' : 0.2258064, 'absent' : 0.7741936})
else:
	alcohol ~= choice({'present' : 0.1114754, 'absent' : 0.8885246})


if (ChHepatitis == 'active'):
	if (RHepatitis == 'present'):
		if (THepatitis == 'present'):
			if (Steatosis == 'present'):
				if (Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.05882353, 'a199_100' : 0.15686275, 'a99_35' : 0.41176471, 'a34_0' : 0.37254900999999996})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.05454545, 'a199_100' : 0.16363636, 'a99_35' : 0.41818182, 'a34_0' : 0.36363637000000004})
				else:
					alt ~= choice({'a850_200' : 0.04761905, 'a199_100' : 0.15873016, 'a99_35' : 0.41269841, 'a34_0' : 0.38095238})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.06451613, 'a199_100' : 0.17741935, 'a99_35' : 0.41935484, 'a34_0' : 0.33870968})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.07017544, 'a199_100' : 0.19298246, 'a99_35' : 0.42105263, 'a34_0' : 0.31578947})
				else:
					alt ~= choice({'a850_200' : 0.07936508, 'a199_100' : 0.19047619, 'a99_35' : 0.41269841, 'a34_0' : 0.3174603199999999})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.06849315, 'a199_100' : 0.16438356, 'a99_35' : 0.42465753, 'a34_0' : 0.34246575999999995})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.05882353, 'a199_100' : 0.17647059, 'a99_35' : 0.42647059, 'a34_0' : 0.3382352900000001})
				else:
					alt ~= choice({'a850_200' : 0.0625, 'a199_100' : 0.175, 'a99_35' : 0.4125, 'a34_0' : 0.3500000000000001})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.075, 'a199_100' : 0.1875, 'a99_35' : 0.425, 'a34_0' : 0.3125})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.08333333, 'a199_100' : 0.20833333, 'a99_35' : 0.43055556, 'a34_0' : 0.27777778})
				else:
					alt ~= choice({'a850_200' : 0.08988764, 'a199_100' : 0.2247191, 'a99_35' : 0.41573034, 'a34_0' : 0.26966292000000003})
	else:
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.0617284, 'a199_100' : 0.1728395, 'a99_35' : 0.4197531, 'a34_0' : 0.34567900000000007})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.05479452, 'a199_100' : 0.16438356, 'a99_35' : 0.42465753, 'a34_0' : 0.35616439})
				else:
					alt ~= choice({'a850_200' : 0.05882353, 'a199_100' : 0.15294118, 'a99_35' : 0.41176471, 'a34_0' : 0.37647058})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.06976744, 'a199_100' : 0.1627907, 'a99_35' : 0.41860465, 'a34_0' : 0.3488372099999999})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.07792208, 'a199_100' : 0.18181818, 'a99_35' : 0.41558442, 'a34_0' : 0.32467532})
				else:
					alt ~= choice({'a850_200' : 0.08333333, 'a199_100' : 0.1875, 'a99_35' : 0.40625, 'a34_0' : 0.32291667})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.06862745, 'a199_100' : 0.16666667, 'a99_35' : 0.42156863, 'a34_0' : 0.34313725000000006})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.0625, 'a199_100' : 0.1666667, 'a99_35' : 0.4270833, 'a34_0' : 0.34375})
				else:
					alt ~= choice({'a850_200' : 0.064, 'a199_100' : 0.168, 'a99_35' : 0.416, 'a34_0' : 0.352})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.08148148, 'a199_100' : 0.17777778, 'a99_35' : 0.42222222, 'a34_0' : 0.31851852})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.08661417, 'a199_100' : 0.19685039, 'a99_35' : 0.42519685, 'a34_0' : 0.29133859000000006})
				else:
					alt ~= choice({'a850_200' : 0.1208791, 'a199_100' : 0.2307692, 'a99_35' : 0.3956044, 'a34_0' : 0.2527473})
elif(ChHepatitis == 'persistent'):
	if(RHepatitis == 'present'):
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.05172414, 'a199_100' : 0.15517241, 'a99_35' : 0.39655172, 'a34_0' : 0.39655173})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.02173913, 'a199_100' : 0.13043478, 'a99_35' : 0.41304348, 'a34_0' : 0.43478260999999996})
				else:
					alt ~= choice({'a850_200' : 0.0021692, 'a199_100' : 0.1084599, 'a99_35' : 0.3904555, 'a34_0' : 0.4989154})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.02272727, 'a199_100' : 0.11363636, 'a99_35' : 0.40909091, 'a34_0' : 0.45454546000000007})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.00269542, 'a199_100' : 0.13477089, 'a99_35' : 0.40431267, 'a34_0' : 0.45822102})
				else:
					alt ~= choice({'a850_200' : 0.00262467, 'a199_100' : 0.1312336, 'a99_35' : 0.36745407, 'a34_0' : 0.49868766})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.01923077, 'a199_100' : 0.11538462, 'a99_35' : 0.40384615, 'a34_0' : 0.46153846})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.002079, 'a199_100' : 0.1247401, 'a99_35' : 0.4158004, 'a34_0' : 0.45738049999999997})
				else:
					alt ~= choice({'a850_200' : 0.00181488, 'a199_100' : 0.12704174, 'a99_35' : 0.39927405, 'a34_0' : 0.4718693300000001})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.01886792, 'a199_100' : 0.13207547, 'a99_35' : 0.41509434, 'a34_0' : 0.43396227})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.02272727, 'a199_100' : 0.15909091, 'a99_35' : 0.40909091, 'a34_0' : 0.40909091})
				else:
					alt ~= choice({'a850_200' : 0.02083333, 'a199_100' : 0.16666667, 'a99_35' : 0.375, 'a34_0' : 0.4375})
	else:
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.01724138, 'a199_100' : 0.12068966, 'a99_35' : 0.39655172, 'a34_0' : 0.46551724000000005})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.00191939, 'a199_100' : 0.11516315, 'a99_35' : 0.40307102, 'a34_0' : 0.47984644})
				else:
					alt ~= choice({'a850_200' : 0.00166389, 'a199_100' : 0.09983361, 'a99_35' : 0.38269551, 'a34_0' : 0.51580699})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.01724138, 'a199_100' : 0.10344828, 'a99_35' : 0.39655172, 'a34_0' : 0.48275862})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.02040816, 'a199_100' : 0.12244898, 'a99_35' : 0.3877551, 'a34_0' : 0.46938776000000004})
				else:
					alt ~= choice({'a850_200' : 0.01818182, 'a199_100' : 0.10909091, 'a99_35' : 0.34545455, 'a34_0' : 0.52727272})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.02777778, 'a199_100' : 0.11111111, 'a99_35' : 0.38888889, 'a34_0' : 0.47222222})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.01492537, 'a199_100' : 0.11940299, 'a99_35' : 0.40298507, 'a34_0' : 0.46268657})
				else:
					alt ~= choice({'a850_200' : 0.01190476, 'a199_100' : 0.10714286, 'a99_35' : 0.38095238, 'a34_0' : 0.5})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.03409091, 'a199_100' : 0.11363636, 'a99_35' : 0.38636364, 'a34_0' : 0.46590909})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.02631579, 'a199_100' : 0.13157895, 'a99_35' : 0.39473684, 'a34_0' : 0.44736842})
				else:
					alt ~= choice({'a850_200' : 0.02777778, 'a199_100' : 0.13888889, 'a99_35' : 0.27777778, 'a34_0' : 0.55555555})
else:
	if(RHepatitis == 'present'):
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.02, 'a199_100' : 0.12, 'a99_35' : 0.4, 'a34_0' : 0.45999999999999996})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.00212314, 'a199_100' : 0.12738854, 'a99_35' : 0.42462845, 'a34_0' : 0.44585987000000005})
				else:
					alt ~= choice({'a850_200' : 0.00191939, 'a199_100' : 0.11516315, 'a99_35' : 0.42226488, 'a34_0' : 0.46065258})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.02, 'a199_100' : 0.12, 'a99_35' : 0.44, 'a34_0' : 0.42000000000000004})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.00249377, 'a199_100' : 0.14962594, 'a99_35' : 0.44887781, 'a34_0' : 0.39900248000000005})
				else:
					alt ~= choice({'a850_200' : 0.0023753, 'a199_100' : 0.1425178, 'a99_35' : 0.4275534, 'a34_0' : 0.42755350000000003})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.01666667, 'a199_100' : 0.11666667, 'a99_35' : 0.45, 'a34_0' : 0.41666665999999997})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.00178253, 'a199_100' : 0.12477718, 'a99_35' : 0.46345811, 'a34_0' : 0.4099821800000001})
				else:
					alt ~= choice({'a850_200' : 0.00144718, 'a199_100' : 0.11577424, 'a99_35' : 0.44862518, 'a34_0' : 0.4341534})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.02816901, 'a199_100' : 0.12676056, 'a99_35' : 0.46478873, 'a34_0' : 0.38028170000000006})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.01724138, 'a199_100' : 0.15517241, 'a99_35' : 0.48275862, 'a34_0' : 0.34482758999999996})
				else:
					alt ~= choice({'a850_200' : 0.00584795, 'a199_100' : 0.23391813, 'a99_35' : 0.46783626, 'a34_0' : 0.29239766})
	else:
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.01818182, 'a199_100' : 0.10909091, 'a99_35' : 0.43636364, 'a34_0' : 0.43636363})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.00172117, 'a199_100' : 0.10327022, 'a99_35' : 0.4475043, 'a34_0' : 0.44750431})
				else:
					alt ~= choice({'a850_200' : 0.00131406, 'a199_100' : 0.09198423, 'a99_35' : 0.42049934, 'a34_0' : 0.48620237})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.025, 'a199_100' : 0.1, 'a99_35' : 0.425, 'a34_0' : 0.44999999999999996})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.01470588, 'a199_100' : 0.11764706, 'a99_35' : 0.44117647, 'a34_0' : 0.42647059})
				else:
					alt ~= choice({'a850_200' : 0.00369004, 'a199_100' : 0.07380074, 'a99_35' : 0.36900369, 'a34_0' : 0.55350553})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.02666667, 'a199_100' : 0.09333333, 'a99_35' : 0.42666667, 'a34_0' : 0.45333333})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.01176471, 'a199_100' : 0.10588235, 'a99_35' : 0.44705882, 'a34_0' : 0.43529412})
				else:
					alt ~= choice({'a850_200' : 0.00149031, 'a199_100' : 0.08941878, 'a99_35' : 0.41728763, 'a34_0' : 0.49180327999999995})
			else:
				if(Cirrhosis == 'decompensate'):
					alt ~= choice({'a850_200' : 0.06896552, 'a199_100' : 0.12068966, 'a99_35' : 0.46551724, 'a34_0' : 0.34482758})
				elif(Cirrhosis == 'compensate'):
					alt ~= choice({'a850_200' : 0.03225806, 'a199_100' : 0.19354839, 'a99_35' : 0.51612903, 'a34_0' : 0.25806452})
				else:
					alt ~= choice({'a850_200' : 0.04569892, 'a199_100' : 0.17473118, 'a99_35' : 0.42741935, 'a34_0' : 0.35215054999999995})


if (ChHepatitis == 'active'):
	if (RHepatitis == 'present'):
		if (THepatitis == 'present'):
			if (Steatosis == 'present'):
				if (Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.01960784, 'a399_150' : 0.1372549, 'a149_40' : 0.47058824, 'a39_0' : 0.37254902})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.01818182, 'a399_150' : 0.12727273, 'a149_40' : 0.49090909, 'a39_0' : 0.36363636})
				else:
					ast ~= choice({'a700_400' : 0.01612903, 'a399_150' : 0.14516129, 'a149_40' : 0.46774194, 'a39_0' : 0.37096774})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.01612903, 'a399_150' : 0.16129032, 'a149_40' : 0.48387097, 'a39_0' : 0.33870968})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.01818182, 'a399_150' : 0.16363636, 'a149_40' : 0.50909091, 'a39_0' : 0.3090909100000001})
				else:
					ast ~= choice({'a700_400' : 0.03225806, 'a399_150' : 0.17741935, 'a149_40' : 0.48387097, 'a39_0' : 0.30645162000000004})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.02777778, 'a399_150' : 0.15277778, 'a149_40' : 0.48611111, 'a39_0' : 0.33333333})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.02941176, 'a399_150' : 0.14705882, 'a149_40' : 0.50000001, 'a39_0' : 0.3235294099999999})
				else:
					ast ~= choice({'a700_400' : 0.02531646, 'a399_150' : 0.15189873, 'a149_40' : 0.48101266, 'a39_0' : 0.34177214999999994})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.025, 'a399_150' : 0.175, 'a149_40' : 0.5, 'a39_0' : 0.30000000000000004})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.02777778, 'a399_150' : 0.18055556, 'a149_40' : 0.52777777, 'a39_0' : 0.26388888999999993})
				else:
					ast ~= choice({'a700_400' : 0.03370787, 'a399_150' : 0.20224719, 'a149_40' : 0.50561797, 'a39_0' : 0.25842697000000003})
	else:
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.02469136, 'a399_150' : 0.16049383, 'a149_40' : 0.4691358, 'a39_0' : 0.34567901})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.02739726, 'a399_150' : 0.1369863, 'a149_40' : 0.47945205, 'a39_0' : 0.35616439})
				else:
					ast ~= choice({'a700_400' : 0.02380952, 'a399_150' : 0.14285714, 'a149_40' : 0.45238095, 'a39_0' : 0.38095239000000003})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.02352941, 'a399_150' : 0.16470588, 'a149_40' : 0.45882353, 'a39_0' : 0.35294117999999997})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.02597403, 'a399_150' : 0.16883117, 'a149_40' : 0.48051948, 'a39_0' : 0.32467532})
				else:
					ast ~= choice({'a700_400' : 0.03125, 'a399_150' : 0.1875, 'a149_40' : 0.4583333, 'a39_0' : 0.32291669999999995})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.02912621, 'a399_150' : 0.16504854, 'a149_40' : 0.46601942, 'a39_0' : 0.33980583})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.03125, 'a399_150' : 0.15625, 'a149_40' : 0.4791667, 'a39_0' : 0.33333330000000005})
				else:
					ast ~= choice({'a700_400' : 0.03174603, 'a399_150' : 0.15873016, 'a149_40' : 0.46031746, 'a39_0' : 0.34920635})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.03676471, 'a399_150' : 0.17647059, 'a149_40' : 0.47058824, 'a39_0' : 0.31617646})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.03937008, 'a399_150' : 0.18110236, 'a149_40' : 0.49606299, 'a39_0' : 0.28346457})
				else:
					ast ~= choice({'a700_400' : 0.05494505, 'a399_150' : 0.23076923, 'a149_40' : 0.46153846, 'a39_0' : 0.25274726000000003})
elif(ChHepatitis == 'persistent'):
	if(RHepatitis == 'present'):
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.01754386, 'a399_150' : 0.14035088, 'a149_40' : 0.45614035, 'a39_0' : 0.38596491})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00221729, 'a399_150' : 0.0886918, 'a149_40' : 0.48780488, 'a39_0' : 0.42128603})
				else:
					ast ~= choice({'a700_400' : 0.00212314, 'a399_150' : 0.08492569, 'a149_40' : 0.44585987, 'a39_0' : 0.4670913})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00221729, 'a399_150' : 0.11086475, 'a149_40' : 0.46563193, 'a39_0' : 0.42128603})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00269542, 'a399_150' : 0.08086253, 'a149_40' : 0.51212938, 'a39_0' : 0.40431267000000004})
				else:
					ast ~= choice({'a700_400' : 0.00269542, 'a399_150' : 0.08086253, 'a149_40' : 0.45822102, 'a39_0' : 0.45822103000000003})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00191939, 'a399_150' : 0.09596929, 'a149_40' : 0.46065259, 'a39_0' : 0.44145873})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00203666, 'a399_150' : 0.0814664, 'a149_40' : 0.48879837, 'a39_0' : 0.42769857})
				else:
					ast ~= choice({'a700_400' : 0.00181488, 'a399_150' : 0.0907441, 'a149_40' : 0.45372051, 'a39_0' : 0.45372051})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00184843, 'a399_150' : 0.11090573, 'a149_40' : 0.4805915, 'a39_0' : 0.40665434})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00226757, 'a399_150' : 0.09070295, 'a149_40' : 0.52154195, 'a39_0' : 0.38548753})
				else:
					ast ~= choice({'a700_400' : 0.00212314, 'a399_150' : 0.10615711, 'a149_40' : 0.48832272, 'a39_0' : 0.40339703000000005})
	else:
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00172117, 'a399_150' : 0.10327022, 'a149_40' : 0.4475043, 'a39_0' : 0.44750431})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00191939, 'a399_150' : 0.07677543, 'a149_40' : 0.46065259, 'a39_0' : 0.46065259000000003})
				else:
					ast ~= choice({'a700_400' : 0.00166389, 'a399_150' : 0.08319468, 'a149_40' : 0.41597338, 'a39_0' : 0.49916805})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00172117, 'a399_150' : 0.10327022, 'a149_40' : 0.4302926, 'a39_0' : 0.46471600999999996})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00203666, 'a399_150' : 0.0814664, 'a149_40' : 0.46843177, 'a39_0' : 0.44806517})
				else:
					ast ~= choice({'a700_400' : 0.00181488, 'a399_150' : 0.0907441, 'a149_40' : 0.41742287, 'a39_0' : 0.49001815000000004})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00142653, 'a399_150' : 0.09985735, 'a149_40' : 0.44222539, 'a39_0' : 0.45649073000000007})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00149031, 'a399_150' : 0.08941878, 'a149_40' : 0.46199702, 'a39_0' : 0.44709389})
				else:
					ast ~= choice({'a700_400' : 0.00120337, 'a399_150' : 0.08423586, 'a149_40' : 0.433213, 'a39_0' : 0.48134776999999995})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.01136364, 'a399_150' : 0.10227273, 'a149_40' : 0.44318182, 'a39_0' : 0.44318181})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.01315789, 'a399_150' : 0.09210526, 'a149_40' : 0.47368421, 'a39_0' : 0.42105264})
				else:
					ast ~= choice({'a700_400' : 0.02777778, 'a399_150' : 0.11111111, 'a149_40' : 0.36111111, 'a39_0' : 0.5})
else:
	if(RHepatitis == 'present'):
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00199601, 'a399_150' : 0.0998004, 'a149_40' : 0.45908184, 'a39_0' : 0.43912175})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00212314, 'a399_150' : 0.08492569, 'a149_40' : 0.48832272, 'a39_0' : 0.42462845000000005})
				else:
					ast ~= choice({'a700_400' : 0.00191939, 'a399_150' : 0.07677543, 'a149_40' : 0.46065259, 'a39_0' : 0.46065259000000003})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00199601, 'a399_150' : 0.0998004, 'a149_40' : 0.47904192, 'a39_0' : 0.41916167000000004})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00249377, 'a399_150' : 0.07481297, 'a149_40' : 0.54862842, 'a39_0' : 0.37406483999999995})
				else:
					ast ~= choice({'a700_400' : 0.00243309, 'a399_150' : 0.0729927, 'a149_40' : 0.51094891, 'a39_0' : 0.4136253000000001})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00166389, 'a399_150' : 0.09983361, 'a149_40' : 0.49916805, 'a39_0' : 0.39933445})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00175131, 'a399_150' : 0.08756567, 'a149_40' : 0.52539405, 'a39_0' : 0.38528897000000006})
				else:
					ast ~= choice({'a700_400' : 0.00142653, 'a399_150' : 0.08559201, 'a149_40' : 0.49928673, 'a39_0' : 0.41369473})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00140647, 'a399_150' : 0.11251758, 'a149_40' : 0.52039381, 'a39_0' : 0.3656821400000001})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00172117, 'a399_150' : 0.10327022, 'a149_40' : 0.58519794, 'a39_0' : 0.30981066999999995})
				else:
					ast ~= choice({'a700_400' : 0.00584795, 'a399_150' : 0.11695906, 'a149_40' : 0.64327486, 'a39_0' : 0.23391812999999995})
	else:
		if(THepatitis == 'present'):
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00181488, 'a399_150' : 0.0907441, 'a149_40' : 0.47186933, 'a39_0' : 0.43557169000000007})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00172117, 'a399_150' : 0.06884682, 'a149_40' : 0.48192771, 'a39_0' : 0.44750429999999997})
				else:
					ast ~= choice({'a700_400' : 0.00133156, 'a399_150' : 0.0665779, 'a149_40' : 0.43941411, 'a39_0' : 0.49267642999999994})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00126422, 'a399_150' : 0.08849558, 'a149_40' : 0.4551201, 'a39_0' : 0.4551201})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00149031, 'a399_150' : 0.07451565, 'a149_40' : 0.49180328, 'a39_0' : 0.43219076})
				else:
					ast ~= choice({'a700_400' : 0.00369004, 'a399_150' : 0.07380074, 'a149_40' : 0.36900369, 'a39_0' : 0.55350553})
		else:
			if(Steatosis == 'present'):
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.00133156, 'a399_150' : 0.09320905, 'a149_40' : 0.45272969, 'a39_0' : 0.4527297})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.00116144, 'a399_150' : 0.08130081, 'a149_40' : 0.48780488, 'a39_0' : 0.42973287000000004})
				else:
					ast ~= choice({'a700_400' : 0.00149031, 'a399_150' : 0.07451565, 'a149_40' : 0.43219076, 'a39_0' : 0.49180327999999995})
			else:
				if(Cirrhosis == 'decompensate'):
					ast ~= choice({'a700_400' : 0.01724138, 'a399_150' : 0.13793103, 'a149_40' : 0.5, 'a39_0' : 0.34482758999999996})
				elif(Cirrhosis == 'compensate'):
					ast ~= choice({'a700_400' : 0.03225806, 'a399_150' : 0.06451613, 'a149_40' : 0.67741936, 'a39_0' : 0.22580645})
				else:
					ast ~= choice({'a700_400' : 0.01075269, 'a399_150' : 0.22580645, 'a149_40' : 0.46774194, 'a39_0' : 0.29569892})


if (Hyperbilirubinemia == 'present'):
	if (PBC == 'present'):
		if (Cirrhosis == 'decompensate'):
			if (gallstones == 'present'):
				if (ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.04347826, 'a19_7' : 0.2173913, 'a6_2' : 0.34782609, 'a1_0' : 0.39130434999999997})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.07407407, 'a19_7' : 0.22222222, 'a6_2' : 0.33333333, 'a1_0' : 0.37037038})
				else:
					bilirubin ~= choice({'a88_20' : 0.07894737, 'a19_7' : 0.23684211, 'a6_2' : 0.34210526, 'a1_0' : 0.34210525999999997})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.01923077, 'a19_7' : 0.11538462, 'a6_2' : 0.36538462, 'a1_0' : 0.49999999000000006})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.01818182, 'a19_7' : 0.11818182, 'a6_2' : 0.38181818, 'a1_0' : 0.48181818})
				else:
					bilirubin ~= choice({'a88_20' : 0.02189781, 'a19_7' : 0.12408759, 'a6_2' : 0.41605839, 'a1_0' : 0.43795621000000007})
		elif(Cirrhosis == 'compensate'):
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.03571429, 'a19_7' : 0.16071429, 'a6_2' : 0.39285714, 'a1_0' : 0.41071428})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.05882353, 'a19_7' : 0.20588235, 'a6_2' : 0.38235294, 'a1_0' : 0.35294117999999997})
				else:
					bilirubin ~= choice({'a88_20' : 0.07692308, 'a19_7' : 0.23076923, 'a6_2' : 0.35897436, 'a1_0' : 0.3333333300000001})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.02020202, 'a19_7' : 0.11111111, 'a6_2' : 0.34343434, 'a1_0' : 0.5252525299999999})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.01941748, 'a19_7' : 0.10679612, 'a6_2' : 0.34951456, 'a1_0' : 0.52427184})
				else:
					bilirubin ~= choice({'a88_20' : 0.02362205, 'a19_7' : 0.11811024, 'a6_2' : 0.37795276, 'a1_0' : 0.48031495})
		else:
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.04225352, 'a19_7' : 0.15492958, 'a6_2' : 0.36619718, 'a1_0' : 0.43661971999999993})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.06, 'a19_7' : 0.2, 'a6_2' : 0.36, 'a1_0' : 0.38})
				else:
					bilirubin ~= choice({'a88_20' : 0.07575758, 'a19_7' : 0.22727273, 'a6_2' : 0.36363636, 'a1_0' : 0.33333333})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.03030303, 'a19_7' : 0.12121212, 'a6_2' : 0.35606061, 'a1_0' : 0.49242423999999996})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.02158273, 'a19_7' : 0.11510791, 'a6_2' : 0.37410072, 'a1_0' : 0.48920864})
				else:
					bilirubin ~= choice({'a88_20' : 0.02061856, 'a19_7' : 0.12371134, 'a6_2' : 0.40721649, 'a1_0' : 0.44845361000000006})
	else:
		if(Cirrhosis == 'decompensate'):
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.01449275, 'a19_7' : 0.11594203, 'a6_2' : 0.39130435, 'a1_0' : 0.47826086999999995})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.00398406, 'a19_7' : 0.11952191, 'a6_2' : 0.35856574, 'a1_0' : 0.51792829})
				else:
					bilirubin ~= choice({'a88_20' : 0.00662252, 'a19_7' : 0.13245033, 'a6_2' : 0.26490066, 'a1_0' : 0.59602649})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.00151286, 'a19_7' : 0.04538578, 'a6_2' : 0.34795764, 'a1_0' : 0.60514372})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.00144718, 'a19_7' : 0.04341534, 'a6_2' : 0.37626628, 'a1_0' : 0.5788712})
				else:
					bilirubin ~= choice({'a88_20' : 0.00114811, 'a19_7' : 0.05740528, 'a6_2' : 0.44776119, 'a1_0' : 0.49368542000000004})
		elif(Cirrhosis == 'compensate'):
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.00311526, 'a19_7' : 0.03115265, 'a6_2' : 0.43613707, 'a1_0' : 0.5295950199999999})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.00892857, 'a19_7' : 0.00892857, 'a6_2' : 0.44642857, 'a1_0' : 0.53571429})
				else:
					bilirubin ~= choice({'a88_20' : 0.01388889, 'a19_7' : 0.01388889, 'a6_2' : 0.41666667, 'a1_0' : 0.55555555})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.00178253, 'a19_7' : 0.01782531, 'a6_2' : 0.28520499, 'a1_0' : 0.6951871700000001})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.00174825, 'a19_7' : 0.00174825, 'a6_2' : 0.2972028, 'a1_0' : 0.6993007})
				else:
					bilirubin ~= choice({'a88_20' : 0.00144509, 'a19_7' : 0.00144509, 'a6_2' : 0.36127168, 'a1_0' : 0.6358381399999999})
		else:
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.00255102, 'a19_7' : 0.00255102, 'a6_2' : 0.33163265, 'a1_0' : 0.6632653100000001})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.0049505, 'a19_7' : 0.0049505, 'a6_2' : 0.2970297, 'a1_0' : 0.6930693})
				else:
					bilirubin ~= choice({'a88_20' : 0.07692308, 'a19_7' : 0.07692308, 'a6_2' : 0.07692308, 'a1_0' : 0.76923076})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.00146843, 'a19_7' : 0.01468429, 'a6_2' : 0.30837004, 'a1_0' : 0.67547724})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.00127877, 'a19_7' : 0.00127877, 'a6_2' : 0.33248082, 'a1_0' : 0.66496164})
				else:
					bilirubin ~= choice({'a88_20' : 0.00181159, 'a19_7' : 0.00181159, 'a6_2' : 0.54347827, 'a1_0' : 0.45289855})
else:
	if(PBC == 'present'):
		if(Cirrhosis == 'decompensate'):
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.04081633, 'a19_7' : 0.14285714, 'a6_2' : 0.40816327, 'a1_0' : 0.40816326})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.06818182, 'a19_7' : 0.20454545, 'a6_2' : 0.36363636, 'a1_0' : 0.36363637000000004})
				else:
					bilirubin ~= choice({'a88_20' : 0.07692308, 'a19_7' : 0.23076923, 'a6_2' : 0.35384615, 'a1_0' : 0.33846153999999995})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.03030303, 'a19_7' : 0.14393939, 'a6_2' : 0.34090909, 'a1_0' : 0.48484848999999997})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.02173913, 'a19_7' : 0.13768116, 'a6_2' : 0.35507246, 'a1_0' : 0.48550725000000006})
				else:
					bilirubin ~= choice({'a88_20' : 0.02590674, 'a19_7' : 0.15025907, 'a6_2' : 0.38341969, 'a1_0' : 0.44041450000000004})
		elif(Cirrhosis == 'compensate'):
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.04255319, 'a19_7' : 0.17021277, 'a6_2' : 0.37234043, 'a1_0' : 0.41489361})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.05084746, 'a19_7' : 0.20338983, 'a6_2' : 0.37288136, 'a1_0' : 0.37288135})
				else:
					bilirubin ~= choice({'a88_20' : 0.07042254, 'a19_7' : 0.22535211, 'a6_2' : 0.36619718, 'a1_0' : 0.33802816999999996})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.032, 'a19_7' : 0.136, 'a6_2' : 0.312, 'a1_0' : 0.52})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.0234375, 'a19_7' : 0.1328125, 'a6_2' : 0.3125, 'a1_0' : 0.53125})
				else:
					bilirubin ~= choice({'a88_20' : 0.02824859, 'a19_7' : 0.14124294, 'a6_2' : 0.33898305, 'a1_0' : 0.49152542})
		else:
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.04477612, 'a19_7' : 0.1641791, 'a6_2' : 0.34328358, 'a1_0' : 0.44776119999999997})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.05555556, 'a19_7' : 0.19444444, 'a6_2' : 0.35185185, 'a1_0' : 0.3981481499999999})
				else:
					bilirubin ~= choice({'a88_20' : 0.08536585, 'a19_7' : 0.24390244, 'a6_2' : 0.36585366, 'a1_0' : 0.3048780499999999})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.03910615, 'a19_7' : 0.15083799, 'a6_2' : 0.32960894, 'a1_0' : 0.48044692})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.02926829, 'a19_7' : 0.14634146, 'a6_2' : 0.33658537, 'a1_0' : 0.48780488})
				else:
					bilirubin ~= choice({'a88_20' : 0.03030303, 'a19_7' : 0.16666667, 'a6_2' : 0.38888889, 'a1_0' : 0.41414141000000004})
	else:
		if(Cirrhosis == 'decompensate'):
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.02830189, 'a19_7' : 0.16037736, 'a6_2' : 0.36792453, 'a1_0' : 0.44339622})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.01818182, 'a19_7' : 0.16363636, 'a6_2' : 0.34545455, 'a1_0' : 0.47272727000000003})
				else:
					bilirubin ~= choice({'a88_20' : 0.01923077, 'a19_7' : 0.19230769, 'a6_2' : 0.01923077, 'a1_0' : 0.76923077})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.01449275, 'a19_7' : 0.07246377, 'a6_2' : 0.24637681, 'a1_0' : 0.66666667})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.01282051, 'a19_7' : 0.07692308, 'a6_2' : 0.26923077, 'a1_0' : 0.64102564})
				else:
					bilirubin ~= choice({'a88_20' : 0.01886792, 'a19_7' : 0.13207547, 'a6_2' : 0.43396226, 'a1_0' : 0.41509434999999995})
		elif(Cirrhosis == 'compensate'):
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.00321543, 'a19_7' : 0.09646302, 'a6_2' : 0.41800643, 'a1_0' : 0.48231512})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.00621118, 'a19_7' : 0.0621118, 'a6_2' : 0.43478261, 'a1_0' : 0.49689441})
				else:
					bilirubin ~= choice({'a88_20' : 0.04545455, 'a19_7' : 0.04545455, 'a6_2' : 0.45454545, 'a1_0' : 0.45454545})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.01818182, 'a19_7' : 0.01818182, 'a6_2' : 0.14545455, 'a1_0' : 0.81818181})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.01694915, 'a19_7' : 0.01694915, 'a6_2' : 0.10169492, 'a1_0' : 0.86440678})
				else:
					bilirubin ~= choice({'a88_20' : 0.03448276, 'a19_7' : 0.03448276, 'a6_2' : 0.10344828, 'a1_0' : 0.8275862})
		else:
			if(gallstones == 'present'):
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.00980392, 'a19_7' : 0.00980392, 'a6_2' : 0.19607843, 'a1_0' : 0.78431373})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.07692308, 'a19_7' : 0.07692308, 'a6_2' : 0.07692308, 'a1_0' : 0.76923076})
				else:
					bilirubin ~= choice({'a88_20' : 0.01587302, 'a19_7' : 0.01587302, 'a6_2' : 0.01587302, 'a1_0' : 0.95238094})
			else:
				if(ChHepatitis == 'active'):
					bilirubin ~= choice({'a88_20' : 0.01234568, 'a19_7' : 0.02469136, 'a6_2' : 0.14814815, 'a1_0' : 0.81481481})
				elif(ChHepatitis == 'persistent'):
					bilirubin ~= choice({'a88_20' : 0.00284091, 'a19_7' : 0.00284091, 'a6_2' : 0.05681818, 'a1_0' : 0.9375})
				else:
					bilirubin ~= choice({'a88_20' : 0.00070872, 'a19_7' : 0.02126152, 'a6_2' : 0.13465627, 'a1_0' : 0.8433734900000001})


if (Cirrhosis == 'decompensate'):
	if (PBC == 'present'):
		carcinoma ~= choice({'present' : 0.3636364, 'absent' : 0.6363635999999999})
	else:
		carcinoma ~= choice({'present' : 0.3, 'absent' : 0.7})
elif(Cirrhosis == 'compensate'):
	if(PBC == 'present'):
		carcinoma ~= choice({'present' : 0.2727273, 'absent' : 0.7272727})
	else:
		carcinoma ~= choice({'present' : 0.2, 'absent' : 0.8})
else:
	if(PBC == 'present'):
		carcinoma ~= choice({'present' : 0.1, 'absent' : 0.9})
	else:
		carcinoma ~= choice({'present' : 0.01, 'absent' : 0.99})


if (Cirrhosis == 'decompensate'):
	edema ~= choice({'present' : 0.3448276, 'absent' : 0.6551724})
elif(Cirrhosis == 'compensate'):
	edema ~= choice({'present' : 0.06451613, 'absent' : 0.93548387})
else:
	edema ~= choice({'present' : 0.1311475, 'absent' : 0.8688525})


if (Cirrhosis == 'decompensate'):
	edge ~= choice({'present' : 0.7586207, 'absent' : 0.24137929999999996})
elif(Cirrhosis == 'compensate'):
	edge ~= choice({'present' : 0.4516129, 'absent' : 0.5483871})
else:
	edge ~= choice({'present' : 0.2344262, 'absent' : 0.7655738})


if (Cirrhosis == 'decompensate'):
	if (PBC == 'present'):
		encephalopathy ~= choice({'present' : 0.05325444, 'absent' : 0.94674556})
	else:
		encephalopathy ~= choice({'present' : 0.05172414, 'absent' : 0.94827586})
elif(Cirrhosis == 'compensate'):
	if(PBC == 'present'):
		encephalopathy ~= choice({'present' : 0.04891304, 'absent' : 0.95108696})
	else:
		encephalopathy ~= choice({'present' : 0.00321543, 'absent' : 0.99678457})
else:
	if(PBC == 'present'):
		encephalopathy ~= choice({'present' : 0.05357143, 'absent' : 0.94642857})
	else:
		encephalopathy ~= choice({'present' : 0.01515152, 'absent' : 0.98484848})


if (ChHepatitis == 'active'):
	if (Cirrhosis == 'decompensate'):
		if (THepatitis == 'present'):
			if (Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.01754386, 'a109_70' : 0.84210526, 'a69_0' : 0.14035088000000007})
			else:
				inr ~= choice({'a200_110' : 0.01298701, 'a109_70' : 0.81818182, 'a69_0' : 0.16883117000000003})
		else:
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.02150538, 'a109_70' : 0.8172043, 'a69_0' : 0.16129032})
			else:
				inr ~= choice({'a200_110' : 0.01666667, 'a109_70' : 0.8, 'a69_0' : 0.18333332999999996})
	elif(Cirrhosis == 'compensate'):
		if(THepatitis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.025, 'a109_70' : 0.85, 'a69_0' : 0.125})
			else:
				inr ~= choice({'a200_110' : 0.01333333, 'a109_70' : 0.86666667, 'a69_0' : 0.12})
		else:
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.02409639, 'a109_70' : 0.86746988, 'a69_0' : 0.10843373})
			else:
				inr ~= choice({'a200_110' : 0.01960784, 'a109_70' : 0.85294118, 'a69_0' : 0.12745098})
	else:
		if(THepatitis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.02197802, 'a109_70' : 0.89010989, 'a69_0' : 0.08791209})
			else:
				inr ~= choice({'a200_110' : 0.01923077, 'a109_70' : 0.90384615, 'a69_0' : 0.07692308000000003})
		else:
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.024, 'a109_70' : 0.904, 'a69_0' : 0.07199999999999995})
			else:
				inr ~= choice({'a200_110' : 0.02197802, 'a109_70' : 0.9010989, 'a69_0' : 0.07692307999999992})
elif(ChHepatitis == 'persistent'):
	if(Cirrhosis == 'decompensate'):
		if(THepatitis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.03030303, 'a109_70' : 0.84848485, 'a69_0' : 0.12121212000000003})
			else:
				inr ~= choice({'a200_110' : 0.03225806, 'a109_70' : 0.79032259, 'a69_0' : 0.17741934999999998})
		else:
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.02898551, 'a109_70' : 0.79710145, 'a69_0' : 0.17391303999999996})
			else:
				inr ~= choice({'a200_110' : 0.02469136, 'a109_70' : 0.75308642, 'a69_0' : 0.22222221999999991})
	elif(Cirrhosis == 'compensate'):
		if(THepatitis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.03508772, 'a109_70' : 0.8245614, 'a69_0' : 0.14035087999999996})
			else:
				inr ~= choice({'a200_110' : 0.04081633, 'a109_70' : 0.83673469, 'a69_0' : 0.12244898000000004})
		else:
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.03571429, 'a109_70' : 0.85714285, 'a69_0' : 0.10714285999999995})
			else:
				inr ~= choice({'a200_110' : 0.03333333, 'a109_70' : 0.83333334, 'a69_0' : 0.13333332999999992})
	else:
		if(THepatitis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.05084746, 'a109_70' : 0.88135593, 'a69_0' : 0.06779661000000003})
			else:
				inr ~= choice({'a200_110' : 0.05, 'a109_70' : 0.9, 'a69_0' : 0.04999999999999993})
		else:
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.05333333, 'a109_70' : 0.90666667, 'a69_0' : 0.040000000000000036})
			else:
				inr ~= choice({'a200_110' : 0.08333333, 'a109_70' : 0.88888889, 'a69_0' : 0.027777779999999974})
else:
	if(Cirrhosis == 'decompensate'):
		if(THepatitis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.03448276, 'a109_70' : 0.81034483, 'a69_0' : 0.15517241000000004})
			else:
				inr ~= choice({'a200_110' : 0.01428571, 'a109_70' : 0.75714286, 'a69_0' : 0.22857143000000002})
		else:
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.02173913, 'a109_70' : 0.75, 'a69_0' : 0.22826086999999995})
			else:
				inr ~= choice({'a200_110' : 0.00172117, 'a109_70' : 0.60240964, 'a69_0' : 0.39586919})
	elif(Cirrhosis == 'compensate'):
		if(THepatitis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.01785714, 'a109_70' : 0.76785715, 'a69_0' : 0.21428570999999996})
			else:
				inr ~= choice({'a200_110' : 0.01785714, 'a109_70' : 0.78571429, 'a69_0' : 0.1964285699999999})
		else:
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.02816901, 'a109_70' : 0.80281691, 'a69_0' : 0.16901407999999996})
			else:
				inr ~= choice({'a200_110' : 0.00321543, 'a109_70' : 0.67524116, 'a69_0' : 0.32154341})
	else:
		if(THepatitis == 'present'):
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.03508772, 'a109_70' : 0.84210526, 'a69_0' : 0.12280701999999999})
			else:
				inr ~= choice({'a200_110' : 0.03703704, 'a109_70' : 0.92592592, 'a69_0' : 0.03703704000000008})
		else:
			if(Hyperbilirubinemia == 'present'):
				inr ~= choice({'a200_110' : 0.05357143, 'a109_70' : 0.89285714, 'a69_0' : 0.053571429999999975})
			else:
				inr ~= choice({'a200_110' : 0.065, 'a109_70' : 0.875, 'a69_0' : 0.06000000000000005})


if (Cirrhosis == 'decompensate'):
	irregular_liver ~= choice({'present' : 0.6034483, 'absent' : 0.39655169999999995})
elif(Cirrhosis == 'compensate'):
	irregular_liver ~= choice({'present' : 0.3548387, 'absent' : 0.6451613})
else:
	irregular_liver ~= choice({'present' : 0.1065574, 'absent' : 0.8934426})


if (bilirubin == 'a88_20'):
	itching ~= choice({'present' : 0.875, 'absent' : 0.125})
elif(bilirubin == 'a19_7'):
	itching ~= choice({'present' : 0.6865672, 'absent' : 0.31343279999999996})
elif(bilirubin == 'a6_2'):
	itching ~= choice({'present' : 0.5477387, 'absent' : 0.4522613})
else:
	itching ~= choice({'present' : 0.3333333, 'absent' : 0.6666667})


if (bilirubin == 'a88_20'):
	jaundice ~= choice({'present' : 0.75, 'absent' : 0.25})
elif(bilirubin == 'a19_7'):
	jaundice ~= choice({'present' : 0.5671642, 'absent' : 0.4328358})
elif(bilirubin == 'a6_2'):
	jaundice ~= choice({'present' : 0.3467337, 'absent' : 0.6532663000000001})
else:
	jaundice ~= choice({'present' : 0.1942446, 'absent' : 0.8057554})


if (Cirrhosis == 'decompensate'):
	palms ~= choice({'present' : 0.5, 'absent' : 0.5})
elif(Cirrhosis == 'compensate'):
	palms ~= choice({'present' : 0.2903226, 'absent' : 0.7096774})
else:
	palms ~= choice({'present' : 0.1409836, 'absent' : 0.8590164})


if (RHepatitis == 'present'):
	if (THepatitis == 'present'):
		if (Cirrhosis == 'decompensate'):
			if (ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.04166667, 'a699_240' : 0.29166667, 'a239_0' : 0.66666666})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.04347826, 'a699_240' : 0.30434783, 'a239_0' : 0.65217391})
			else:
				phosphatase ~= choice({'a4000_700' : 0.04166667, 'a699_240' : 0.33333333, 'a239_0' : 0.625})
		elif(Cirrhosis == 'compensate'):
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.03773585, 'a699_240' : 0.26415094, 'a239_0' : 0.69811321})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.025, 'a699_240' : 0.25, 'a239_0' : 0.725})
			else:
				phosphatase ~= choice({'a4000_700' : 0.02702703, 'a699_240' : 0.27027027, 'a239_0' : 0.7027027})
		else:
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.03571429, 'a699_240' : 0.21428571, 'a239_0' : 0.75})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.02272727, 'a699_240' : 0.20454545, 'a239_0' : 0.77272728})
			else:
				phosphatase ~= choice({'a4000_700' : 0.02325581, 'a699_240' : 0.18604651, 'a239_0' : 0.7906976800000001})
	else:
		if(Cirrhosis == 'decompensate'):
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.02898551, 'a699_240' : 0.28985507, 'a239_0' : 0.68115942})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.03389831, 'a699_240' : 0.3220339, 'a239_0' : 0.64406779})
			else:
				phosphatase ~= choice({'a4000_700' : 0.04545455, 'a699_240' : 0.37878788, 'a239_0' : 0.5757575699999999})
		elif(Cirrhosis == 'compensate'):
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.02985075, 'a699_240' : 0.29850746, 'a239_0' : 0.67164179})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.02040816, 'a699_240' : 0.28571429, 'a239_0' : 0.6938775500000001})
			else:
				phosphatase ~= choice({'a4000_700' : 0.02083333, 'a699_240' : 0.3125, 'a239_0' : 0.66666667})
		else:
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.02597403, 'a699_240' : 0.24675325, 'a239_0' : 0.72727272})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.015625, 'a699_240' : 0.234375, 'a239_0' : 0.75})
			else:
				phosphatase ~= choice({'a4000_700' : 0.00584795, 'a699_240' : 0.23391813, 'a239_0' : 0.76023392})
else:
	if(THepatitis == 'present'):
		if(Cirrhosis == 'decompensate'):
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.04761905, 'a699_240' : 0.28571429, 'a239_0' : 0.66666666})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.04918033, 'a699_240' : 0.31147541, 'a239_0' : 0.63934426})
			else:
				phosphatase ~= choice({'a4000_700' : 0.05555556, 'a699_240' : 0.34722222, 'a239_0' : 0.5972222199999999})
		elif(Cirrhosis == 'compensate'):
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.04166667, 'a699_240' : 0.27777778, 'a239_0' : 0.68055555})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.03703704, 'a699_240' : 0.25925926, 'a239_0' : 0.7037036999999999})
			else:
				phosphatase ~= choice({'a4000_700' : 0.05357143, 'a699_240' : 0.26785714, 'a239_0' : 0.67857143})
		else:
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.04651163, 'a699_240' : 0.22093023, 'a239_0' : 0.73255814})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.04054054, 'a699_240' : 0.2027027, 'a239_0' : 0.75675676})
			else:
				phosphatase ~= choice({'a4000_700' : 0.07407407, 'a699_240' : 0.14814815, 'a239_0' : 0.77777778})
	else:
		if(Cirrhosis == 'decompensate'):
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.04597701, 'a699_240' : 0.29885057, 'a239_0' : 0.65517242})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.04494382, 'a699_240' : 0.33707865, 'a239_0' : 0.61797753})
			else:
				phosphatase ~= choice({'a4000_700' : 0.06896552, 'a699_240' : 0.48275862, 'a239_0' : 0.44827585999999997})
		elif(Cirrhosis == 'compensate'):
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.04494382, 'a699_240' : 0.33707865, 'a239_0' : 0.61797753})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.03896104, 'a699_240' : 0.31168831, 'a239_0' : 0.6493506499999999})
			else:
				phosphatase ~= choice({'a4000_700' : 0.06451613, 'a699_240' : 0.38709677, 'a239_0' : 0.5483871})
		else:
			if(ChHepatitis == 'active'):
				phosphatase ~= choice({'a4000_700' : 0.03296703, 'a699_240' : 0.21978022, 'a239_0' : 0.74725275})
			elif(ChHepatitis == 'persistent'):
				phosphatase ~= choice({'a4000_700' : 0.02777778, 'a699_240' : 0.19444444, 'a239_0' : 0.77777778})
			else:
				phosphatase ~= choice({'a4000_700' : 0.2118451, 'a699_240' : 0.3394077, 'a239_0' : 0.4487472})


if (Cirrhosis == 'decompensate'):
	if (PBC == 'present'):
		platelet ~= choice({'a597_300' : 0.06547619, 'a299_150' : 0.63690476, 'a149_100' : 0.17857143, 'a99_0' : 0.11904762000000002})
	else:
		platelet ~= choice({'a597_300' : 0.06896552, 'a299_150' : 0.46551724, 'a149_100' : 0.27586207, 'a99_0' : 0.18965516999999998})
elif(Cirrhosis == 'compensate'):
	if(PBC == 'present'):
		platelet ~= choice({'a597_300' : 0.06557377, 'a299_150' : 0.63934426, 'a149_100' : 0.17486339, 'a99_0' : 0.12021857999999996})
	else:
		platelet ~= choice({'a597_300' : 0.06451613, 'a299_150' : 0.64516129, 'a149_100' : 0.16129032, 'a99_0' : 0.12903226})
else:
	if(PBC == 'present'):
		platelet ~= choice({'a597_300' : 0.06428571, 'a299_150' : 0.67142857, 'a149_100' : 0.15714286, 'a99_0' : 0.10714285999999995})
	else:
		platelet ~= choice({'a597_300' : 0.09393939, 'a299_150' : 0.73636364, 'a149_100' : 0.13939394, 'a99_0' : 0.03030303000000001})


if (Cirrhosis == 'decompensate'):
	proteins ~= choice({'a10_6' : 0.99827883, 'a5_2' : 0.0017211700000000496})
elif(Cirrhosis == 'compensate'):
	proteins ~= choice({'a10_6' : 0.99678457, 'a5_2' : 0.003215430000000019})
else:
	proteins ~= choice({'a10_6' : 0.98032787, 'a5_2' : 0.01967213000000001})


if (bilirubin == 'a88_20'):
	skin ~= choice({'present' : 0.99378882, 'absent' : 0.006211179999999983})
elif(bilirubin == 'a19_7'):
	skin ~= choice({'present' : 0.8955224, 'absent' : 0.10447759999999995})
elif(bilirubin == 'a6_2'):
	skin ~= choice({'present' : 0.7035176, 'absent' : 0.29648240000000003})
else:
	skin ~= choice({'present' : 0.1822542, 'absent' : 0.8177458})


if (Cirrhosis == 'decompensate'):
	spiders ~= choice({'present' : 0.6034483, 'absent' : 0.39655169999999995})
elif(Cirrhosis == 'compensate'):
	spiders ~= choice({'present' : 0.483871, 'absent' : 0.5161290000000001})
else:
	spiders ~= choice({'present' : 0.1836066, 'absent' : 0.8163933999999999})


if (Cirrhosis == 'decompensate'):
	if (RHepatitis == 'present'):
		if (THepatitis == 'present'):
			spleen ~= choice({'present' : 0.3235294, 'absent' : 0.6764706})
		else:
			spleen ~= choice({'present' : 0.3703704, 'absent' : 0.6296296})
	else:
		if(THepatitis == 'present'):
			spleen ~= choice({'present' : 0.3623188, 'absent' : 0.6376812000000001})
		else:
			spleen ~= choice({'present' : 0.4827586, 'absent' : 0.5172414000000001})
elif(Cirrhosis == 'compensate'):
	if(RHepatitis == 'present'):
		if(THepatitis == 'present'):
			spleen ~= choice({'present' : 0.3023256, 'absent' : 0.6976743999999999})
		else:
			spleen ~= choice({'present' : 0.2444444, 'absent' : 0.7555556})
	else:
		if(THepatitis == 'present'):
			spleen ~= choice({'present' : 0.2156863, 'absent' : 0.7843137})
		else:
			spleen ~= choice({'present' : 0.2580645, 'absent' : 0.7419355000000001})
else:
	if(RHepatitis == 'present'):
		if(THepatitis == 'present'):
			spleen ~= choice({'present' : 0.1621622, 'absent' : 0.8378378})
		else:
			spleen ~= choice({'present' : 0.1176471, 'absent' : 0.8823529})
	else:
		if(THepatitis == 'present'):
			spleen ~= choice({'present' : 0.1111111, 'absent' : 0.8888889})
		else:
			spleen ~= choice({'present' : 0.1007067, 'absent' : 0.8992933})


if (encephalopathy == 'present'):
	urea ~= choice({'a165_50' : 0.2173913, 'a49_40' : 0.1304348, 'a39_0' : 0.6521739})
else:
	urea ~= choice({'a165_50' : 0.03550296, 'a49_40' : 0.06508876, 'a39_0' : 0.8994082800000001})


if (proteins == 'a10_6'):
	ascites ~= choice({'present' : 0.1280932, 'absent' : 0.8719068})
else:
	ascites ~= choice({'present' : 0.5833333, 'absent' : 0.41666669999999995})


if (platelet == 'a597_300'):
	if (inr == 'a200_110'):
		bleeding ~= choice({'present' : 0.1428571, 'absent' : 0.8571429})
	elif(inr == 'a109_70'):
		bleeding ~= choice({'present' : 0.106383, 'absent' : 0.893617})
	else:
		bleeding ~= choice({'present' : 0.09090909, 'absent' : 0.90909091})
elif(platelet == 'a299_150'):
	if(inr == 'a200_110'):
		bleeding ~= choice({'present' : 0.1304348, 'absent' : 0.8695652})
	elif(inr == 'a109_70'):
		bleeding ~= choice({'present' : 0.1373494, 'absent' : 0.8626506})
	else:
		bleeding ~= choice({'present' : 0.425, 'absent' : 0.575})
elif(platelet == 'a149_100'):
	if(inr == 'a200_110'):
		bleeding ~= choice({'present' : 0.2, 'absent' : 0.8})
	elif(inr == 'a109_70'):
		bleeding ~= choice({'present' : 0.1333333, 'absent' : 0.8666667})
	else:
		bleeding ~= choice({'present' : 0.25, 'absent' : 0.75})
else:
	if(inr == 'a200_110'):
		bleeding ~= choice({'present' : 0.5, 'absent' : 0.5})
	elif(inr == 'a109_70'):
		bleeding ~= choice({'present' : 0.255814, 'absent' : 0.744186})
	else:
		bleeding ~= choice({'present' : 0.6666667, 'absent' : 0.33333330000000005})


if (encephalopathy == 'present'):
	consciousness ~= choice({'present' : 0.3043478, 'absent' : 0.6956522})
else:
	consciousness ~= choice({'present' : 0.01627219, 'absent' : 0.98372781})


if (encephalopathy == 'present'):
	density ~= choice({'present' : 0.7391304, 'absent' : 0.26086960000000003})
else:
	density ~= choice({'present' : 0.3772189, 'absent' : 0.6227811})


'''
compiler = SPPL_Compiler(data)
namespace = compiler.execute_module()
model=namespace.model
ChHepatitis = Id('ChHepatitis')
Cirrhosis = Id('Cirrhosis')
ESR = Id('ESR')
Hyperbilirubinemia = Id('Hyperbilirubinemia')
PBC = Id('PBC')
RHepatitis = Id('RHepatitis')
Steatosis = Id('Steatosis')
THepatitis = Id('THepatitis')
age = Id('age')
albumin = Id('albumin')
alcohol = Id('alcohol')
alcoholism = Id('alcoholism')
alt = Id('alt')
ama = Id('ama')
amylase = Id('amylase')
anorexia = Id('anorexia')
ascites = Id('ascites')
ast = Id('ast')
bilirubin = Id('bilirubin')
bleeding = Id('bleeding')
carcinoma = Id('carcinoma')
choledocholithotomy = Id('choledocholithotomy')
cholesterol = Id('cholesterol')
consciousness = Id('consciousness')
density = Id('density')
diabetes = Id('diabetes')
edema = Id('edema')
edge = Id('edge')
encephalopathy = Id('encephalopathy')
fat = Id('fat')
fatigue = Id('fatigue')
fibrosis = Id('fibrosis')
flatulence = Id('flatulence')
gallstones = Id('gallstones')
ggtp = Id('ggtp')
hbc_anti = Id('hbc_anti')
hbeag = Id('hbeag')
hbsag = Id('hbsag')
hbsag_anti = Id('hbsag_anti')
hcv_anti = Id('hcv_anti')
hepatalgia = Id('hepatalgia')
hepatomegaly = Id('hepatomegaly')
hepatotoxic = Id('hepatotoxic')
hospital = Id('hospital')
injections = Id('injections')
inr = Id('inr')
irregular_liver = Id('irregular_liver')
itching = Id('itching')
jaundice = Id('jaundice')
joints = Id('joints')
le_cells = Id('le_cells')
nausea = Id('nausea')
obesity = Id('obesity')
pain = Id('pain')
pain_ruq = Id('pain_ruq')
palms = Id('palms')
phosphatase = Id('phosphatase')
platelet = Id('platelet')
pressure_ruq = Id('pressure_ruq')
proteins = Id('proteins')
sex = Id('sex')
skin = Id('skin')
spiders = Id('spiders')
spleen = Id('spleen')
surgery = Id('surgery')
transfusion = Id('transfusion')
triglycerides = Id('triglycerides')
upper_pain = Id('upper_pain')
urea = Id('urea')
vh_amn = Id('vh_amn')
events = [pressure_ruq << {'absent'},hepatomegaly << {'absent'},le_cells << {'absent'},RHepatitis << {'absent'},hepatalgia << {'absent'},triglycerides << {'a17_4'},bilirubin << {'a1_0'},vh_amn << {'absent'},irregular_liver << {'absent'},PBC << {'present'},ama << {'present'},injections << {'present'},hbsag_anti << {'absent'},fibrosis << {'present'},skin << {'absent'},hbc_anti << {'present'},hepatomegaly << {'absent'},age << {'age51_65'},flatulence << {'absent'},edema << {'absent'},hepatotoxic << {'present'},flatulence << {'present'},choledocholithotomy << {'absent'},Cirrhosis << {'absent'},skin << {'present'},phosphatase << {'a4000_700'},bilirubin << {'a19_7'},palms << {'present'},ama << {'present'},le_cells << {'absent'},Cirrhosis << {'absent'},choledocholithotomy << {'absent'},bleeding << {'absent'},nausea << {'present'},hepatalgia << {'absent'},consciousness << {'absent'},palms << {'present'},irregular_liver << {'absent'},proteins << {'a10_6'},fat << {'present'},hbc_anti << {'present'},ascites << {'absent'},triglycerides << {'a1_0'},carcinoma << {'absent'},alcoholism << {'absent'},Cirrhosis << {'decompensate'},joints << {'present'},edge << {'present'},hbsag << {'absent'},Steatosis << {'absent'},(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a850_200'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a39_0'}) & (bilirubin << {'a88_20'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a39_0'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a39_0'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a19_7'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a149_40'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age65_100'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a850_200'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a149_40'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a149_40'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a19_7'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a149_40'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'})]
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
