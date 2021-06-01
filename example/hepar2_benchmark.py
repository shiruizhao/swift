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


if ((diabetes == 'present')):
	obesity ~= choice({'present' : 0.24, 'absent' : 0.76})
else:
	obesity ~= choice({'present' : 0.06231454, 'absent' : 0.93768546})


sex ~= choice({'female' : 0.5979971,'male' : 0.4020029})


surgery ~= choice({'present' : 0.4234621,'absent' : 0.5765379})


if ((gallstones == 'present')):
	upper_pain ~= choice({'present' : 0.411215, 'absent' : 0.588785})
else:
	upper_pain ~= choice({'present' : 0.3868243, 'absent' : 0.6131757})


vh_amn ~= choice({'present' : 0.1731044,'absent' : 0.8268956})


if ((age == 'age65_100') and (sex == 'female')):
	Hyperbilirubinemia ~= choice({'present' : 0.002849, 'absent' : 0.997151})
elif ((age == 'age65_100') and (sex == 'male')):
	Hyperbilirubinemia ~= choice({'present' : 0.0052356, 'absent' : 0.9947644})
elif ((age == 'age51_65') and (sex == 'female')):
	Hyperbilirubinemia ~= choice({'present' : 0.01129944, 'absent' : 0.98870056})
elif ((age == 'age51_65') and (sex == 'male')):
	Hyperbilirubinemia ~= choice({'present' : 0.0212766, 'absent' : 0.9787234})
elif ((age == 'age31_50') and (sex == 'female')):
	Hyperbilirubinemia ~= choice({'present' : 0.04597701, 'absent' : 0.95402299})
elif ((age == 'age31_50') and (sex == 'male')):
	Hyperbilirubinemia ~= choice({'present' : 0.07692308, 'absent' : 0.92307692})
elif ((age == 'age0_30') and (sex == 'female')):
	Hyperbilirubinemia ~= choice({'present' : 0.21875, 'absent' : 0.78125})
else:
	Hyperbilirubinemia ~= choice({'present' : 0.453125, 'absent' : 0.546875})


if ((sex == 'female') and (age == 'age65_100')):
	PBC ~= choice({'present' : 0.6571429, 'absent' : 0.3428571})
elif ((sex == 'female') and (age == 'age51_65')):
	PBC ~= choice({'present' : 0.700565, 'absent' : 0.299435})
elif ((sex == 'female') and (age == 'age31_50')):
	PBC ~= choice({'present' : 0.6149425, 'absent' : 0.3850575})
elif ((sex == 'female') and (age == 'age0_30')):
	PBC ~= choice({'present' : 0.125, 'absent' : 0.875})
elif ((sex == 'male') and (age == 'age65_100')):
	PBC ~= choice({'present' : 0.3684211, 'absent' : 0.6315789})
elif ((sex == 'male') and (age == 'age51_65')):
	PBC ~= choice({'present' : 0.08510638, 'absent' : 0.91489362})
elif ((sex == 'male') and (age == 'age31_50')):
	PBC ~= choice({'present' : 0.06730769, 'absent' : 0.93269231})
else:
	PBC ~= choice({'present' : 0.00156006, 'absent' : 0.99843994})


if ((hepatotoxic == 'present')):
	RHepatitis ~= choice({'present' : 0.01754386, 'absent' : 0.98245614})
else:
	RHepatitis ~= choice({'present' : 0.02492212, 'absent' : 0.97507788})


if ((obesity == 'present') and (alcoholism == 'present')):
	Steatosis ~= choice({'present' : 0.3636364, 'absent' : 0.6363636})
elif ((obesity == 'present') and (alcoholism == 'absent')):
	Steatosis ~= choice({'present' : 0.1891892, 'absent' : 0.8108108})
elif ((obesity == 'absent') and (alcoholism == 'present')):
	Steatosis ~= choice({'present' : 0.2380952, 'absent' : 0.7619048})
else:
	Steatosis ~= choice({'present' : 0.06349206, 'absent' : 0.93650794})


if ((hepatotoxic == 'present') and (alcoholism == 'present')):
	THepatitis ~= choice({'present' : 0.2, 'absent' : 0.8})
elif ((hepatotoxic == 'present') and (alcoholism == 'absent')):
	THepatitis ~= choice({'present' : 0.00191939, 'absent' : 0.99808061})
elif ((hepatotoxic == 'absent') and (alcoholism == 'present')):
	THepatitis ~= choice({'present' : 0.08888889, 'absent' : 0.91111111})
else:
	THepatitis ~= choice({'present' : 0.0326087, 'absent' : 0.9673913})


if ((PBC == 'present')):
	ama ~= choice({'present' : 0.5678571, 'absent' : 0.4321429})
else:
	ama ~= choice({'present' : 0.01193317, 'absent' : 0.98806683})


if ((gallstones == 'present')):
	amylase ~= choice({'a1400_500' : 0.01869159, 'a499_300' : 0.04672897, 'a299_0' : 0.93457944})
else:
	amylase ~= choice({'a1400_500' : 0.01013514, 'a499_300' : 0.01689189, 'a299_0' : 0.97297297})


if ((RHepatitis == 'present') and (THepatitis == 'present')):
	anorexia ~= choice({'present' : 0.1818182, 'absent' : 0.8181818})
elif ((RHepatitis == 'present') and (THepatitis == 'absent')):
	anorexia ~= choice({'present' : 0.1176471, 'absent' : 0.8823529})
elif ((RHepatitis == 'absent') and (THepatitis == 'present')):
	anorexia ~= choice({'present' : 0.2222222, 'absent' : 0.7777778})
else:
	anorexia ~= choice({'present' : 0.280916, 'absent' : 0.719084})


if ((gallstones == 'present')):
	choledocholithotomy ~= choice({'present' : 0.7102804, 'absent' : 0.2897196})
else:
	choledocholithotomy ~= choice({'present' : 0.03716216, 'absent' : 0.96283784})


if ((gallstones == 'present')):
	fat ~= choice({'present' : 0.1775701, 'absent' : 0.8224299})
else:
	fat ~= choice({'present' : 0.2804054, 'absent' : 0.7195946})


if ((gallstones == 'present')):
	flatulence ~= choice({'present' : 0.3925234, 'absent' : 0.6074766})
else:
	flatulence ~= choice({'present' : 0.4307432, 'absent' : 0.5692568})


if ((RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	hepatomegaly ~= choice({'present' : 0.6097561, 'absent' : 0.3902439})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	hepatomegaly ~= choice({'present' : 0.68, 'absent' : 0.32})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	hepatomegaly ~= choice({'present' : 0.5918367, 'absent' : 0.4081633})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'absent')):
	hepatomegaly ~= choice({'present' : 0.673913, 'absent' : 0.326087})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	hepatomegaly ~= choice({'present' : 0.5901639, 'absent' : 0.4098361})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	hepatomegaly ~= choice({'present' : 0.6527778, 'absent' : 0.3472222})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	hepatomegaly ~= choice({'present' : 0.5555556, 'absent' : 0.4444444})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'absent')):
	hepatomegaly ~= choice({'present' : 0.7058823, 'absent' : 0.2941177})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	hepatomegaly ~= choice({'present' : 0.6, 'absent' : 0.4})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	hepatomegaly ~= choice({'present' : 0.6756757, 'absent' : 0.3243243})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	hepatomegaly ~= choice({'present' : 0.5897436, 'absent' : 0.4102564})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'absent')):
	hepatomegaly ~= choice({'present' : 0.7777778, 'absent' : 0.2222222})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	hepatomegaly ~= choice({'present' : 0.5866667, 'absent' : 0.4133333})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	hepatomegaly ~= choice({'present' : 0.6865672, 'absent' : 0.3134328})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	hepatomegaly ~= choice({'present' : 0.375, 'absent' : 0.625})
else:
	hepatomegaly ~= choice({'present' : 0.6973684, 'absent' : 0.3026316})


if ((hospital == 'present') and (surgery == 'present') and (choledocholithotomy == 'present')):
	injections ~= choice({'present' : 0.8, 'absent' : 0.2})
elif ((hospital == 'present') and (surgery == 'present') and (choledocholithotomy == 'absent')):
	injections ~= choice({'present' : 0.715847, 'absent' : 0.284153})
elif ((hospital == 'present') and (surgery == 'absent') and (choledocholithotomy == 'present')):
	injections ~= choice({'present' : 0.8333333, 'absent' : 0.1666667})
elif ((hospital == 'present') and (surgery == 'absent') and (choledocholithotomy == 'absent')):
	injections ~= choice({'present' : 0.4818182, 'absent' : 0.5181818})
elif ((hospital == 'absent') and (surgery == 'present') and (choledocholithotomy == 'present')):
	injections ~= choice({'present' : 0.375, 'absent' : 0.625})
elif ((hospital == 'absent') and (surgery == 'present') and (choledocholithotomy == 'absent')):
	injections ~= choice({'present' : 0.2333333, 'absent' : 0.7666667})
elif ((hospital == 'absent') and (surgery == 'absent') and (choledocholithotomy == 'present')):
	injections ~= choice({'present' : 0.01098901, 'absent' : 0.98901099})
else:
	injections ~= choice({'present' : 0.0647482, 'absent' : 0.9352518})


if ((PBC == 'present')):
	joints ~= choice({'present' : 0.1285714, 'absent' : 0.8714286})
else:
	joints ~= choice({'present' : 0.1002387, 'absent' : 0.8997613})


if ((PBC == 'present')):
	le_cells ~= choice({'present' : 0.1214286, 'absent' : 0.8785714})
else:
	le_cells ~= choice({'present' : 0.04057279, 'absent' : 0.95942721})


if ((RHepatitis == 'present') and (THepatitis == 'present')):
	nausea ~= choice({'present' : 0.3636364, 'absent' : 0.6363636})
elif ((RHepatitis == 'present') and (THepatitis == 'absent')):
	nausea ~= choice({'present' : 0.3529412, 'absent' : 0.6470588})
elif ((RHepatitis == 'absent') and (THepatitis == 'present')):
	nausea ~= choice({'present' : 0.3703704, 'absent' : 0.6296296})
else:
	nausea ~= choice({'present' : 0.2854962, 'absent' : 0.7145038})


if ((PBC == 'present') and (joints == 'present')):
	pain ~= choice({'present' : 0.3888889, 'absent' : 0.6111111})
elif ((PBC == 'present') and (joints == 'absent')):
	pain ~= choice({'present' : 0.147541, 'absent' : 0.852459})
elif ((PBC == 'absent') and (joints == 'present')):
	pain ~= choice({'present' : 0.8095238, 'absent' : 0.1904762})
else:
	pain ~= choice({'present' : 0.1830239, 'absent' : 0.8169761})


if ((Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	pain_ruq ~= choice({'present' : 0.3934426, 'absent' : 0.6065574})
elif ((Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	pain_ruq ~= choice({'present' : 0.4776119, 'absent' : 0.5223881})
elif ((Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	pain_ruq ~= choice({'present' : 0.2857143, 'absent' : 0.7142857})
else:
	pain_ruq ~= choice({'present' : 0.421875, 'absent' : 0.578125})


if ((hospital == 'present') and (surgery == 'present') and (choledocholithotomy == 'present')):
	transfusion ~= choice({'present' : 0.3333333, 'absent' : 0.6666667})
elif ((hospital == 'present') and (surgery == 'present') and (choledocholithotomy == 'absent')):
	transfusion ~= choice({'present' : 0.2896175, 'absent' : 0.7103825})
elif ((hospital == 'present') and (surgery == 'absent') and (choledocholithotomy == 'present')):
	transfusion ~= choice({'present' : 0.1666667, 'absent' : 0.8333333})
elif ((hospital == 'present') and (surgery == 'absent') and (choledocholithotomy == 'absent')):
	transfusion ~= choice({'present' : 0.1181818, 'absent' : 0.8818182})
elif ((hospital == 'absent') and (surgery == 'present') and (choledocholithotomy == 'present')):
	transfusion ~= choice({'present' : 0.125, 'absent' : 0.875})
elif ((hospital == 'absent') and (surgery == 'present') and (choledocholithotomy == 'absent')):
	transfusion ~= choice({'present' : 0.3, 'absent' : 0.7})
elif ((hospital == 'absent') and (surgery == 'absent') and (choledocholithotomy == 'present')):
	transfusion ~= choice({'present' : 0.01098901, 'absent' : 0.98901099})
else:
	transfusion ~= choice({'present' : 0.01079137, 'absent' : 0.98920863})


if ((Steatosis == 'present')):
	triglycerides ~= choice({'a17_4' : 0.1791045, 'a3_2' : 0.1641791, 'a1_0' : 0.6567164})
else:
	triglycerides ~= choice({'a17_4' : 0.02373418, 'a3_2' : 0.03164557, 'a1_0' : 0.94462025})


if ((transfusion == 'present') and (vh_amn == 'present') and (injections == 'present')):
	ChHepatitis ~= choice({'active' : 0.2094241, 'persistent' : 0.0052356, 'absent' : 0.7853403})
elif ((transfusion == 'present') and (vh_amn == 'present') and (injections == 'absent')):
	ChHepatitis ~= choice({'active' : 0.4615385, 'persistent' : 0.3076923, 'absent' : 0.2307692})
elif ((transfusion == 'present') and (vh_amn == 'absent') and (injections == 'present')):
	ChHepatitis ~= choice({'active' : 0.06, 'persistent' : 0.06, 'absent' : 0.88})
elif ((transfusion == 'present') and (vh_amn == 'absent') and (injections == 'absent')):
	ChHepatitis ~= choice({'active' : 0.13043478, 'persistent' : 0.04347826, 'absent' : 0.82608696})
elif ((transfusion == 'absent') and (vh_amn == 'present') and (injections == 'present')):
	ChHepatitis ~= choice({'active' : 0.15384615, 'persistent' : 0.05128205, 'absent' : 0.7948718})
elif ((transfusion == 'absent') and (vh_amn == 'present') and (injections == 'absent')):
	ChHepatitis ~= choice({'active' : 0.24, 'persistent' : 0.14, 'absent' : 0.62})
elif ((transfusion == 'absent') and (vh_amn == 'absent') and (injections == 'present')):
	ChHepatitis ~= choice({'active' : 0.07692308, 'persistent' : 0.00591716, 'absent' : 0.91715976})
else:
	ChHepatitis ~= choice({'active' : 0.13095238, 'persistent' : 0.05357143, 'absent' : 0.81547619})


if ((PBC == 'present') and (ChHepatitis == 'active') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.2704918, 'a49_15' : 0.1721312, 'a14_0' : 0.557377})
elif ((PBC == 'present') and (ChHepatitis == 'active') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.2972973, 'a49_15' : 0.1837838, 'a14_0' : 0.5189189})
elif ((PBC == 'present') and (ChHepatitis == 'active') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.2941177, 'a49_15' : 0.1911765, 'a14_0' : 0.5147058})
elif ((PBC == 'present') and (ChHepatitis == 'active') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.3205575, 'a49_15' : 0.2055749, 'a14_0' : 0.4738676})
elif ((PBC == 'present') and (ChHepatitis == 'persistent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.3093923, 'a49_15' : 0.1767956, 'a14_0' : 0.5138121})
elif ((PBC == 'present') and (ChHepatitis == 'persistent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.3315508, 'a49_15' : 0.171123, 'a14_0' : 0.4973262})
elif ((PBC == 'present') and (ChHepatitis == 'persistent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.3333333, 'a49_15' : 0.172043, 'a14_0' : 0.4946237})
elif ((PBC == 'present') and (ChHepatitis == 'persistent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.368, 'a49_15' : 0.184, 'a14_0' : 0.448})
elif ((PBC == 'present') and (ChHepatitis == 'absent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.3425926, 'a49_15' : 0.1712963, 'a14_0' : 0.4861111})
elif ((PBC == 'present') and (ChHepatitis == 'absent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.3629893, 'a49_15' : 0.1779359, 'a14_0' : 0.4590747})
elif ((PBC == 'present') and (ChHepatitis == 'absent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.3636364, 'a49_15' : 0.1818182, 'a14_0' : 0.4545455})
elif ((PBC == 'present') and (ChHepatitis == 'absent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.4321429, 'a49_15' : 0.2107143, 'a14_0' : 0.3571429})
elif ((PBC == 'absent') and (ChHepatitis == 'active') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.2682927, 'a49_15' : 0.1768293, 'a14_0' : 0.554878})
elif ((PBC == 'absent') and (ChHepatitis == 'active') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.175, 'a49_15' : 0.1625, 'a14_0' : 0.6625})
elif ((PBC == 'absent') and (ChHepatitis == 'active') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.1045752, 'a49_15' : 0.1633987, 'a14_0' : 0.7320261})
elif ((PBC == 'absent') and (ChHepatitis == 'active') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.03296703, 'a49_15' : 0.21978022, 'a14_0' : 0.74725275})
elif ((PBC == 'absent') and (ChHepatitis == 'persistent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.06024096, 'a49_15' : 0.12048193, 'a14_0' : 0.81927711})
elif ((PBC == 'absent') and (ChHepatitis == 'persistent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.08602151, 'a49_15' : 0.08602151, 'a14_0' : 0.82795698})
elif ((PBC == 'absent') and (ChHepatitis == 'persistent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.05434783, 'a49_15' : 0.07608696, 'a14_0' : 0.86956521})
elif ((PBC == 'absent') and (ChHepatitis == 'persistent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.05555556, 'a49_15' : 0.05555556, 'a14_0' : 0.88888888})
elif ((PBC == 'absent') and (ChHepatitis == 'absent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.07594937, 'a49_15' : 0.06329114, 'a14_0' : 0.86075949})
elif ((PBC == 'absent') and (ChHepatitis == 'absent') and (Steatosis == 'present') and (Hyperbilirubinemia == 'absent')):
	ESR ~= choice({'a200_50' : 0.13432836, 'a49_15' : 0.05970149, 'a14_0' : 0.80597015})
elif ((PBC == 'absent') and (ChHepatitis == 'absent') and (Steatosis == 'absent') and (Hyperbilirubinemia == 'present')):
	ESR ~= choice({'a200_50' : 0.01785714, 'a49_15' : 0.07142857, 'a14_0' : 0.91071429})
else:
	ESR ~= choice({'a200_50' : 0.04733728, 'a49_15' : 0.05325444, 'a14_0' : 0.89940828})


if ((PBC == 'present') and (Steatosis == 'present') and (ChHepatitis == 'active')):
	cholesterol ~= choice({'a999_350' : 0.08965517, 'a349_240' : 0.28275862, 'a239_0' : 0.62758621})
elif ((PBC == 'present') and (Steatosis == 'present') and (ChHepatitis == 'persistent')):
	cholesterol ~= choice({'a999_350' : 0.09659091, 'a349_240' : 0.30113636, 'a239_0' : 0.60227273})
elif ((PBC == 'present') and (Steatosis == 'present') and (ChHepatitis == 'absent')):
	cholesterol ~= choice({'a999_350' : 0.1034483, 'a349_240' : 0.3256705, 'a239_0' : 0.5708812})
elif ((PBC == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'active')):
	cholesterol ~= choice({'a999_350' : 0.1015873, 'a349_240' : 0.3047619, 'a239_0' : 0.5936508})
elif ((PBC == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'persistent')):
	cholesterol ~= choice({'a999_350' : 0.1050955, 'a349_240' : 0.3152866, 'a239_0' : 0.5796179})
elif ((PBC == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'absent')):
	cholesterol ~= choice({'a999_350' : 0.125, 'a349_240' : 0.3642857, 'a239_0' : 0.5107143})
elif ((PBC == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'active')):
	cholesterol ~= choice({'a999_350' : 0.09174312, 'a349_240' : 0.27981651, 'a239_0' : 0.62844037})
elif ((PBC == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'persistent')):
	cholesterol ~= choice({'a999_350' : 0.06918239, 'a349_240' : 0.23899371, 'a239_0' : 0.6918239})
elif ((PBC == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'absent')):
	cholesterol ~= choice({'a999_350' : 0.04477612, 'a349_240' : 0.2238806, 'a239_0' : 0.73134328})
elif ((PBC == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'active')):
	cholesterol ~= choice({'a999_350' : 0.03296703, 'a349_240' : 0.06593407, 'a239_0' : 0.9010989})
elif ((PBC == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'persistent')):
	cholesterol ~= choice({'a999_350' : 0.00277008, 'a349_240' : 0.02770083, 'a239_0' : 0.96952909})
else:
	cholesterol ~= choice({'a999_350' : 0.00044425, 'a349_240' : 0.09773434, 'a239_0' : 0.90182141})


if ((ChHepatitis == 'active') and (THepatitis == 'present') and (RHepatitis == 'present')):
	fatigue ~= choice({'present' : 0.6363636, 'absent' : 0.3636364})
elif ((ChHepatitis == 'active') and (THepatitis == 'present') and (RHepatitis == 'absent')):
	fatigue ~= choice({'present' : 0.625, 'absent' : 0.375})
elif ((ChHepatitis == 'active') and (THepatitis == 'absent') and (RHepatitis == 'present')):
	fatigue ~= choice({'present' : 0.6236559, 'absent' : 0.3763441})
elif ((ChHepatitis == 'active') and (THepatitis == 'absent') and (RHepatitis == 'absent')):
	fatigue ~= choice({'present' : 0.6043956, 'absent' : 0.3956044})
elif ((ChHepatitis == 'persistent') and (THepatitis == 'present') and (RHepatitis == 'present')):
	fatigue ~= choice({'present' : 0.6071429, 'absent' : 0.3928571})
elif ((ChHepatitis == 'persistent') and (THepatitis == 'present') and (RHepatitis == 'absent')):
	fatigue ~= choice({'present' : 0.5932203, 'absent' : 0.4067797})
elif ((ChHepatitis == 'persistent') and (THepatitis == 'absent') and (RHepatitis == 'present')):
	fatigue ~= choice({'present' : 0.5892857, 'absent' : 0.4107143})
elif ((ChHepatitis == 'persistent') and (THepatitis == 'absent') and (RHepatitis == 'absent')):
	fatigue ~= choice({'present' : 0.5277778, 'absent' : 0.4722222})
elif ((ChHepatitis == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present')):
	fatigue ~= choice({'present' : 0.6153846, 'absent' : 0.3846154})
elif ((ChHepatitis == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent')):
	fatigue ~= choice({'present' : 0.6666667, 'absent' : 0.3333333})
elif ((ChHepatitis == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present')):
	fatigue ~= choice({'present' : 0.7058823, 'absent' : 0.2941177})
else:
	fatigue ~= choice({'present' : 0.5359849, 'absent' : 0.4640151})


if ((ChHepatitis == 'active')):
	fibrosis ~= choice({'present' : 0.3, 'absent' : 0.7})
elif ((ChHepatitis == 'persistent')):
	fibrosis ~= choice({'present' : 0.05, 'absent' : 0.95})
else:
	fibrosis ~= choice({'present' : 0.001, 'absent' : 0.999})


if ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1590909, 'a69_30' : 0.1477273, 'a29_10' : 0.1136364, 'a9_0' : 0.5795454})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1696429, 'a69_30' : 0.1607143, 'a29_10' : 0.125, 'a9_0' : 0.5446428})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1546392, 'a69_30' : 0.1443299, 'a29_10' : 0.1134021, 'a9_0' : 0.5876288})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1730769, 'a69_30' : 0.1538462, 'a29_10' : 0.125, 'a9_0' : 0.5480769})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1666667, 'a69_30' : 0.1481482, 'a29_10' : 0.1111111, 'a9_0' : 0.574074})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1854839, 'a69_30' : 0.1612903, 'a29_10' : 0.1209677, 'a9_0' : 0.5322581})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1694915, 'a69_30' : 0.1610169, 'a29_10' : 0.1186441, 'a9_0' : 0.5508475})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1832061, 'a69_30' : 0.1755725, 'a29_10' : 0.129771, 'a9_0' : 0.5114504})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1759259, 'a69_30' : 0.1574074, 'a29_10' : 0.1111111, 'a9_0' : 0.5555556})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1913044, 'a69_30' : 0.173913, 'a29_10' : 0.1217391, 'a9_0' : 0.5130435})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1885246, 'a69_30' : 0.1639344, 'a29_10' : 0.1065574, 'a9_0' : 0.5409836})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.2108843, 'a69_30' : 0.1836735, 'a29_10' : 0.1156463, 'a9_0' : 0.4897959})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1666667, 'a69_30' : 0.1590909, 'a29_10' : 0.1212121, 'a9_0' : 0.5530303})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1768707, 'a69_30' : 0.1632653, 'a29_10' : 0.1292517, 'a9_0' : 0.5306123})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1652893, 'a69_30' : 0.1487603, 'a29_10' : 0.1157025, 'a9_0' : 0.5702479})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1742424, 'a69_30' : 0.1590909, 'a29_10' : 0.1287879, 'a9_0' : 0.5378788})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1726619, 'a69_30' : 0.1510791, 'a29_10' : 0.1151079, 'a9_0' : 0.5611511})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1893491, 'a69_30' : 0.1656805, 'a29_10' : 0.1242604, 'a9_0' : 0.52071})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1741935, 'a69_30' : 0.1677419, 'a29_10' : 0.1225806, 'a9_0' : 0.535484})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1857923, 'a69_30' : 0.1857923, 'a29_10' : 0.1311475, 'a9_0' : 0.4972678})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1748252, 'a69_30' : 0.1678322, 'a29_10' : 0.1188811, 'a9_0' : 0.5384615})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.19375, 'a69_30' : 0.18125, 'a29_10' : 0.125, 'a9_0' : 0.5})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1918605, 'a69_30' : 0.1744186, 'a29_10' : 0.1104651, 'a9_0' : 0.5232558})
elif ((PBC == 'present') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.2142857, 'a69_30' : 0.1932773, 'a29_10' : 0.1176471, 'a9_0' : 0.4747899})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1756757, 'a69_30' : 0.1621622, 'a29_10' : 0.1216216, 'a9_0' : 0.5405405})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.18, 'a69_30' : 0.1666667, 'a29_10' : 0.1333333, 'a9_0' : 0.52})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1666667, 'a69_30' : 0.15, 'a29_10' : 0.125, 'a9_0' : 0.5583333})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1782946, 'a69_30' : 0.1627907, 'a29_10' : 0.131783, 'a9_0' : 0.5271317})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1764706, 'a69_30' : 0.1544118, 'a29_10' : 0.1176471, 'a9_0' : 0.5514705})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1939394, 'a69_30' : 0.169697, 'a29_10' : 0.1272727, 'a9_0' : 0.5090909})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1776316, 'a69_30' : 0.1710526, 'a29_10' : 0.125, 'a9_0' : 0.5263158})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1899441, 'a69_30' : 0.1899441, 'a29_10' : 0.1340782, 'a9_0' : 0.4860335})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1785714, 'a69_30' : 0.1714286, 'a29_10' : 0.1214286, 'a9_0' : 0.5285714})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1987179, 'a69_30' : 0.1858974, 'a29_10' : 0.1282051, 'a9_0' : 0.4871795})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1964286, 'a69_30' : 0.1785714, 'a29_10' : 0.1130952, 'a9_0' : 0.5119048})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.2207792, 'a69_30' : 0.1991342, 'a29_10' : 0.1212121, 'a9_0' : 0.4588745})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.178771, 'a69_30' : 0.1731844, 'a29_10' : 0.122905, 'a9_0' : 0.5251396})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1804878, 'a69_30' : 0.1756098, 'a29_10' : 0.1365854, 'a9_0' : 0.507317})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1698113, 'a69_30' : 0.1572327, 'a29_10' : 0.1257862, 'a9_0' : 0.5471698})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1843575, 'a69_30' : 0.1675978, 'a29_10' : 0.1340782, 'a9_0' : 0.5139665})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1822917, 'a69_30' : 0.1614583, 'a29_10' : 0.1197917, 'a9_0' : 0.5364583})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1977612, 'a69_30' : 0.1791045, 'a29_10' : 0.1268657, 'a9_0' : 0.4962687})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1826087, 'a69_30' : 0.1782609, 'a29_10' : 0.126087, 'a9_0' : 0.5130434})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.1939799, 'a69_30' : 0.1939799, 'a29_10' : 0.1371237, 'a9_0' : 0.4749164})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1846847, 'a69_30' : 0.1846847, 'a29_10' : 0.1261261, 'a9_0' : 0.5045045})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.2007435, 'a69_30' : 0.197026, 'a29_10' : 0.133829, 'a9_0' : 0.4684015})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1986755, 'a69_30' : 0.192053, 'a29_10' : 0.1225166, 'a9_0' : 0.486755})
elif ((PBC == 'present') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.2392857, 'a69_30' : 0.225, 'a29_10' : 0.1321429, 'a9_0' : 0.4035714})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.1509434, 'a69_30' : 0.1415094, 'a29_10' : 0.1226415, 'a9_0' : 0.5849057})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.10526316, 'a69_30' : 0.09210526, 'a29_10' : 0.13157895, 'a9_0' : 0.67105263})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.05555556, 'a69_30' : 0.03703704, 'a29_10' : 0.09259259, 'a9_0' : 0.81481481})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.06122449, 'a69_30' : 0.02040816, 'a29_10' : 0.10204082, 'a9_0' : 0.81632653})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.05649718, 'a69_30' : 0.00188324, 'a29_10' : 0.07532957, 'a9_0' : 0.86629001})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.07393715, 'a69_30' : 0.00184843, 'a29_10' : 0.09242144, 'a9_0' : 0.83179298})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.06557377, 'a69_30' : 0.04918033, 'a29_10' : 0.09836066, 'a9_0' : 0.78688524})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.078125, 'a69_30' : 0.078125, 'a29_10' : 0.125, 'a9_0' : 0.71875})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.04166667, 'a69_30' : 0.02083333, 'a29_10' : 0.08333333, 'a9_0' : 0.85416667})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.04761905, 'a69_30' : 0.02380952, 'a29_10' : 0.0952381, 'a9_0' : 0.83333333})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.04338395, 'a69_30' : 0.0021692, 'a29_10' : 0.04338395, 'a9_0' : 0.9110629})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.06651885, 'a69_30' : 0.00221729, 'a29_10' : 0.0443459, 'a9_0' : 0.88691796})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.05714286, 'a69_30' : 0.04285714, 'a29_10' : 0.1, 'a9_0' : 0.8})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.07142857, 'a69_30' : 0.07142857, 'a29_10' : 0.13095238, 'a9_0' : 0.72619048})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.04545455, 'a69_30' : 0.03030303, 'a29_10' : 0.10606061, 'a9_0' : 0.81818181})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.04615385, 'a69_30' : 0.03076923, 'a29_10' : 0.12307692, 'a9_0' : 0.8})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.04285714, 'a69_30' : 0.01428571, 'a29_10' : 0.08571429, 'a9_0' : 0.85714286})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.0617284, 'a69_30' : 0.01234568, 'a29_10' : 0.09876543, 'a9_0' : 0.82716049})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.06024096, 'a69_30' : 0.04819277, 'a29_10' : 0.09638554, 'a9_0' : 0.79518073})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.07070707, 'a69_30' : 0.08080808, 'a29_10' : 0.12121212, 'a9_0' : 0.72727273})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.04166667, 'a69_30' : 0.04166667, 'a29_10' : 0.09722222, 'a9_0' : 0.81944444})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.03030303, 'a69_30' : 0.03030303, 'a29_10' : 0.10606061, 'a9_0' : 0.83333333})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.02702703, 'a69_30' : 0.01351351, 'a29_10' : 0.05405405, 'a9_0' : 0.90540541})
elif ((PBC == 'absent') and (THepatitis == 'present') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.07380074, 'a69_30' : 0.00369004, 'a29_10' : 0.03690037, 'a9_0' : 0.88560885})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.06349206, 'a69_30' : 0.04761905, 'a29_10' : 0.11111111, 'a9_0' : 0.77777778})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.07594937, 'a69_30' : 0.07594937, 'a29_10' : 0.13924051, 'a9_0' : 0.70886075})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.0483871, 'a69_30' : 0.03225806, 'a29_10' : 0.11290323, 'a9_0' : 0.80645161})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.05, 'a69_30' : 0.03333333, 'a29_10' : 0.13333333, 'a9_0' : 0.78333334})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.04615385, 'a69_30' : 0.01538462, 'a29_10' : 0.09230769, 'a9_0' : 0.84615384})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.06756757, 'a69_30' : 0.01351351, 'a29_10' : 0.10810811, 'a9_0' : 0.81081081})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.06410256, 'a69_30' : 0.05128205, 'a29_10' : 0.1025641, 'a9_0' : 0.78205129})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.07692308, 'a69_30' : 0.08791209, 'a29_10' : 0.13186813, 'a9_0' : 0.7032967})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.04545455, 'a69_30' : 0.04545455, 'a29_10' : 0.10606061, 'a9_0' : 0.80303029})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.03448276, 'a69_30' : 0.03448276, 'a29_10' : 0.12068966, 'a9_0' : 0.81034482})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.03076923, 'a69_30' : 0.01538462, 'a29_10' : 0.06153846, 'a9_0' : 0.89230769})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'present') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.11695906, 'a69_30' : 0.00584795, 'a29_10' : 0.05847953, 'a9_0' : 0.81871346})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.06493506, 'a69_30' : 0.06493506, 'a29_10' : 0.11688312, 'a9_0' : 0.75324676})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.07692308, 'a69_30' : 0.08547009, 'a29_10' : 0.14529915, 'a9_0' : 0.69230768})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.04444444, 'a69_30' : 0.04444444, 'a29_10' : 0.12222222, 'a9_0' : 0.7888889})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.04210526, 'a69_30' : 0.04210526, 'a29_10' : 0.13684211, 'a9_0' : 0.77894737})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.03703704, 'a69_30' : 0.02777778, 'a29_10' : 0.10185185, 'a9_0' : 0.83333333})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'present') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.07462687, 'a69_30' : 0.02985075, 'a29_10' : 0.13432836, 'a9_0' : 0.76119402})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.05660377, 'a69_30' : 0.06603774, 'a29_10' : 0.12264151, 'a9_0' : 0.75471698})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'active') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.08791209, 'a69_30' : 0.14285714, 'a29_10' : 0.17582418, 'a9_0' : 0.59340659})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.04395604, 'a69_30' : 0.07692308, 'a29_10' : 0.13186813, 'a9_0' : 0.74725275})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'persistent') and (Hyperbilirubinemia == 'absent')):
	ggtp ~= choice({'a640_70' : 0.00277008, 'a69_30' : 0.05540166, 'a29_10' : 0.19390582, 'a9_0' : 0.74792244})
elif ((PBC == 'absent') and (THepatitis == 'absent') and (RHepatitis == 'absent') and (Steatosis == 'absent') and (ChHepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	ggtp ~= choice({'a640_70' : 0.00177936, 'a69_30' : 0.00177936, 'a29_10' : 0.01779359, 'a9_0' : 0.97864769})
else:
	ggtp ~= choice({'a640_70' : 0.08, 'a69_30' : 0.096, 'a29_10' : 0.144, 'a9_0' : 0.68})


if ((vh_amn == 'present') and (ChHepatitis == 'active')):
	hbc_anti ~= choice({'present' : 0.00355872, 'absent' : 0.99644128})
elif ((vh_amn == 'present') and (ChHepatitis == 'persistent')):
	hbc_anti ~= choice({'present' : 0.00763359, 'absent' : 0.99236641})
elif ((vh_amn == 'present') and (ChHepatitis == 'absent')):
	hbc_anti ~= choice({'present' : 0.0875, 'absent' : 0.9125})
elif ((vh_amn == 'absent') and (ChHepatitis == 'active')):
	hbc_anti ~= choice({'present' : 0.07936508, 'absent' : 0.92063492})
elif ((vh_amn == 'absent') and (ChHepatitis == 'persistent')):
	hbc_anti ~= choice({'present' : 0.1304348, 'absent' : 0.8695652})
else:
	hbc_anti ~= choice({'present' : 0.101626, 'absent' : 0.898374})


if ((vh_amn == 'present') and (ChHepatitis == 'active')):
	hbeag ~= choice({'present' : 0.00355872, 'absent' : 0.99644128})
elif ((vh_amn == 'present') and (ChHepatitis == 'persistent')):
	hbeag ~= choice({'present' : 0.00763359, 'absent' : 0.99236641})
elif ((vh_amn == 'present') and (ChHepatitis == 'absent')):
	hbeag ~= choice({'present' : 0.00124844, 'absent' : 0.99875156})
elif ((vh_amn == 'absent') and (ChHepatitis == 'active')):
	hbeag ~= choice({'present' : 0.00158479, 'absent' : 0.99841521})
elif ((vh_amn == 'absent') and (ChHepatitis == 'persistent')):
	hbeag ~= choice({'present' : 0.04347826, 'absent' : 0.95652174})
else:
	hbeag ~= choice({'present' : 0.00203252, 'absent' : 0.99796748})


if ((vh_amn == 'present') and (ChHepatitis == 'active')):
	hbsag ~= choice({'present' : 0.5, 'absent' : 0.5})
elif ((vh_amn == 'present') and (ChHepatitis == 'persistent')):
	hbsag ~= choice({'present' : 0.4615385, 'absent' : 0.5384615})
elif ((vh_amn == 'present') and (ChHepatitis == 'absent')):
	hbsag ~= choice({'present' : 0.1125, 'absent' : 0.8875})
elif ((vh_amn == 'absent') and (ChHepatitis == 'active')):
	hbsag ~= choice({'present' : 0.1904762, 'absent' : 0.8095238})
elif ((vh_amn == 'absent') and (ChHepatitis == 'persistent')):
	hbsag ~= choice({'present' : 0.04347826, 'absent' : 0.95652174})
else:
	hbsag ~= choice({'present' : 0.04674797, 'absent' : 0.95325203})


if ((vh_amn == 'present') and (ChHepatitis == 'active') and (hbsag == 'present')):
	hbsag_anti ~= choice({'present' : 0.0070922, 'absent' : 0.9929078})
elif ((vh_amn == 'present') and (ChHepatitis == 'active') and (hbsag == 'absent')):
	hbsag_anti ~= choice({'present' : 0.07142857, 'absent' : 0.92857143})
elif ((vh_amn == 'present') and (ChHepatitis == 'persistent') and (hbsag == 'present')):
	hbsag_anti ~= choice({'present' : 0.01639344, 'absent' : 0.98360656})
elif ((vh_amn == 'present') and (ChHepatitis == 'persistent') and (hbsag == 'absent')):
	hbsag_anti ~= choice({'present' : 0.01408451, 'absent' : 0.98591549})
elif ((vh_amn == 'present') and (ChHepatitis == 'absent') and (hbsag == 'present')):
	hbsag_anti ~= choice({'present' : 0.01098901, 'absent' : 0.98901099})
elif ((vh_amn == 'present') and (ChHepatitis == 'absent') and (hbsag == 'absent')):
	hbsag_anti ~= choice({'present' : 0.04225352, 'absent' : 0.95774648})
elif ((vh_amn == 'absent') and (ChHepatitis == 'active') and (hbsag == 'present')):
	hbsag_anti ~= choice({'present' : 0.08333333, 'absent' : 0.91666667})
elif ((vh_amn == 'absent') and (ChHepatitis == 'active') and (hbsag == 'absent')):
	hbsag_anti ~= choice({'present' : 0.00195695, 'absent' : 0.99804305})
elif ((vh_amn == 'absent') and (ChHepatitis == 'persistent') and (hbsag == 'present')):
	hbsag_anti ~= choice({'present' : 0.09090909, 'absent' : 0.90909091})
elif ((vh_amn == 'absent') and (ChHepatitis == 'persistent') and (hbsag == 'absent')):
	hbsag_anti ~= choice({'present' : 0.00452489, 'absent' : 0.99547511})
elif ((vh_amn == 'absent') and (ChHepatitis == 'absent') and (hbsag == 'present')):
	hbsag_anti ~= choice({'present' : 0.004329, 'absent' : 0.995671})
else:
	hbsag_anti ~= choice({'present' : 0.01492537, 'absent' : 0.98507463})


if ((vh_amn == 'present') and (ChHepatitis == 'active')):
	hcv_anti ~= choice({'present' : 0.00355872, 'absent' : 0.99644128})
elif ((vh_amn == 'present') and (ChHepatitis == 'persistent')):
	hcv_anti ~= choice({'present' : 0.00763359, 'absent' : 0.99236641})
elif ((vh_amn == 'present') and (ChHepatitis == 'absent')):
	hcv_anti ~= choice({'present' : 0.00124844, 'absent' : 0.99875156})
elif ((vh_amn == 'absent') and (ChHepatitis == 'active')):
	hcv_anti ~= choice({'present' : 0.00158479, 'absent' : 0.99841521})
elif ((vh_amn == 'absent') and (ChHepatitis == 'persistent')):
	hcv_anti ~= choice({'present' : 0.004329, 'absent' : 0.995671})
else:
	hcv_anti ~= choice({'present' : 0.00203252, 'absent' : 0.99796748})


if ((hepatomegaly == 'present')):
	hepatalgia ~= choice({'present' : 0.3142251, 'absent' : 0.6857749})
else:
	hepatalgia ~= choice({'present' : 0.03070175, 'absent' : 0.96929825})


if ((gallstones == 'present') and (PBC == 'present') and (ChHepatitis == 'active')):
	pressure_ruq ~= choice({'present' : 0.3333333, 'absent' : 0.6666667})
elif ((gallstones == 'present') and (PBC == 'present') and (ChHepatitis == 'persistent')):
	pressure_ruq ~= choice({'present' : 0.328125, 'absent' : 0.671875})
elif ((gallstones == 'present') and (PBC == 'present') and (ChHepatitis == 'absent')):
	pressure_ruq ~= choice({'present' : 0.3292683, 'absent' : 0.6707317})
elif ((gallstones == 'present') and (PBC == 'absent') and (ChHepatitis == 'active')):
	pressure_ruq ~= choice({'present' : 0.4, 'absent' : 0.6})
elif ((gallstones == 'present') and (PBC == 'absent') and (ChHepatitis == 'persistent')):
	pressure_ruq ~= choice({'present' : 0.09090909, 'absent' : 0.90909091})
elif ((gallstones == 'present') and (PBC == 'absent') and (ChHepatitis == 'absent')):
	pressure_ruq ~= choice({'present' : 0.2857143, 'absent' : 0.7142857})
elif ((gallstones == 'absent') and (PBC == 'present') and (ChHepatitis == 'active')):
	pressure_ruq ~= choice({'present' : 0.3424658, 'absent' : 0.6575342})
elif ((gallstones == 'absent') and (PBC == 'present') and (ChHepatitis == 'persistent')):
	pressure_ruq ~= choice({'present' : 0.3227513, 'absent' : 0.6772487})
elif ((gallstones == 'absent') and (PBC == 'present') and (ChHepatitis == 'absent')):
	pressure_ruq ~= choice({'present' : 0.2929293, 'absent' : 0.7070707})
elif ((gallstones == 'absent') and (PBC == 'absent') and (ChHepatitis == 'active')):
	pressure_ruq ~= choice({'present' : 0.4691358, 'absent' : 0.5308642})
elif ((gallstones == 'absent') and (PBC == 'absent') and (ChHepatitis == 'persistent')):
	pressure_ruq ~= choice({'present' : 0.4285714, 'absent' : 0.5714286})
else:
	pressure_ruq ~= choice({'present' : 0.4532374, 'absent' : 0.5467626})


if ((fibrosis == 'present') and (Steatosis == 'present')):
	Cirrhosis ~= choice({'decompensate' : 0.56, 'compensate' : 0.24, 'absent' : 0.2})
elif ((fibrosis == 'present') and (Steatosis == 'absent')):
	Cirrhosis ~= choice({'decompensate' : 0.49, 'compensate' : 0.21, 'absent' : 0.3})
elif ((fibrosis == 'absent') and (Steatosis == 'present')):
	Cirrhosis ~= choice({'decompensate' : 0.35, 'compensate' : 0.15, 'absent' : 0.5})
else:
	Cirrhosis ~= choice({'decompensate' : 0.001, 'compensate' : 0.001, 'absent' : 0.998})


if ((Cirrhosis == 'decompensate')):
	albumin ~= choice({'a70_50' : 0.91222031, 'a49_30' : 0.08605852, 'a29_0' : 0.00172117})
elif ((Cirrhosis == 'compensate')):
	albumin ~= choice({'a70_50' : 0.96463023, 'a49_30' : 0.00321543, 'a29_0' : 0.03215434})
else:
	albumin ~= choice({'a70_50' : 0.7393443, 'a49_30' : 0.1426229, 'a29_0' : 0.1180328})


if ((Cirrhosis == 'decompensate')):
	alcohol ~= choice({'present' : 0.2068966, 'absent' : 0.7931034})
elif ((Cirrhosis == 'compensate')):
	alcohol ~= choice({'present' : 0.2258064, 'absent' : 0.7741936})
else:
	alcohol ~= choice({'present' : 0.1114754, 'absent' : 0.8885246})


if ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.05882353, 'a199_100' : 0.15686275, 'a99_35' : 0.41176471, 'a34_0' : 0.37254902})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.05454545, 'a199_100' : 0.16363636, 'a99_35' : 0.41818182, 'a34_0' : 0.36363636})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.04761905, 'a199_100' : 0.15873016, 'a99_35' : 0.41269841, 'a34_0' : 0.38095238})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.06451613, 'a199_100' : 0.17741935, 'a99_35' : 0.41935484, 'a34_0' : 0.33870968})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.07017544, 'a199_100' : 0.19298246, 'a99_35' : 0.42105263, 'a34_0' : 0.31578947})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.07936508, 'a199_100' : 0.19047619, 'a99_35' : 0.41269841, 'a34_0' : 0.31746032})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.06849315, 'a199_100' : 0.16438356, 'a99_35' : 0.42465753, 'a34_0' : 0.34246575})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.05882353, 'a199_100' : 0.17647059, 'a99_35' : 0.42647059, 'a34_0' : 0.33823529})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.0625, 'a199_100' : 0.175, 'a99_35' : 0.4125, 'a34_0' : 0.35})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.075, 'a199_100' : 0.1875, 'a99_35' : 0.425, 'a34_0' : 0.3125})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.08333333, 'a199_100' : 0.20833333, 'a99_35' : 0.43055556, 'a34_0' : 0.27777778})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.08988764, 'a199_100' : 0.2247191, 'a99_35' : 0.41573034, 'a34_0' : 0.26966292})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.0617284, 'a199_100' : 0.1728395, 'a99_35' : 0.4197531, 'a34_0' : 0.345679})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.05479452, 'a199_100' : 0.16438356, 'a99_35' : 0.42465753, 'a34_0' : 0.35616438})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.05882353, 'a199_100' : 0.15294118, 'a99_35' : 0.41176471, 'a34_0' : 0.37647059})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.06976744, 'a199_100' : 0.1627907, 'a99_35' : 0.41860465, 'a34_0' : 0.34883721})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.07792208, 'a199_100' : 0.18181818, 'a99_35' : 0.41558442, 'a34_0' : 0.32467532})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.08333333, 'a199_100' : 0.1875, 'a99_35' : 0.40625, 'a34_0' : 0.32291667})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.06862745, 'a199_100' : 0.16666667, 'a99_35' : 0.42156863, 'a34_0' : 0.34313725})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.0625, 'a199_100' : 0.1666667, 'a99_35' : 0.4270833, 'a34_0' : 0.34375})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.064, 'a199_100' : 0.168, 'a99_35' : 0.416, 'a34_0' : 0.352})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.08148148, 'a199_100' : 0.17777778, 'a99_35' : 0.42222222, 'a34_0' : 0.31851852})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.08661417, 'a199_100' : 0.19685039, 'a99_35' : 0.42519685, 'a34_0' : 0.29133858})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.1208791, 'a199_100' : 0.2307692, 'a99_35' : 0.3956044, 'a34_0' : 0.2527472})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.05172414, 'a199_100' : 0.15517241, 'a99_35' : 0.39655172, 'a34_0' : 0.39655172})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.02173913, 'a199_100' : 0.13043478, 'a99_35' : 0.41304348, 'a34_0' : 0.43478261})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.0021692, 'a199_100' : 0.1084599, 'a99_35' : 0.3904555, 'a34_0' : 0.4989154})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.02272727, 'a199_100' : 0.11363636, 'a99_35' : 0.40909091, 'a34_0' : 0.45454545})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.00269542, 'a199_100' : 0.13477089, 'a99_35' : 0.40431267, 'a34_0' : 0.45822102})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.00262467, 'a199_100' : 0.1312336, 'a99_35' : 0.36745407, 'a34_0' : 0.49868766})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.01923077, 'a199_100' : 0.11538462, 'a99_35' : 0.40384615, 'a34_0' : 0.46153846})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.002079, 'a199_100' : 0.1247401, 'a99_35' : 0.4158004, 'a34_0' : 0.4573805})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.00181488, 'a199_100' : 0.12704174, 'a99_35' : 0.39927405, 'a34_0' : 0.47186933})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.01886792, 'a199_100' : 0.13207547, 'a99_35' : 0.41509434, 'a34_0' : 0.43396226})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.02272727, 'a199_100' : 0.15909091, 'a99_35' : 0.40909091, 'a34_0' : 0.40909091})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.02083333, 'a199_100' : 0.16666667, 'a99_35' : 0.375, 'a34_0' : 0.4375})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.01724138, 'a199_100' : 0.12068966, 'a99_35' : 0.39655172, 'a34_0' : 0.46551724})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.00191939, 'a199_100' : 0.11516315, 'a99_35' : 0.40307102, 'a34_0' : 0.47984645})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.00166389, 'a199_100' : 0.09983361, 'a99_35' : 0.38269551, 'a34_0' : 0.51580699})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.01724138, 'a199_100' : 0.10344828, 'a99_35' : 0.39655172, 'a34_0' : 0.48275862})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.02040816, 'a199_100' : 0.12244898, 'a99_35' : 0.3877551, 'a34_0' : 0.46938776})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.01818182, 'a199_100' : 0.10909091, 'a99_35' : 0.34545455, 'a34_0' : 0.52727272})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.02777778, 'a199_100' : 0.11111111, 'a99_35' : 0.38888889, 'a34_0' : 0.47222222})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.01492537, 'a199_100' : 0.11940299, 'a99_35' : 0.40298507, 'a34_0' : 0.46268657})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.01190476, 'a199_100' : 0.10714286, 'a99_35' : 0.38095238, 'a34_0' : 0.5})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.03409091, 'a199_100' : 0.11363636, 'a99_35' : 0.38636364, 'a34_0' : 0.46590909})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.02631579, 'a199_100' : 0.13157895, 'a99_35' : 0.39473684, 'a34_0' : 0.44736842})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.02777778, 'a199_100' : 0.13888889, 'a99_35' : 0.27777778, 'a34_0' : 0.55555555})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.02, 'a199_100' : 0.12, 'a99_35' : 0.4, 'a34_0' : 0.46})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.00212314, 'a199_100' : 0.12738854, 'a99_35' : 0.42462845, 'a34_0' : 0.44585987})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.00191939, 'a199_100' : 0.11516315, 'a99_35' : 0.42226488, 'a34_0' : 0.46065259})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.02, 'a199_100' : 0.12, 'a99_35' : 0.44, 'a34_0' : 0.42})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.00249377, 'a199_100' : 0.14962594, 'a99_35' : 0.44887781, 'a34_0' : 0.39900249})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.0023753, 'a199_100' : 0.1425178, 'a99_35' : 0.4275534, 'a34_0' : 0.4275534})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.01666667, 'a199_100' : 0.11666667, 'a99_35' : 0.45, 'a34_0' : 0.41666667})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.00178253, 'a199_100' : 0.12477718, 'a99_35' : 0.46345811, 'a34_0' : 0.40998217})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.00144718, 'a199_100' : 0.11577424, 'a99_35' : 0.44862518, 'a34_0' : 0.4341534})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.02816901, 'a199_100' : 0.12676056, 'a99_35' : 0.46478873, 'a34_0' : 0.38028169})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.01724138, 'a199_100' : 0.15517241, 'a99_35' : 0.48275862, 'a34_0' : 0.34482759})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.00584795, 'a199_100' : 0.23391813, 'a99_35' : 0.46783626, 'a34_0' : 0.29239766})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.01818182, 'a199_100' : 0.10909091, 'a99_35' : 0.43636364, 'a34_0' : 0.43636364})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.00172117, 'a199_100' : 0.10327022, 'a99_35' : 0.4475043, 'a34_0' : 0.4475043})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.00131406, 'a199_100' : 0.09198423, 'a99_35' : 0.42049934, 'a34_0' : 0.48620237})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.025, 'a199_100' : 0.1, 'a99_35' : 0.425, 'a34_0' : 0.45})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.01470588, 'a199_100' : 0.11764706, 'a99_35' : 0.44117647, 'a34_0' : 0.42647059})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.00369004, 'a199_100' : 0.07380074, 'a99_35' : 0.36900369, 'a34_0' : 0.55350553})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.02666667, 'a199_100' : 0.09333333, 'a99_35' : 0.42666667, 'a34_0' : 0.45333333})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.01176471, 'a199_100' : 0.10588235, 'a99_35' : 0.44705882, 'a34_0' : 0.43529412})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	alt ~= choice({'a850_200' : 0.00149031, 'a199_100' : 0.08941878, 'a99_35' : 0.41728763, 'a34_0' : 0.49180328})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	alt ~= choice({'a850_200' : 0.06896552, 'a199_100' : 0.12068966, 'a99_35' : 0.46551724, 'a34_0' : 0.34482759})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	alt ~= choice({'a850_200' : 0.03225806, 'a199_100' : 0.19354839, 'a99_35' : 0.51612903, 'a34_0' : 0.25806452})
else:
	alt ~= choice({'a850_200' : 0.04569892, 'a199_100' : 0.17473118, 'a99_35' : 0.42741935, 'a34_0' : 0.35215054})


if ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.01960784, 'a399_150' : 0.1372549, 'a149_40' : 0.47058824, 'a39_0' : 0.37254902})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.01818182, 'a399_150' : 0.12727273, 'a149_40' : 0.49090909, 'a39_0' : 0.36363636})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.01612903, 'a399_150' : 0.14516129, 'a149_40' : 0.46774194, 'a39_0' : 0.37096774})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.01612903, 'a399_150' : 0.16129032, 'a149_40' : 0.48387097, 'a39_0' : 0.33870968})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.01818182, 'a399_150' : 0.16363636, 'a149_40' : 0.50909091, 'a39_0' : 0.30909091})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.03225806, 'a399_150' : 0.17741935, 'a149_40' : 0.48387097, 'a39_0' : 0.30645161})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.02777778, 'a399_150' : 0.15277778, 'a149_40' : 0.48611111, 'a39_0' : 0.33333333})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.02941176, 'a399_150' : 0.14705882, 'a149_40' : 0.50000001, 'a39_0' : 0.32352941})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.02531646, 'a399_150' : 0.15189873, 'a149_40' : 0.48101266, 'a39_0' : 0.34177215})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.025, 'a399_150' : 0.175, 'a149_40' : 0.5, 'a39_0' : 0.3})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.02777778, 'a399_150' : 0.18055556, 'a149_40' : 0.52777777, 'a39_0' : 0.26388889})
elif ((ChHepatitis == 'active') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.03370787, 'a399_150' : 0.20224719, 'a149_40' : 0.50561797, 'a39_0' : 0.25842697})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.02469136, 'a399_150' : 0.16049383, 'a149_40' : 0.4691358, 'a39_0' : 0.34567901})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.02739726, 'a399_150' : 0.1369863, 'a149_40' : 0.47945205, 'a39_0' : 0.35616438})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.02380952, 'a399_150' : 0.14285714, 'a149_40' : 0.45238095, 'a39_0' : 0.38095238})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.02352941, 'a399_150' : 0.16470588, 'a149_40' : 0.45882353, 'a39_0' : 0.35294118})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.02597403, 'a399_150' : 0.16883117, 'a149_40' : 0.48051948, 'a39_0' : 0.32467532})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.03125, 'a399_150' : 0.1875, 'a149_40' : 0.4583333, 'a39_0' : 0.3229167})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.02912621, 'a399_150' : 0.16504854, 'a149_40' : 0.46601942, 'a39_0' : 0.33980583})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.03125, 'a399_150' : 0.15625, 'a149_40' : 0.4791667, 'a39_0' : 0.3333333})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.03174603, 'a399_150' : 0.15873016, 'a149_40' : 0.46031746, 'a39_0' : 0.34920635})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.03676471, 'a399_150' : 0.17647059, 'a149_40' : 0.47058824, 'a39_0' : 0.31617647})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.03937008, 'a399_150' : 0.18110236, 'a149_40' : 0.49606299, 'a39_0' : 0.28346457})
elif ((ChHepatitis == 'active') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.05494505, 'a399_150' : 0.23076923, 'a149_40' : 0.46153846, 'a39_0' : 0.25274725})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.01754386, 'a399_150' : 0.14035088, 'a149_40' : 0.45614035, 'a39_0' : 0.38596491})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00221729, 'a399_150' : 0.0886918, 'a149_40' : 0.48780488, 'a39_0' : 0.42128603})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00212314, 'a399_150' : 0.08492569, 'a149_40' : 0.44585987, 'a39_0' : 0.4670913})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00221729, 'a399_150' : 0.11086475, 'a149_40' : 0.46563193, 'a39_0' : 0.42128603})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00269542, 'a399_150' : 0.08086253, 'a149_40' : 0.51212938, 'a39_0' : 0.40431267})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00269542, 'a399_150' : 0.08086253, 'a149_40' : 0.45822102, 'a39_0' : 0.45822102})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00191939, 'a399_150' : 0.09596929, 'a149_40' : 0.46065259, 'a39_0' : 0.44145873})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00203666, 'a399_150' : 0.0814664, 'a149_40' : 0.48879837, 'a39_0' : 0.42769857})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00181488, 'a399_150' : 0.0907441, 'a149_40' : 0.45372051, 'a39_0' : 0.45372051})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00184843, 'a399_150' : 0.11090573, 'a149_40' : 0.4805915, 'a39_0' : 0.40665434})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00226757, 'a399_150' : 0.09070295, 'a149_40' : 0.52154195, 'a39_0' : 0.38548753})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00212314, 'a399_150' : 0.10615711, 'a149_40' : 0.48832272, 'a39_0' : 0.40339703})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00172117, 'a399_150' : 0.10327022, 'a149_40' : 0.4475043, 'a39_0' : 0.4475043})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00191939, 'a399_150' : 0.07677543, 'a149_40' : 0.46065259, 'a39_0' : 0.46065259})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00166389, 'a399_150' : 0.08319468, 'a149_40' : 0.41597338, 'a39_0' : 0.49916805})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00172117, 'a399_150' : 0.10327022, 'a149_40' : 0.4302926, 'a39_0' : 0.46471601})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00203666, 'a399_150' : 0.0814664, 'a149_40' : 0.46843177, 'a39_0' : 0.44806517})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00181488, 'a399_150' : 0.0907441, 'a149_40' : 0.41742287, 'a39_0' : 0.49001815})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00142653, 'a399_150' : 0.09985735, 'a149_40' : 0.44222539, 'a39_0' : 0.45649073})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00149031, 'a399_150' : 0.08941878, 'a149_40' : 0.46199702, 'a39_0' : 0.44709389})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00120337, 'a399_150' : 0.08423586, 'a149_40' : 0.433213, 'a39_0' : 0.48134777})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.01136364, 'a399_150' : 0.10227273, 'a149_40' : 0.44318182, 'a39_0' : 0.44318182})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.01315789, 'a399_150' : 0.09210526, 'a149_40' : 0.47368421, 'a39_0' : 0.42105263})
elif ((ChHepatitis == 'persistent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.02777778, 'a399_150' : 0.11111111, 'a149_40' : 0.36111111, 'a39_0' : 0.5})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00199601, 'a399_150' : 0.0998004, 'a149_40' : 0.45908184, 'a39_0' : 0.43912176})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00212314, 'a399_150' : 0.08492569, 'a149_40' : 0.48832272, 'a39_0' : 0.42462845})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00191939, 'a399_150' : 0.07677543, 'a149_40' : 0.46065259, 'a39_0' : 0.46065259})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00199601, 'a399_150' : 0.0998004, 'a149_40' : 0.47904192, 'a39_0' : 0.41916168})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00249377, 'a399_150' : 0.07481297, 'a149_40' : 0.54862842, 'a39_0' : 0.37406484})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00243309, 'a399_150' : 0.0729927, 'a149_40' : 0.51094891, 'a39_0' : 0.4136253})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00166389, 'a399_150' : 0.09983361, 'a149_40' : 0.49916805, 'a39_0' : 0.39933444})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00175131, 'a399_150' : 0.08756567, 'a149_40' : 0.52539405, 'a39_0' : 0.38528897})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00142653, 'a399_150' : 0.08559201, 'a149_40' : 0.49928673, 'a39_0' : 0.41369472})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00140647, 'a399_150' : 0.11251758, 'a149_40' : 0.52039381, 'a39_0' : 0.36568214})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00172117, 'a399_150' : 0.10327022, 'a149_40' : 0.58519794, 'a39_0' : 0.30981067})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00584795, 'a399_150' : 0.11695906, 'a149_40' : 0.64327486, 'a39_0' : 0.23391813})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00181488, 'a399_150' : 0.0907441, 'a149_40' : 0.47186933, 'a39_0' : 0.43557169})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00172117, 'a399_150' : 0.06884682, 'a149_40' : 0.48192771, 'a39_0' : 0.4475043})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00133156, 'a399_150' : 0.0665779, 'a149_40' : 0.43941411, 'a39_0' : 0.49267643})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00126422, 'a399_150' : 0.08849558, 'a149_40' : 0.4551201, 'a39_0' : 0.4551201})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00149031, 'a399_150' : 0.07451565, 'a149_40' : 0.49180328, 'a39_0' : 0.43219076})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present') and (Steatosis == 'absent') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00369004, 'a399_150' : 0.07380074, 'a149_40' : 0.36900369, 'a39_0' : 0.55350553})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.00133156, 'a399_150' : 0.09320905, 'a149_40' : 0.45272969, 'a39_0' : 0.45272969})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.00116144, 'a399_150' : 0.08130081, 'a149_40' : 0.48780488, 'a39_0' : 0.42973287})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'present') and (Cirrhosis == 'absent')):
	ast ~= choice({'a700_400' : 0.00149031, 'a399_150' : 0.07451565, 'a149_40' : 0.43219076, 'a39_0' : 0.49180328})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'decompensate')):
	ast ~= choice({'a700_400' : 0.01724138, 'a399_150' : 0.13793103, 'a149_40' : 0.5, 'a39_0' : 0.34482759})
elif ((ChHepatitis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'absent') and (Steatosis == 'absent') and (Cirrhosis == 'compensate')):
	ast ~= choice({'a700_400' : 0.03225806, 'a399_150' : 0.06451613, 'a149_40' : 0.67741936, 'a39_0' : 0.22580645})
else:
	ast ~= choice({'a700_400' : 0.01075269, 'a399_150' : 0.22580645, 'a149_40' : 0.46774194, 'a39_0' : 0.29569892})


if ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.04347826, 'a19_7' : 0.2173913, 'a6_2' : 0.34782609, 'a1_0' : 0.39130435})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.07407407, 'a19_7' : 0.22222222, 'a6_2' : 0.33333333, 'a1_0' : 0.37037037})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.07894737, 'a19_7' : 0.23684211, 'a6_2' : 0.34210526, 'a1_0' : 0.34210526})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.01923077, 'a19_7' : 0.11538462, 'a6_2' : 0.36538462, 'a1_0' : 0.5})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.01818182, 'a19_7' : 0.11818182, 'a6_2' : 0.38181818, 'a1_0' : 0.48181818})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.02189781, 'a19_7' : 0.12408759, 'a6_2' : 0.41605839, 'a1_0' : 0.4379562})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.03571429, 'a19_7' : 0.16071429, 'a6_2' : 0.39285714, 'a1_0' : 0.41071429})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.05882353, 'a19_7' : 0.20588235, 'a6_2' : 0.38235294, 'a1_0' : 0.35294118})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.07692308, 'a19_7' : 0.23076923, 'a6_2' : 0.35897436, 'a1_0' : 0.33333333})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.02020202, 'a19_7' : 0.11111111, 'a6_2' : 0.34343434, 'a1_0' : 0.52525253})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.01941748, 'a19_7' : 0.10679612, 'a6_2' : 0.34951456, 'a1_0' : 0.52427184})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.02362205, 'a19_7' : 0.11811024, 'a6_2' : 0.37795276, 'a1_0' : 0.48031496})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.04225352, 'a19_7' : 0.15492958, 'a6_2' : 0.36619718, 'a1_0' : 0.43661972})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.06, 'a19_7' : 0.2, 'a6_2' : 0.36, 'a1_0' : 0.38})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.07575758, 'a19_7' : 0.22727273, 'a6_2' : 0.36363636, 'a1_0' : 0.33333333})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.03030303, 'a19_7' : 0.12121212, 'a6_2' : 0.35606061, 'a1_0' : 0.49242424})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.02158273, 'a19_7' : 0.11510791, 'a6_2' : 0.37410072, 'a1_0' : 0.48920863})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.02061856, 'a19_7' : 0.12371134, 'a6_2' : 0.40721649, 'a1_0' : 0.44845361})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.01449275, 'a19_7' : 0.11594203, 'a6_2' : 0.39130435, 'a1_0' : 0.47826087})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.00398406, 'a19_7' : 0.11952191, 'a6_2' : 0.35856574, 'a1_0' : 0.51792829})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.00662252, 'a19_7' : 0.13245033, 'a6_2' : 0.26490066, 'a1_0' : 0.59602649})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.00151286, 'a19_7' : 0.04538578, 'a6_2' : 0.34795764, 'a1_0' : 0.60514372})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.00144718, 'a19_7' : 0.04341534, 'a6_2' : 0.37626628, 'a1_0' : 0.5788712})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.00114811, 'a19_7' : 0.05740528, 'a6_2' : 0.44776119, 'a1_0' : 0.49368542})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.00311526, 'a19_7' : 0.03115265, 'a6_2' : 0.43613707, 'a1_0' : 0.52959502})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.00892857, 'a19_7' : 0.00892857, 'a6_2' : 0.44642857, 'a1_0' : 0.53571429})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.01388889, 'a19_7' : 0.01388889, 'a6_2' : 0.41666667, 'a1_0' : 0.55555555})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.00178253, 'a19_7' : 0.01782531, 'a6_2' : 0.28520499, 'a1_0' : 0.69518717})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.00174825, 'a19_7' : 0.00174825, 'a6_2' : 0.2972028, 'a1_0' : 0.6993007})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.00144509, 'a19_7' : 0.00144509, 'a6_2' : 0.36127168, 'a1_0' : 0.63583814})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.00255102, 'a19_7' : 0.00255102, 'a6_2' : 0.33163265, 'a1_0' : 0.66326531})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.0049505, 'a19_7' : 0.0049505, 'a6_2' : 0.2970297, 'a1_0' : 0.6930693})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.07692308, 'a19_7' : 0.07692308, 'a6_2' : 0.07692308, 'a1_0' : 0.76923076})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.00146843, 'a19_7' : 0.01468429, 'a6_2' : 0.30837004, 'a1_0' : 0.67547724})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.00127877, 'a19_7' : 0.00127877, 'a6_2' : 0.33248082, 'a1_0' : 0.66496164})
elif ((Hyperbilirubinemia == 'present') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.00181159, 'a19_7' : 0.00181159, 'a6_2' : 0.54347827, 'a1_0' : 0.45289855})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.04081633, 'a19_7' : 0.14285714, 'a6_2' : 0.40816327, 'a1_0' : 0.40816327})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.06818182, 'a19_7' : 0.20454545, 'a6_2' : 0.36363636, 'a1_0' : 0.36363636})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.07692308, 'a19_7' : 0.23076923, 'a6_2' : 0.35384615, 'a1_0' : 0.33846154})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.03030303, 'a19_7' : 0.14393939, 'a6_2' : 0.34090909, 'a1_0' : 0.48484848})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.02173913, 'a19_7' : 0.13768116, 'a6_2' : 0.35507246, 'a1_0' : 0.48550725})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.02590674, 'a19_7' : 0.15025907, 'a6_2' : 0.38341969, 'a1_0' : 0.44041451})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.04255319, 'a19_7' : 0.17021277, 'a6_2' : 0.37234043, 'a1_0' : 0.41489362})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.05084746, 'a19_7' : 0.20338983, 'a6_2' : 0.37288136, 'a1_0' : 0.37288136})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.07042254, 'a19_7' : 0.22535211, 'a6_2' : 0.36619718, 'a1_0' : 0.33802817})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.032, 'a19_7' : 0.136, 'a6_2' : 0.312, 'a1_0' : 0.52})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.0234375, 'a19_7' : 0.1328125, 'a6_2' : 0.3125, 'a1_0' : 0.53125})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.02824859, 'a19_7' : 0.14124294, 'a6_2' : 0.33898305, 'a1_0' : 0.49152542})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.04477612, 'a19_7' : 0.1641791, 'a6_2' : 0.34328358, 'a1_0' : 0.44776119})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.05555556, 'a19_7' : 0.19444444, 'a6_2' : 0.35185185, 'a1_0' : 0.39814815})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.08536585, 'a19_7' : 0.24390244, 'a6_2' : 0.36585366, 'a1_0' : 0.30487805})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.03910615, 'a19_7' : 0.15083799, 'a6_2' : 0.32960894, 'a1_0' : 0.48044693})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.02926829, 'a19_7' : 0.14634146, 'a6_2' : 0.33658537, 'a1_0' : 0.48780488})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'present') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.03030303, 'a19_7' : 0.16666667, 'a6_2' : 0.38888889, 'a1_0' : 0.41414141})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.02830189, 'a19_7' : 0.16037736, 'a6_2' : 0.36792453, 'a1_0' : 0.44339623})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.01818182, 'a19_7' : 0.16363636, 'a6_2' : 0.34545455, 'a1_0' : 0.47272727})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.01923077, 'a19_7' : 0.19230769, 'a6_2' : 0.01923077, 'a1_0' : 0.76923077})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.01449275, 'a19_7' : 0.07246377, 'a6_2' : 0.24637681, 'a1_0' : 0.66666667})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.01282051, 'a19_7' : 0.07692308, 'a6_2' : 0.26923077, 'a1_0' : 0.64102564})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'decompensate') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.01886792, 'a19_7' : 0.13207547, 'a6_2' : 0.43396226, 'a1_0' : 0.41509434})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.00321543, 'a19_7' : 0.09646302, 'a6_2' : 0.41800643, 'a1_0' : 0.48231511})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.00621118, 'a19_7' : 0.0621118, 'a6_2' : 0.43478261, 'a1_0' : 0.49689441})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.04545455, 'a19_7' : 0.04545455, 'a6_2' : 0.45454545, 'a1_0' : 0.45454545})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.01818182, 'a19_7' : 0.01818182, 'a6_2' : 0.14545455, 'a1_0' : 0.81818181})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.01694915, 'a19_7' : 0.01694915, 'a6_2' : 0.10169492, 'a1_0' : 0.86440678})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'compensate') and (gallstones == 'absent') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.03448276, 'a19_7' : 0.03448276, 'a6_2' : 0.10344828, 'a1_0' : 0.8275862})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.00980392, 'a19_7' : 0.00980392, 'a6_2' : 0.19607843, 'a1_0' : 0.78431373})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.07692308, 'a19_7' : 0.07692308, 'a6_2' : 0.07692308, 'a1_0' : 0.76923076})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'present') and (ChHepatitis == 'absent')):
	bilirubin ~= choice({'a88_20' : 0.01587302, 'a19_7' : 0.01587302, 'a6_2' : 0.01587302, 'a1_0' : 0.95238094})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'active')):
	bilirubin ~= choice({'a88_20' : 0.01234568, 'a19_7' : 0.02469136, 'a6_2' : 0.14814815, 'a1_0' : 0.81481481})
elif ((Hyperbilirubinemia == 'absent') and (PBC == 'absent') and (Cirrhosis == 'absent') and (gallstones == 'absent') and (ChHepatitis == 'persistent')):
	bilirubin ~= choice({'a88_20' : 0.00284091, 'a19_7' : 0.00284091, 'a6_2' : 0.05681818, 'a1_0' : 0.9375})
else:
	bilirubin ~= choice({'a88_20' : 0.00070872, 'a19_7' : 0.02126152, 'a6_2' : 0.13465627, 'a1_0' : 0.84337349})


if ((Cirrhosis == 'decompensate') and (PBC == 'present')):
	carcinoma ~= choice({'present' : 0.3636364, 'absent' : 0.6363636})
elif ((Cirrhosis == 'decompensate') and (PBC == 'absent')):
	carcinoma ~= choice({'present' : 0.3, 'absent' : 0.7})
elif ((Cirrhosis == 'compensate') and (PBC == 'present')):
	carcinoma ~= choice({'present' : 0.2727273, 'absent' : 0.7272727})
elif ((Cirrhosis == 'compensate') and (PBC == 'absent')):
	carcinoma ~= choice({'present' : 0.2, 'absent' : 0.8})
elif ((Cirrhosis == 'absent') and (PBC == 'present')):
	carcinoma ~= choice({'present' : 0.1, 'absent' : 0.9})
else:
	carcinoma ~= choice({'present' : 0.01, 'absent' : 0.99})


if ((Cirrhosis == 'decompensate')):
	edema ~= choice({'present' : 0.3448276, 'absent' : 0.6551724})
elif ((Cirrhosis == 'compensate')):
	edema ~= choice({'present' : 0.06451613, 'absent' : 0.93548387})
else:
	edema ~= choice({'present' : 0.1311475, 'absent' : 0.8688525})


if ((Cirrhosis == 'decompensate')):
	edge ~= choice({'present' : 0.7586207, 'absent' : 0.2413793})
elif ((Cirrhosis == 'compensate')):
	edge ~= choice({'present' : 0.4516129, 'absent' : 0.5483871})
else:
	edge ~= choice({'present' : 0.2344262, 'absent' : 0.7655738})


if ((Cirrhosis == 'decompensate') and (PBC == 'present')):
	encephalopathy ~= choice({'present' : 0.05325444, 'absent' : 0.94674556})
elif ((Cirrhosis == 'decompensate') and (PBC == 'absent')):
	encephalopathy ~= choice({'present' : 0.05172414, 'absent' : 0.94827586})
elif ((Cirrhosis == 'compensate') and (PBC == 'present')):
	encephalopathy ~= choice({'present' : 0.04891304, 'absent' : 0.95108696})
elif ((Cirrhosis == 'compensate') and (PBC == 'absent')):
	encephalopathy ~= choice({'present' : 0.00321543, 'absent' : 0.99678457})
elif ((Cirrhosis == 'absent') and (PBC == 'present')):
	encephalopathy ~= choice({'present' : 0.05357143, 'absent' : 0.94642857})
else:
	encephalopathy ~= choice({'present' : 0.01515152, 'absent' : 0.98484848})


if ((ChHepatitis == 'active') and (Cirrhosis == 'decompensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.01754386, 'a109_70' : 0.84210526, 'a69_0' : 0.14035088})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'decompensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.01298701, 'a109_70' : 0.81818182, 'a69_0' : 0.16883117})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'decompensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.02150538, 'a109_70' : 0.8172043, 'a69_0' : 0.16129032})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'decompensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.01666667, 'a109_70' : 0.8, 'a69_0' : 0.18333333})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'compensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.025, 'a109_70' : 0.85, 'a69_0' : 0.125})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'compensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.01333333, 'a109_70' : 0.86666667, 'a69_0' : 0.12})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'compensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.02409639, 'a109_70' : 0.86746988, 'a69_0' : 0.10843373})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'compensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.01960784, 'a109_70' : 0.85294118, 'a69_0' : 0.12745098})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'absent') and (THepatitis == 'present') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.02197802, 'a109_70' : 0.89010989, 'a69_0' : 0.08791209})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'absent') and (THepatitis == 'present') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.01923077, 'a109_70' : 0.90384615, 'a69_0' : 0.07692308})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'absent') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.024, 'a109_70' : 0.904, 'a69_0' : 0.072})
elif ((ChHepatitis == 'active') and (Cirrhosis == 'absent') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.02197802, 'a109_70' : 0.9010989, 'a69_0' : 0.07692308})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'decompensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.03030303, 'a109_70' : 0.84848485, 'a69_0' : 0.12121212})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'decompensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.03225806, 'a109_70' : 0.79032259, 'a69_0' : 0.17741935})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'decompensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.02898551, 'a109_70' : 0.79710145, 'a69_0' : 0.17391304})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'decompensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.02469136, 'a109_70' : 0.75308642, 'a69_0' : 0.22222222})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'compensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.03508772, 'a109_70' : 0.8245614, 'a69_0' : 0.14035088})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'compensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.04081633, 'a109_70' : 0.83673469, 'a69_0' : 0.12244898})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'compensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.03571429, 'a109_70' : 0.85714285, 'a69_0' : 0.10714286})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'compensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.03333333, 'a109_70' : 0.83333334, 'a69_0' : 0.13333333})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'absent') and (THepatitis == 'present') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.05084746, 'a109_70' : 0.88135593, 'a69_0' : 0.06779661})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'absent') and (THepatitis == 'present') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.05, 'a109_70' : 0.9, 'a69_0' : 0.05})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'absent') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.05333333, 'a109_70' : 0.90666667, 'a69_0' : 0.04})
elif ((ChHepatitis == 'persistent') and (Cirrhosis == 'absent') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.08333333, 'a109_70' : 0.88888889, 'a69_0' : 0.02777778})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'decompensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.03448276, 'a109_70' : 0.81034483, 'a69_0' : 0.15517241})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'decompensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.01428571, 'a109_70' : 0.75714286, 'a69_0' : 0.22857143})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'decompensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.02173913, 'a109_70' : 0.75, 'a69_0' : 0.22826087})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'decompensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.00172117, 'a109_70' : 0.60240964, 'a69_0' : 0.39586919})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'compensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.01785714, 'a109_70' : 0.76785715, 'a69_0' : 0.21428571})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'compensate') and (THepatitis == 'present') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.01785714, 'a109_70' : 0.78571429, 'a69_0' : 0.19642857})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'compensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.02816901, 'a109_70' : 0.80281691, 'a69_0' : 0.16901408})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'compensate') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.00321543, 'a109_70' : 0.67524116, 'a69_0' : 0.32154341})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'absent') and (THepatitis == 'present') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.03508772, 'a109_70' : 0.84210526, 'a69_0' : 0.12280702})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'absent') and (THepatitis == 'present') and (Hyperbilirubinemia == 'absent')):
	inr ~= choice({'a200_110' : 0.03703704, 'a109_70' : 0.92592592, 'a69_0' : 0.03703704})
elif ((ChHepatitis == 'absent') and (Cirrhosis == 'absent') and (THepatitis == 'absent') and (Hyperbilirubinemia == 'present')):
	inr ~= choice({'a200_110' : 0.05357143, 'a109_70' : 0.89285714, 'a69_0' : 0.05357143})
else:
	inr ~= choice({'a200_110' : 0.065, 'a109_70' : 0.875, 'a69_0' : 0.06})


if ((Cirrhosis == 'decompensate')):
	irregular_liver ~= choice({'present' : 0.6034483, 'absent' : 0.3965517})
elif ((Cirrhosis == 'compensate')):
	irregular_liver ~= choice({'present' : 0.3548387, 'absent' : 0.6451613})
else:
	irregular_liver ~= choice({'present' : 0.1065574, 'absent' : 0.8934426})


if ((bilirubin == 'a88_20')):
	itching ~= choice({'present' : 0.875, 'absent' : 0.125})
elif ((bilirubin == 'a19_7')):
	itching ~= choice({'present' : 0.6865672, 'absent' : 0.3134328})
elif ((bilirubin == 'a6_2')):
	itching ~= choice({'present' : 0.5477387, 'absent' : 0.4522613})
else:
	itching ~= choice({'present' : 0.3333333, 'absent' : 0.6666667})


if ((bilirubin == 'a88_20')):
	jaundice ~= choice({'present' : 0.75, 'absent' : 0.25})
elif ((bilirubin == 'a19_7')):
	jaundice ~= choice({'present' : 0.5671642, 'absent' : 0.4328358})
elif ((bilirubin == 'a6_2')):
	jaundice ~= choice({'present' : 0.3467337, 'absent' : 0.6532663})
else:
	jaundice ~= choice({'present' : 0.1942446, 'absent' : 0.8057554})


if ((Cirrhosis == 'decompensate')):
	palms ~= choice({'present' : 0.5, 'absent' : 0.5})
elif ((Cirrhosis == 'compensate')):
	palms ~= choice({'present' : 0.2903226, 'absent' : 0.7096774})
else:
	palms ~= choice({'present' : 0.1409836, 'absent' : 0.8590164})


if ((RHepatitis == 'present') and (THepatitis == 'present') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.04166667, 'a699_240' : 0.29166667, 'a239_0' : 0.66666666})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.04347826, 'a699_240' : 0.30434783, 'a239_0' : 0.65217391})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.04166667, 'a699_240' : 0.33333333, 'a239_0' : 0.625})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Cirrhosis == 'compensate') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.03773585, 'a699_240' : 0.26415094, 'a239_0' : 0.69811321})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Cirrhosis == 'compensate') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.025, 'a699_240' : 0.25, 'a239_0' : 0.725})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Cirrhosis == 'compensate') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.02702703, 'a699_240' : 0.27027027, 'a239_0' : 0.7027027})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Cirrhosis == 'absent') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.03571429, 'a699_240' : 0.21428571, 'a239_0' : 0.75})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Cirrhosis == 'absent') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.02272727, 'a699_240' : 0.20454545, 'a239_0' : 0.77272728})
elif ((RHepatitis == 'present') and (THepatitis == 'present') and (Cirrhosis == 'absent') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.02325581, 'a699_240' : 0.18604651, 'a239_0' : 0.79069768})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.02898551, 'a699_240' : 0.28985507, 'a239_0' : 0.68115942})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.03389831, 'a699_240' : 0.3220339, 'a239_0' : 0.64406779})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.04545455, 'a699_240' : 0.37878788, 'a239_0' : 0.57575757})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Cirrhosis == 'compensate') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.02985075, 'a699_240' : 0.29850746, 'a239_0' : 0.67164179})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Cirrhosis == 'compensate') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.02040816, 'a699_240' : 0.28571429, 'a239_0' : 0.69387755})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Cirrhosis == 'compensate') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.02083333, 'a699_240' : 0.3125, 'a239_0' : 0.66666667})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Cirrhosis == 'absent') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.02597403, 'a699_240' : 0.24675325, 'a239_0' : 0.72727272})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Cirrhosis == 'absent') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.015625, 'a699_240' : 0.234375, 'a239_0' : 0.75})
elif ((RHepatitis == 'present') and (THepatitis == 'absent') and (Cirrhosis == 'absent') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.00584795, 'a699_240' : 0.23391813, 'a239_0' : 0.76023392})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.04761905, 'a699_240' : 0.28571429, 'a239_0' : 0.66666666})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.04918033, 'a699_240' : 0.31147541, 'a239_0' : 0.63934426})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.05555556, 'a699_240' : 0.34722222, 'a239_0' : 0.59722222})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Cirrhosis == 'compensate') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.04166667, 'a699_240' : 0.27777778, 'a239_0' : 0.68055555})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Cirrhosis == 'compensate') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.03703704, 'a699_240' : 0.25925926, 'a239_0' : 0.7037037})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Cirrhosis == 'compensate') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.05357143, 'a699_240' : 0.26785714, 'a239_0' : 0.67857143})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Cirrhosis == 'absent') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.04651163, 'a699_240' : 0.22093023, 'a239_0' : 0.73255814})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Cirrhosis == 'absent') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.04054054, 'a699_240' : 0.2027027, 'a239_0' : 0.75675676})
elif ((RHepatitis == 'absent') and (THepatitis == 'present') and (Cirrhosis == 'absent') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.07407407, 'a699_240' : 0.14814815, 'a239_0' : 0.77777778})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.04597701, 'a699_240' : 0.29885057, 'a239_0' : 0.65517242})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.04494382, 'a699_240' : 0.33707865, 'a239_0' : 0.61797753})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Cirrhosis == 'decompensate') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.06896552, 'a699_240' : 0.48275862, 'a239_0' : 0.44827586})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Cirrhosis == 'compensate') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.04494382, 'a699_240' : 0.33707865, 'a239_0' : 0.61797753})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Cirrhosis == 'compensate') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.03896104, 'a699_240' : 0.31168831, 'a239_0' : 0.64935065})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Cirrhosis == 'compensate') and (ChHepatitis == 'absent')):
	phosphatase ~= choice({'a4000_700' : 0.06451613, 'a699_240' : 0.38709677, 'a239_0' : 0.5483871})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Cirrhosis == 'absent') and (ChHepatitis == 'active')):
	phosphatase ~= choice({'a4000_700' : 0.03296703, 'a699_240' : 0.21978022, 'a239_0' : 0.74725275})
elif ((RHepatitis == 'absent') and (THepatitis == 'absent') and (Cirrhosis == 'absent') and (ChHepatitis == 'persistent')):
	phosphatase ~= choice({'a4000_700' : 0.02777778, 'a699_240' : 0.19444444, 'a239_0' : 0.77777778})
else:
	phosphatase ~= choice({'a4000_700' : 0.2118451, 'a699_240' : 0.3394077, 'a239_0' : 0.4487472})


if ((Cirrhosis == 'decompensate') and (PBC == 'present')):
	platelet ~= choice({'a597_300' : 0.06547619, 'a299_150' : 0.63690476, 'a149_100' : 0.17857143, 'a99_0' : 0.11904762})
elif ((Cirrhosis == 'decompensate') and (PBC == 'absent')):
	platelet ~= choice({'a597_300' : 0.06896552, 'a299_150' : 0.46551724, 'a149_100' : 0.27586207, 'a99_0' : 0.18965517})
elif ((Cirrhosis == 'compensate') and (PBC == 'present')):
	platelet ~= choice({'a597_300' : 0.06557377, 'a299_150' : 0.63934426, 'a149_100' : 0.17486339, 'a99_0' : 0.12021858})
elif ((Cirrhosis == 'compensate') and (PBC == 'absent')):
	platelet ~= choice({'a597_300' : 0.06451613, 'a299_150' : 0.64516129, 'a149_100' : 0.16129032, 'a99_0' : 0.12903226})
elif ((Cirrhosis == 'absent') and (PBC == 'present')):
	platelet ~= choice({'a597_300' : 0.06428571, 'a299_150' : 0.67142857, 'a149_100' : 0.15714286, 'a99_0' : 0.10714286})
else:
	platelet ~= choice({'a597_300' : 0.09393939, 'a299_150' : 0.73636364, 'a149_100' : 0.13939394, 'a99_0' : 0.03030303})


if ((Cirrhosis == 'decompensate')):
	proteins ~= choice({'a10_6' : 0.99827883, 'a5_2' : 0.00172117})
elif ((Cirrhosis == 'compensate')):
	proteins ~= choice({'a10_6' : 0.99678457, 'a5_2' : 0.00321543})
else:
	proteins ~= choice({'a10_6' : 0.98032787, 'a5_2' : 0.01967213})


if ((bilirubin == 'a88_20')):
	skin ~= choice({'present' : 0.99378882, 'absent' : 0.00621118})
elif ((bilirubin == 'a19_7')):
	skin ~= choice({'present' : 0.8955224, 'absent' : 0.1044776})
elif ((bilirubin == 'a6_2')):
	skin ~= choice({'present' : 0.7035176, 'absent' : 0.2964824})
else:
	skin ~= choice({'present' : 0.1822542, 'absent' : 0.8177458})


if ((Cirrhosis == 'decompensate')):
	spiders ~= choice({'present' : 0.6034483, 'absent' : 0.3965517})
elif ((Cirrhosis == 'compensate')):
	spiders ~= choice({'present' : 0.483871, 'absent' : 0.516129})
else:
	spiders ~= choice({'present' : 0.1836066, 'absent' : 0.8163934})


if ((Cirrhosis == 'decompensate') and (RHepatitis == 'present') and (THepatitis == 'present')):
	spleen ~= choice({'present' : 0.3235294, 'absent' : 0.6764706})
elif ((Cirrhosis == 'decompensate') and (RHepatitis == 'present') and (THepatitis == 'absent')):
	spleen ~= choice({'present' : 0.3703704, 'absent' : 0.6296296})
elif ((Cirrhosis == 'decompensate') and (RHepatitis == 'absent') and (THepatitis == 'present')):
	spleen ~= choice({'present' : 0.3623188, 'absent' : 0.6376812})
elif ((Cirrhosis == 'decompensate') and (RHepatitis == 'absent') and (THepatitis == 'absent')):
	spleen ~= choice({'present' : 0.4827586, 'absent' : 0.5172414})
elif ((Cirrhosis == 'compensate') and (RHepatitis == 'present') and (THepatitis == 'present')):
	spleen ~= choice({'present' : 0.3023256, 'absent' : 0.6976744})
elif ((Cirrhosis == 'compensate') and (RHepatitis == 'present') and (THepatitis == 'absent')):
	spleen ~= choice({'present' : 0.2444444, 'absent' : 0.7555556})
elif ((Cirrhosis == 'compensate') and (RHepatitis == 'absent') and (THepatitis == 'present')):
	spleen ~= choice({'present' : 0.2156863, 'absent' : 0.7843137})
elif ((Cirrhosis == 'compensate') and (RHepatitis == 'absent') and (THepatitis == 'absent')):
	spleen ~= choice({'present' : 0.2580645, 'absent' : 0.7419355})
elif ((Cirrhosis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'present')):
	spleen ~= choice({'present' : 0.1621622, 'absent' : 0.8378378})
elif ((Cirrhosis == 'absent') and (RHepatitis == 'present') and (THepatitis == 'absent')):
	spleen ~= choice({'present' : 0.1176471, 'absent' : 0.8823529})
elif ((Cirrhosis == 'absent') and (RHepatitis == 'absent') and (THepatitis == 'present')):
	spleen ~= choice({'present' : 0.1111111, 'absent' : 0.8888889})
else:
	spleen ~= choice({'present' : 0.1007067, 'absent' : 0.8992933})


if ((encephalopathy == 'present')):
	urea ~= choice({'a165_50' : 0.2173913, 'a49_40' : 0.1304348, 'a39_0' : 0.6521739})
else:
	urea ~= choice({'a165_50' : 0.03550296, 'a49_40' : 0.06508876, 'a39_0' : 0.89940828})


if ((proteins == 'a10_6')):
	ascites ~= choice({'present' : 0.1280932, 'absent' : 0.8719068})
else:
	ascites ~= choice({'present' : 0.5833333, 'absent' : 0.4166667})


if ((platelet == 'a597_300') and (inr == 'a200_110')):
	bleeding ~= choice({'present' : 0.1428571, 'absent' : 0.8571429})
elif ((platelet == 'a597_300') and (inr == 'a109_70')):
	bleeding ~= choice({'present' : 0.106383, 'absent' : 0.893617})
elif ((platelet == 'a597_300') and (inr == 'a69_0')):
	bleeding ~= choice({'present' : 0.09090909, 'absent' : 0.90909091})
elif ((platelet == 'a299_150') and (inr == 'a200_110')):
	bleeding ~= choice({'present' : 0.1304348, 'absent' : 0.8695652})
elif ((platelet == 'a299_150') and (inr == 'a109_70')):
	bleeding ~= choice({'present' : 0.1373494, 'absent' : 0.8626506})
elif ((platelet == 'a299_150') and (inr == 'a69_0')):
	bleeding ~= choice({'present' : 0.425, 'absent' : 0.575})
elif ((platelet == 'a149_100') and (inr == 'a200_110')):
	bleeding ~= choice({'present' : 0.2, 'absent' : 0.8})
elif ((platelet == 'a149_100') and (inr == 'a109_70')):
	bleeding ~= choice({'present' : 0.1333333, 'absent' : 0.8666667})
elif ((platelet == 'a149_100') and (inr == 'a69_0')):
	bleeding ~= choice({'present' : 0.25, 'absent' : 0.75})
elif ((platelet == 'a99_0') and (inr == 'a200_110')):
	bleeding ~= choice({'present' : 0.5, 'absent' : 0.5})
elif ((platelet == 'a99_0') and (inr == 'a109_70')):
	bleeding ~= choice({'present' : 0.255814, 'absent' : 0.744186})
else:
	bleeding ~= choice({'present' : 0.6666667, 'absent' : 0.3333333})


if ((encephalopathy == 'present')):
	consciousness ~= choice({'present' : 0.3043478, 'absent' : 0.6956522})
else:
	consciousness ~= choice({'present' : 0.01627219, 'absent' : 0.98372781})


if ((encephalopathy == 'present')):
	density ~= choice({'present' : 0.7391304, 'absent' : 0.2608696})
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
events = [flatulence << {'present'},surgery << {'absent'},Steatosis << {'absent'},nausea << {'absent'},transfusion << {'present'},palms << {'absent'},alcoholism << {'present'},carcinoma << {'absent'},RHepatitis << {'absent'},amylase << {'a499_300'},encephalopathy << {'absent'},ast << {'a399_150'},choledocholithotomy << {'present'},Hyperbilirubinemia << {'present'},density << {'present'},fatigue << {'absent'},sex << {'male'},le_cells << {'present'},pain_ruq << {'present'},urea << {'a49_40'},amylase << {'a299_0'},fat << {'present'},skin << {'absent'},alt << {'a34_0'},flatulence << {'present'},albumin << {'a70_50'},nausea << {'present'},fat << {'present'},le_cells << {'absent'},Hyperbilirubinemia << {'present'},ama << {'absent'},nausea << {'absent'},ChHepatitis << {'persistent'},phosphatase << {'a239_0'},injections << {'absent'},hepatalgia << {'present'},carcinoma << {'present'},hospital << {'absent'},diabetes << {'absent'},joints << {'present'},upper_pain << {'absent'},spiders << {'absent'},gallstones << {'present'},ChHepatitis << {'absent'},upper_pain << {'absent'},hbsag_anti << {'absent'},upper_pain << {'present'},proteins << {'a5_2'},fibrosis << {'present'},pain_ruq << {'present'},(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a39_0'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a19_7'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a850_200'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a39_0'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a149_40'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age65_100'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a19_7'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a19_7'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a39_0'}) & (bilirubin << {'a19_7'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a6_2'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a88_20'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a149_40'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age51_65'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a1_0'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a850_200'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'present'}) & (injections << {'present'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a99_35'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a9_0'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a1_0'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'absent'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age31_50'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a640_70'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'absent'}) & (ast << {'a39_0'}) & (bilirubin << {'a19_7'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'present'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a34_0'}) & (ama << {'absent'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a39_0'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'present'}),(ChHepatitis << {'active'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a69_30'}) & (hbc_anti << {'absent'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'present'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'absent'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a597_300'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a199_100'}) & (ama << {'present'}) & (amylase << {'a1400_500'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a399_150'}) & (bilirubin << {'a88_20'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a349_240'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a4000_700'}) & (platelet << {'a299_150'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a17_4'}) & (upper_pain << {'present'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a49_15'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'absent'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a499_300'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a700_400'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a349_240'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'present'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'present'}) & (itching << {'present'}) & (jaundice << {'absent'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'absent'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'present'}) & (phosphatase << {'a699_240'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'present'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'compensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age0_30'}) & (albumin << {'a29_0'}) & (alcohol << {'present'}) & (alcoholism << {'absent'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'present'}) & (ast << {'a39_0'}) & (bilirubin << {'a19_7'}) & (bleeding << {'absent'}) & (carcinoma << {'absent'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'present'}) & (density << {'present'}) & (diabetes << {'present'}) & (edema << {'present'}) & (edge << {'absent'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'present'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'present'}) & (hospital << {'absent'}) & (injections << {'present'}) & (inr << {'a200_110'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'present'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'absent'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a239_0'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'absent'}) & (proteins << {'a10_6'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a200_50'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'absent'}) & (age << {'age65_100'}) & (albumin << {'a49_30'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a850_200'}) & (ama << {'absent'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a149_40'}) & (bilirubin << {'a88_20'}) & (bleeding << {'absent'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a239_0'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'absent'}) & (edge << {'present'}) & (encephalopathy << {'absent'}) & (fat << {'absent'}) & (fatigue << {'absent'}) & (fibrosis << {'absent'}) & (flatulence << {'absent'}) & (gallstones << {'present'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'absent'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'absent'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'present'}) & (phosphatase << {'a239_0'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'male'}) & (skin << {'present'}) & (spiders << {'absent'}) & (spleen << {'present'}) & (surgery << {'present'}) & (transfusion << {'absent'}) & (triglycerides << {'a3_2'}) & (upper_pain << {'present'}) & (urea << {'a39_0'}) & (vh_amn << {'present'}),(ChHepatitis << {'absent'}) & (Cirrhosis << {'decompensate'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'absent'}) & (PBC << {'present'}) & (RHepatitis << {'absent'}) & (Steatosis << {'absent'}) & (THepatitis << {'present'}) & (age << {'age31_50'}) & (albumin << {'a70_50'}) & (alcohol << {'present'}) & (alcoholism << {'present'}) & (alt << {'a99_35'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'present'}) & (ascites << {'present'}) & (ast << {'a700_400'}) & (bilirubin << {'a6_2'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'absent'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'absent'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'absent'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'absent'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'present'}) & (hbsag << {'absent'}) & (hbsag_anti << {'present'}) & (hcv_anti << {'absent'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a109_70'}) & (irregular_liver << {'present'}) & (itching << {'absent'}) & (jaundice << {'absent'}) & (joints << {'present'}) & (le_cells << {'absent'}) & (nausea << {'present'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'present'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a149_100'}) & (pressure_ruq << {'present'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'present'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'absent'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a165_50'}) & (vh_amn << {'absent'}),(ChHepatitis << {'persistent'}) & (Cirrhosis << {'absent'}) & (ESR << {'a14_0'}) & (Hyperbilirubinemia << {'present'}) & (PBC << {'absent'}) & (RHepatitis << {'present'}) & (Steatosis << {'present'}) & (THepatitis << {'present'}) & (age << {'age51_65'}) & (albumin << {'a49_30'}) & (alcohol << {'absent'}) & (alcoholism << {'absent'}) & (alt << {'a34_0'}) & (ama << {'present'}) & (amylase << {'a299_0'}) & (anorexia << {'absent'}) & (ascites << {'absent'}) & (ast << {'a399_150'}) & (bilirubin << {'a19_7'}) & (bleeding << {'present'}) & (carcinoma << {'present'}) & (choledocholithotomy << {'present'}) & (cholesterol << {'a999_350'}) & (consciousness << {'absent'}) & (density << {'present'}) & (diabetes << {'absent'}) & (edema << {'present'}) & (edge << {'present'}) & (encephalopathy << {'present'}) & (fat << {'present'}) & (fatigue << {'present'}) & (fibrosis << {'present'}) & (flatulence << {'present'}) & (gallstones << {'absent'}) & (ggtp << {'a29_10'}) & (hbc_anti << {'present'}) & (hbeag << {'absent'}) & (hbsag << {'absent'}) & (hbsag_anti << {'absent'}) & (hcv_anti << {'present'}) & (hepatalgia << {'absent'}) & (hepatomegaly << {'absent'}) & (hepatotoxic << {'absent'}) & (hospital << {'present'}) & (injections << {'absent'}) & (inr << {'a69_0'}) & (irregular_liver << {'absent'}) & (itching << {'present'}) & (jaundice << {'present'}) & (joints << {'absent'}) & (le_cells << {'absent'}) & (nausea << {'absent'}) & (obesity << {'present'}) & (pain << {'present'}) & (pain_ruq << {'absent'}) & (palms << {'absent'}) & (phosphatase << {'a699_240'}) & (platelet << {'a99_0'}) & (pressure_ruq << {'absent'}) & (proteins << {'a5_2'}) & (sex << {'female'}) & (skin << {'absent'}) & (spiders << {'present'}) & (spleen << {'absent'}) & (surgery << {'absent'}) & (transfusion << {'present'}) & (triglycerides << {'a1_0'}) & (upper_pain << {'absent'}) & (urea << {'a49_40'}) & (vh_amn << {'absent'})]
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
