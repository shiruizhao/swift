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
Age ~= choice({'Adolescent' : 0.2,'Adult' : 0.6,'Senior' : 0.2})


Mileage ~= choice({'FiveThou' : 0.1,'TwentyThou' : 0.4,'FiftyThou' : 0.4,'Domino' : 0.1})


if (Age == 'Adolescent'):
	SocioEcon ~= choice({'Prole' : 0.4, 'Middle' : 0.4, 'UpperMiddle' : 0.19, 'Wealthy' : 0.010000000000000009})
elif(Age == 'Adult'):
	SocioEcon ~= choice({'Prole' : 0.4, 'Middle' : 0.4, 'UpperMiddle' : 0.19, 'Wealthy' : 0.010000000000000009})
else:
	SocioEcon ~= choice({'Prole' : 0.5, 'Middle' : 0.2, 'UpperMiddle' : 0.29, 'Wealthy' : 0.010000000000000009})


if (SocioEcon == 'Prole'):
	if (Age == 'Adolescent'):
		GoodStudent ~= choice({'True' : 0.1, 'False' : 0.9})
	elif(Age == 'Adult'):
		GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
	else:
		GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
elif(SocioEcon == 'Middle'):
	if(Age == 'Adolescent'):
		GoodStudent ~= choice({'True' : 0.2, 'False' : 0.8})
	elif(Age == 'Adult'):
		GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
	else:
		GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
elif(SocioEcon == 'UpperMiddle'):
	if(Age == 'Adolescent'):
		GoodStudent ~= choice({'True' : 0.5, 'False' : 0.5})
	elif(Age == 'Adult'):
		GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
	else:
		GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
else:
	if(Age == 'Adolescent'):
		GoodStudent ~= choice({'True' : 0.4, 'False' : 0.6})
	elif(Age == 'Adult'):
		GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
	else:
		GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})


if (SocioEcon == 'Prole'):
	OtherCar ~= choice({'True' : 0.5, 'False' : 0.5})
elif(SocioEcon == 'Middle'):
	OtherCar ~= choice({'True' : 0.8, 'False' : 0.19999999999999996})
elif(SocioEcon == 'UpperMiddle'):
	OtherCar ~= choice({'True' : 0.9, 'False' : 0.09999999999999998})
else:
	OtherCar ~= choice({'True' : 0.95, 'False' : 0.050000000000000044})


if (Age == 'Adolescent'):
	if (SocioEcon == 'Prole'):
		RiskAversion ~= choice({'Psychopath' : 0.02, 'Adventurous' : 0.58, 'Normal' : 0.3, 'Cautious' : 0.10000000000000009})
	elif(SocioEcon == 'Middle'):
		RiskAversion ~= choice({'Psychopath' : 0.02, 'Adventurous' : 0.38, 'Normal' : 0.5, 'Cautious' : 0.09999999999999998})
	elif(SocioEcon == 'UpperMiddle'):
		RiskAversion ~= choice({'Psychopath' : 0.02, 'Adventurous' : 0.48, 'Normal' : 0.4, 'Cautious' : 0.09999999999999998})
	else:
		RiskAversion ~= choice({'Psychopath' : 0.02, 'Adventurous' : 0.58, 'Normal' : 0.3, 'Cautious' : 0.10000000000000009})
elif(Age == 'Adult'):
	if(SocioEcon == 'Prole'):
		RiskAversion ~= choice({'Psychopath' : 0.015, 'Adventurous' : 0.285, 'Normal' : 0.5, 'Cautious' : 0.19999999999999996})
	elif(SocioEcon == 'Middle'):
		RiskAversion ~= choice({'Psychopath' : 0.015, 'Adventurous' : 0.185, 'Normal' : 0.6, 'Cautious' : 0.19999999999999996})
	elif(SocioEcon == 'UpperMiddle'):
		RiskAversion ~= choice({'Psychopath' : 0.015, 'Adventurous' : 0.285, 'Normal' : 0.5, 'Cautious' : 0.19999999999999996})
	else:
		RiskAversion ~= choice({'Psychopath' : 0.015, 'Adventurous' : 0.285, 'Normal' : 0.4, 'Cautious' : 0.30000000000000004})
else:
	if(SocioEcon == 'Prole'):
		RiskAversion ~= choice({'Psychopath' : 0.01, 'Adventurous' : 0.09, 'Normal' : 0.4, 'Cautious' : 0.5})
	elif(SocioEcon == 'Middle'):
		RiskAversion ~= choice({'Psychopath' : 0.01, 'Adventurous' : 0.04, 'Normal' : 0.35, 'Cautious' : 0.6000000000000001})
	elif(SocioEcon == 'UpperMiddle'):
		RiskAversion ~= choice({'Psychopath' : 0.01, 'Adventurous' : 0.09, 'Normal' : 0.4, 'Cautious' : 0.5})
	else:
		RiskAversion ~= choice({'Psychopath' : 0.01, 'Adventurous' : 0.09, 'Normal' : 0.4, 'Cautious' : 0.5})


if (Age == 'Adolescent'):
	if (RiskAversion == 'Psychopath'):
		SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
	elif(RiskAversion == 'Adventurous'):
		SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
	elif(RiskAversion == 'Normal'):
		SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
	else:
		SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
elif(Age == 'Adult'):
	if(RiskAversion == 'Psychopath'):
		SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
	elif(RiskAversion == 'Adventurous'):
		SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
	elif(RiskAversion == 'Normal'):
		SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
	else:
		SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
else:
	if(RiskAversion == 'Psychopath'):
		SeniorTrain ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(RiskAversion == 'Adventurous'):
		SeniorTrain ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(RiskAversion == 'Normal'):
		SeniorTrain ~= choice({'True' : 0.3, 'False' : 0.7})
	else:
		SeniorTrain ~= choice({'True' : 0.9, 'False' : 0.09999999999999998})


if (SocioEcon == 'Prole'):
	if (RiskAversion == 'Psychopath'):
		VehicleYear ~= choice({'Current' : 0.15, 'Older' : 0.85})
	elif(RiskAversion == 'Adventurous'):
		VehicleYear ~= choice({'Current' : 0.15, 'Older' : 0.85})
	elif(RiskAversion == 'Normal'):
		VehicleYear ~= choice({'Current' : 0.15, 'Older' : 0.85})
	else:
		VehicleYear ~= choice({'Current' : 0.15, 'Older' : 0.85})
elif(SocioEcon == 'Middle'):
	if(RiskAversion == 'Psychopath'):
		VehicleYear ~= choice({'Current' : 0.3, 'Older' : 0.7})
	elif(RiskAversion == 'Adventurous'):
		VehicleYear ~= choice({'Current' : 0.3, 'Older' : 0.7})
	elif(RiskAversion == 'Normal'):
		VehicleYear ~= choice({'Current' : 0.3, 'Older' : 0.7})
	else:
		VehicleYear ~= choice({'Current' : 0.3, 'Older' : 0.7})
elif(SocioEcon == 'UpperMiddle'):
	if(RiskAversion == 'Psychopath'):
		VehicleYear ~= choice({'Current' : 0.8, 'Older' : 0.19999999999999996})
	elif(RiskAversion == 'Adventurous'):
		VehicleYear ~= choice({'Current' : 0.8, 'Older' : 0.19999999999999996})
	elif(RiskAversion == 'Normal'):
		VehicleYear ~= choice({'Current' : 0.8, 'Older' : 0.19999999999999996})
	else:
		VehicleYear ~= choice({'Current' : 0.8, 'Older' : 0.19999999999999996})
else:
	if(RiskAversion == 'Psychopath'):
		VehicleYear ~= choice({'Current' : 0.9, 'Older' : 0.09999999999999998})
	elif(RiskAversion == 'Adventurous'):
		VehicleYear ~= choice({'Current' : 0.9, 'Older' : 0.09999999999999998})
	elif(RiskAversion == 'Normal'):
		VehicleYear ~= choice({'Current' : 0.9, 'Older' : 0.09999999999999998})
	else:
		VehicleYear ~= choice({'Current' : 0.9, 'Older' : 0.09999999999999998})


if (RiskAversion == 'Psychopath'):
	if (SocioEcon == 'Prole'):
		AntiTheft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(SocioEcon == 'Middle'):
		AntiTheft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(SocioEcon == 'UpperMiddle'):
		AntiTheft ~= choice({'True' : 0.05, 'False' : 0.95})
	else:
		AntiTheft ~= choice({'True' : 0.5, 'False' : 0.5})
elif(RiskAversion == 'Adventurous'):
	if(SocioEcon == 'Prole'):
		AntiTheft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(SocioEcon == 'Middle'):
		AntiTheft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(SocioEcon == 'UpperMiddle'):
		AntiTheft ~= choice({'True' : 0.2, 'False' : 0.8})
	else:
		AntiTheft ~= choice({'True' : 0.5, 'False' : 0.5})
elif(RiskAversion == 'Normal'):
	if(SocioEcon == 'Prole'):
		AntiTheft ~= choice({'True' : 0.1, 'False' : 0.9})
	elif(SocioEcon == 'Middle'):
		AntiTheft ~= choice({'True' : 0.3, 'False' : 0.7})
	elif(SocioEcon == 'UpperMiddle'):
		AntiTheft ~= choice({'True' : 0.9, 'False' : 0.09999999999999998})
	else:
		AntiTheft ~= choice({'True' : 0.8, 'False' : 0.19999999999999996})
else:
	if(SocioEcon == 'Prole'):
		AntiTheft ~= choice({'True' : 0.95, 'False' : 0.050000000000000044})
	elif(SocioEcon == 'Middle'):
		AntiTheft ~= choice({'True' : 0.999999, 'False' : 1.0000000000287557e-06})
	elif(SocioEcon == 'UpperMiddle'):
		AntiTheft ~= choice({'True' : 0.999999, 'False' : 1.0000000000287557e-06})
	else:
		AntiTheft ~= choice({'True' : 0.999999, 'False' : 1.0000000000287557e-06})


if (Age == 'Adolescent'):
	if (SeniorTrain == 'True'):
		DrivingSkill ~= choice({'SubStandard' : 0.5, 'Normal' : 0.45, 'Expert' : 0.050000000000000044})
	else:
		DrivingSkill ~= choice({'SubStandard' : 0.5, 'Normal' : 0.45, 'Expert' : 0.050000000000000044})
elif(Age == 'Adult'):
	if(SeniorTrain == 'True'):
		DrivingSkill ~= choice({'SubStandard' : 0.3, 'Normal' : 0.6, 'Expert' : 0.10000000000000009})
	else:
		DrivingSkill ~= choice({'SubStandard' : 0.3, 'Normal' : 0.6, 'Expert' : 0.10000000000000009})
else:
	if(SeniorTrain == 'True'):
		DrivingSkill ~= choice({'SubStandard' : 0.1, 'Normal' : 0.6, 'Expert' : 0.30000000000000004})
	else:
		DrivingSkill ~= choice({'SubStandard' : 0.4, 'Normal' : 0.5, 'Expert' : 0.09999999999999998})


if (RiskAversion == 'Psychopath'):
	if (SocioEcon == 'Prole'):
		HomeBase ~= choice({'Secure' : 1e-06, 'City' : 0.8, 'Suburb' : 0.049999, 'Rural' : 0.1499999999999999})
	elif(SocioEcon == 'Middle'):
		HomeBase ~= choice({'Secure' : 0.15, 'City' : 0.8, 'Suburb' : 0.04, 'Rural' : 0.009999999999999898})
	elif(SocioEcon == 'UpperMiddle'):
		HomeBase ~= choice({'Secure' : 0.35, 'City' : 0.6, 'Suburb' : 0.04, 'Rural' : 0.010000000000000009})
	else:
		HomeBase ~= choice({'Secure' : 0.489999, 'City' : 0.5, 'Suburb' : 1e-06, 'Rural' : 0.009999999999999898})
elif(RiskAversion == 'Adventurous'):
	if(SocioEcon == 'Prole'):
		HomeBase ~= choice({'Secure' : 1e-06, 'City' : 0.8, 'Suburb' : 0.05, 'Rural' : 0.14999899999999988})
	elif(SocioEcon == 'Middle'):
		HomeBase ~= choice({'Secure' : 0.01, 'City' : 0.25, 'Suburb' : 0.6, 'Rural' : 0.14})
	elif(SocioEcon == 'UpperMiddle'):
		HomeBase ~= choice({'Secure' : 0.2, 'City' : 0.4, 'Suburb' : 0.3, 'Rural' : 0.09999999999999987})
	else:
		HomeBase ~= choice({'Secure' : 0.95, 'City' : 1e-06, 'Suburb' : 1e-06, 'Rural' : 0.04999799999999999})
elif(RiskAversion == 'Normal'):
	if(SocioEcon == 'Prole'):
		HomeBase ~= choice({'Secure' : 1e-06, 'City' : 0.8, 'Suburb' : 0.05, 'Rural' : 0.14999899999999988})
	elif(SocioEcon == 'Middle'):
		HomeBase ~= choice({'Secure' : 0.299999, 'City' : 1e-06, 'Suburb' : 0.6, 'Rural' : 0.10000000000000009})
	elif(SocioEcon == 'UpperMiddle'):
		HomeBase ~= choice({'Secure' : 0.5, 'City' : 1e-06, 'Suburb' : 0.4, 'Rural' : 0.09999899999999995})
	else:
		HomeBase ~= choice({'Secure' : 0.85, 'City' : 1e-06, 'Suburb' : 0.001, 'Rural' : 0.148999})
else:
	if(SocioEcon == 'Prole'):
		HomeBase ~= choice({'Secure' : 1e-06, 'City' : 0.8, 'Suburb' : 0.05, 'Rural' : 0.14999899999999988})
	elif(SocioEcon == 'Middle'):
		HomeBase ~= choice({'Secure' : 0.95, 'City' : 1e-06, 'Suburb' : 0.024445, 'Rural' : 0.025553999999999966})
	elif(SocioEcon == 'UpperMiddle'):
		HomeBase ~= choice({'Secure' : 0.999997, 'City' : 1e-06, 'Suburb' : 1e-06, 'Rural' : 9.999999999177334e-07})
	else:
		HomeBase ~= choice({'Secure' : 0.999997, 'City' : 1e-06, 'Suburb' : 1e-06, 'Rural' : 9.999999999177334e-07})


if (SocioEcon == 'Prole'):
	if (RiskAversion == 'Psychopath'):
		MakeModel ~= choice({'SportsCar' : 0.1, 'Economy' : 0.7, 'FamilySedan' : 0.2, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
	elif(RiskAversion == 'Adventurous'):
		MakeModel ~= choice({'SportsCar' : 0.1, 'Economy' : 0.7, 'FamilySedan' : 0.2, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
	elif(RiskAversion == 'Normal'):
		MakeModel ~= choice({'SportsCar' : 0.1, 'Economy' : 0.7, 'FamilySedan' : 0.2, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
	else:
		MakeModel ~= choice({'SportsCar' : 0.1, 'Economy' : 0.7, 'FamilySedan' : 0.2, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif(SocioEcon == 'Middle'):
	if(RiskAversion == 'Psychopath'):
		MakeModel ~= choice({'SportsCar' : 0.15, 'Economy' : 0.2, 'FamilySedan' : 0.65, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
	elif(RiskAversion == 'Adventurous'):
		MakeModel ~= choice({'SportsCar' : 0.15, 'Economy' : 0.2, 'FamilySedan' : 0.65, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
	elif(RiskAversion == 'Normal'):
		MakeModel ~= choice({'SportsCar' : 0.15, 'Economy' : 0.2, 'FamilySedan' : 0.65, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
	else:
		MakeModel ~= choice({'SportsCar' : 0.15, 'Economy' : 0.2, 'FamilySedan' : 0.65, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif(SocioEcon == 'UpperMiddle'):
	if(RiskAversion == 'Psychopath'):
		MakeModel ~= choice({'SportsCar' : 0.2, 'Economy' : 0.05, 'FamilySedan' : 0.3, 'Luxury' : 0.45, 'SuperLuxury' : 0.0})
	elif(RiskAversion == 'Adventurous'):
		MakeModel ~= choice({'SportsCar' : 0.2, 'Economy' : 0.05, 'FamilySedan' : 0.3, 'Luxury' : 0.45, 'SuperLuxury' : 0.0})
	elif(RiskAversion == 'Normal'):
		MakeModel ~= choice({'SportsCar' : 0.2, 'Economy' : 0.05, 'FamilySedan' : 0.3, 'Luxury' : 0.45, 'SuperLuxury' : 0.0})
	else:
		MakeModel ~= choice({'SportsCar' : 0.2, 'Economy' : 0.05, 'FamilySedan' : 0.3, 'Luxury' : 0.45, 'SuperLuxury' : 0.0})
else:
	if(RiskAversion == 'Psychopath'):
		MakeModel ~= choice({'SportsCar' : 0.3, 'Economy' : 0.01, 'FamilySedan' : 0.09, 'Luxury' : 0.4, 'SuperLuxury' : 0.19999999999999996})
	elif(RiskAversion == 'Adventurous'):
		MakeModel ~= choice({'SportsCar' : 0.3, 'Economy' : 0.01, 'FamilySedan' : 0.09, 'Luxury' : 0.4, 'SuperLuxury' : 0.19999999999999996})
	elif(RiskAversion == 'Normal'):
		MakeModel ~= choice({'SportsCar' : 0.3, 'Economy' : 0.01, 'FamilySedan' : 0.09, 'Luxury' : 0.4, 'SuperLuxury' : 0.19999999999999996})
	else:
		MakeModel ~= choice({'SportsCar' : 0.3, 'Economy' : 0.01, 'FamilySedan' : 0.09, 'Luxury' : 0.4, 'SuperLuxury' : 0.19999999999999996})


if (MakeModel == 'SportsCar'):
	if (VehicleYear == 'Current'):
		RuggedAuto ~= choice({'EggShell' : 0.95, 'Football' : 0.04, 'Tank' : 0.010000000000000009})
	else:
		RuggedAuto ~= choice({'EggShell' : 0.95, 'Football' : 0.04, 'Tank' : 0.010000000000000009})
elif(MakeModel == 'Economy'):
	if(VehicleYear == 'Current'):
		RuggedAuto ~= choice({'EggShell' : 0.5, 'Football' : 0.5, 'Tank' : 0.0})
	else:
		RuggedAuto ~= choice({'EggShell' : 0.9, 'Football' : 0.1, 'Tank' : 0.0})
elif(MakeModel == 'FamilySedan'):
	if(VehicleYear == 'Current'):
		RuggedAuto ~= choice({'EggShell' : 0.2, 'Football' : 0.6, 'Tank' : 0.19999999999999996})
	else:
		RuggedAuto ~= choice({'EggShell' : 0.05, 'Football' : 0.55, 'Tank' : 0.3999999999999999})
elif(MakeModel == 'Luxury'):
	if(VehicleYear == 'Current'):
		RuggedAuto ~= choice({'EggShell' : 0.1, 'Football' : 0.6, 'Tank' : 0.30000000000000004})
	else:
		RuggedAuto ~= choice({'EggShell' : 0.1, 'Football' : 0.6, 'Tank' : 0.30000000000000004})
else:
	if(VehicleYear == 'Current'):
		RuggedAuto ~= choice({'EggShell' : 0.05, 'Football' : 0.55, 'Tank' : 0.3999999999999999})
	else:
		RuggedAuto ~= choice({'EggShell' : 0.05, 'Football' : 0.55, 'Tank' : 0.3999999999999999})


if (MakeModel == 'SportsCar'):
	if (VehicleYear == 'Current'):
		Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
	else:
		Airbag ~= choice({'True' : 0.1, 'False' : 0.9})
elif(MakeModel == 'Economy'):
	if(VehicleYear == 'Current'):
		Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
	else:
		Airbag ~= choice({'True' : 0.05, 'False' : 0.95})
elif(MakeModel == 'FamilySedan'):
	if(VehicleYear == 'Current'):
		Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
	else:
		Airbag ~= choice({'True' : 0.2, 'False' : 0.8})
elif(MakeModel == 'Luxury'):
	if(VehicleYear == 'Current'):
		Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
	else:
		Airbag ~= choice({'True' : 0.6, 'False' : 0.4})
else:
	if(VehicleYear == 'Current'):
		Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
	else:
		Airbag ~= choice({'True' : 0.1, 'False' : 0.9})


if (MakeModel == 'SportsCar'):
	if (VehicleYear == 'Current'):
		Antilock ~= choice({'True' : 0.9, 'False' : 0.09999999999999998})
	else:
		Antilock ~= choice({'True' : 0.1, 'False' : 0.9})
elif(MakeModel == 'Economy'):
	if(VehicleYear == 'Current'):
		Antilock ~= choice({'True' : 0.001, 'False' : 0.999})
	else:
		Antilock ~= choice({'True' : 0.0, 'False' : 1.0})
elif(MakeModel == 'FamilySedan'):
	if(VehicleYear == 'Current'):
		Antilock ~= choice({'True' : 0.4, 'False' : 0.6})
	else:
		Antilock ~= choice({'True' : 0.0, 'False' : 1.0})
elif(MakeModel == 'Luxury'):
	if(VehicleYear == 'Current'):
		Antilock ~= choice({'True' : 0.99, 'False' : 0.010000000000000009})
	else:
		Antilock ~= choice({'True' : 0.3, 'False' : 0.7})
else:
	if(VehicleYear == 'Current'):
		Antilock ~= choice({'True' : 0.99, 'False' : 0.010000000000000009})
	else:
		Antilock ~= choice({'True' : 0.15, 'False' : 0.85})


if (MakeModel == 'SportsCar'):
	if (VehicleYear == 'Current'):
		if (Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.8, 'FiftyThou' : 0.09, 'Million' : 0.010000000000000009})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.8, 'FiftyThou' : 0.09, 'Million' : 0.010000000000000009})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.8, 'FiftyThou' : 0.09, 'Million' : 0.010000000000000009})
		else:
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.8, 'FiftyThou' : 0.09, 'Million' : 0.010000000000000009})
	else:
		if(Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 0.03, 'TenThou' : 0.3, 'TwentyThou' : 0.6, 'FiftyThou' : 0.06, 'Million' : 0.010000000000000009})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 0.16, 'TenThou' : 0.5, 'TwentyThou' : 0.3, 'FiftyThou' : 0.03, 'Million' : 0.010000000000000009})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 0.4, 'TenThou' : 0.47, 'TwentyThou' : 0.1, 'FiftyThou' : 0.02, 'Million' : 0.010000000000000009})
		else:
			CarValue ~= choice({'FiveThou' : 0.9, 'TenThou' : 0.06, 'TwentyThou' : 0.02, 'FiftyThou' : 0.01, 'Million' : 0.010000000000000009})
elif(MakeModel == 'Economy'):
	if(VehicleYear == 'Current'):
		if(Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.8, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 0.0})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.8, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 0.0})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.8, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 0.0})
		else:
			CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.8, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 0.0})
	else:
		if(Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 0.25, 'TenThou' : 0.7, 'TwentyThou' : 0.05, 'FiftyThou' : 0.0, 'Million' : 0.0})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 0.7, 'TenThou' : 0.2999, 'TwentyThou' : 0.0001, 'FiftyThou' : 0.0, 'Million' : 0.0})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 0.99, 'TenThou' : 0.009999, 'TwentyThou' : 1e-06, 'FiftyThou' : 0.0, 'Million' : 0.0})
		else:
			CarValue ~= choice({'FiveThou' : 0.999998, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif(MakeModel == 'FamilySedan'):
	if(VehicleYear == 'Current'):
		if(Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.9, 'FiftyThou' : 0.0, 'Million' : 0.0})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.9, 'FiftyThou' : 0.0, 'Million' : 0.0})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.9, 'FiftyThou' : 0.0, 'Million' : 0.0})
		else:
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.9, 'FiftyThou' : 0.0, 'Million' : 0.0})
	else:
		if(Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 0.2, 'TenThou' : 0.3, 'TwentyThou' : 0.5, 'FiftyThou' : 0.0, 'Million' : 0.0})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 0.5, 'TenThou' : 0.3, 'TwentyThou' : 0.2, 'FiftyThou' : 0.0, 'Million' : 0.0})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 0.7, 'TenThou' : 0.2, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 1.1102230246251565e-16})
		else:
			CarValue ~= choice({'FiveThou' : 0.99, 'TenThou' : 0.009999, 'TwentyThou' : 1e-06, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif(MakeModel == 'Luxury'):
	if(VehicleYear == 'Current'):
		if(Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 1.0, 'Million' : 0.0})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 1.0, 'Million' : 0.0})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 1.0, 'Million' : 0.0})
		else:
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 1.0, 'Million' : 0.0})
	else:
		if(Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 0.01, 'TenThou' : 0.09, 'TwentyThou' : 0.2, 'FiftyThou' : 0.7, 'Million' : 0.0})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 0.05, 'TenThou' : 0.15, 'TwentyThou' : 0.3, 'FiftyThou' : 0.5, 'Million' : 0.0})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.3, 'TwentyThou' : 0.3, 'FiftyThou' : 0.3, 'Million' : 0.0})
		else:
			CarValue ~= choice({'FiveThou' : 0.2, 'TenThou' : 0.2, 'TwentyThou' : 0.3, 'FiftyThou' : 0.3, 'Million' : 0.0})
else:
	if(VehicleYear == 'Current'):
		if(Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 0.0, 'Million' : 1.0})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 0.0, 'Million' : 1.0})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 0.0, 'Million' : 1.0})
		else:
			CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 0.0, 'Million' : 1.0})
	else:
		if(Mileage == 'FiveThou'):
			CarValue ~= choice({'FiveThou' : 1e-06, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 1e-06, 'Million' : 0.999996})
		elif(Mileage == 'TwentyThou'):
			CarValue ~= choice({'FiveThou' : 1e-06, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 1e-06, 'Million' : 0.999996})
		elif(Mileage == 'FiftyThou'):
			CarValue ~= choice({'FiveThou' : 1e-06, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 1e-06, 'Million' : 0.999996})
		else:
			CarValue ~= choice({'FiveThou' : 1e-06, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 1e-06, 'Million' : 0.999996})


if (RuggedAuto == 'EggShell'):
	if (Airbag == 'True'):
		Cushioning ~= choice({'Poor' : 0.5, 'Fair' : 0.3, 'Good' : 0.2, 'Excellent' : 0.0})
	else:
		Cushioning ~= choice({'Poor' : 0.7, 'Fair' : 0.3, 'Good' : 0.0, 'Excellent' : 0.0})
elif(RuggedAuto == 'Football'):
	if(Airbag == 'True'):
		Cushioning ~= choice({'Poor' : 0.0, 'Fair' : 0.1, 'Good' : 0.6, 'Excellent' : 0.30000000000000004})
	else:
		Cushioning ~= choice({'Poor' : 0.1, 'Fair' : 0.6, 'Good' : 0.3, 'Excellent' : 0.0})
else:
	if(Airbag == 'True'):
		Cushioning ~= choice({'Poor' : 0.0, 'Fair' : 0.0, 'Good' : 0.0, 'Excellent' : 1.0})
	else:
		Cushioning ~= choice({'Poor' : 0.0, 'Fair' : 0.0, 'Good' : 0.7, 'Excellent' : 0.30000000000000004})


if (DrivingSkill == 'SubStandard'):
	if (RiskAversion == 'Psychopath'):
		DrivHist ~= choice({'Zero' : 0.001, 'One' : 0.004, 'Many' : 0.995})
	elif(RiskAversion == 'Adventurous'):
		DrivHist ~= choice({'Zero' : 0.002, 'One' : 0.008, 'Many' : 0.99})
	elif(RiskAversion == 'Normal'):
		DrivHist ~= choice({'Zero' : 0.03, 'One' : 0.15, 'Many' : 0.8200000000000001})
	else:
		DrivHist ~= choice({'Zero' : 0.3, 'One' : 0.3, 'Many' : 0.4})
elif(DrivingSkill == 'Normal'):
	if(RiskAversion == 'Psychopath'):
		DrivHist ~= choice({'Zero' : 0.1, 'One' : 0.3, 'Many' : 0.6})
	elif(RiskAversion == 'Adventurous'):
		DrivHist ~= choice({'Zero' : 0.5, 'One' : 0.3, 'Many' : 0.19999999999999996})
	elif(RiskAversion == 'Normal'):
		DrivHist ~= choice({'Zero' : 0.9, 'One' : 0.07, 'Many' : 0.030000000000000027})
	else:
		DrivHist ~= choice({'Zero' : 0.95, 'One' : 0.04, 'Many' : 0.010000000000000009})
else:
	if(RiskAversion == 'Psychopath'):
		DrivHist ~= choice({'Zero' : 0.3, 'One' : 0.3, 'Many' : 0.4})
	elif(RiskAversion == 'Adventurous'):
		DrivHist ~= choice({'Zero' : 0.6, 'One' : 0.3, 'Many' : 0.10000000000000009})
	elif(RiskAversion == 'Normal'):
		DrivHist ~= choice({'Zero' : 0.99, 'One' : 0.009999, 'Many' : 1.0000000000287557e-06})
	else:
		DrivHist ~= choice({'Zero' : 0.999998, 'One' : 1e-06, 'Many' : 9.999999999177334e-07})


if (DrivingSkill == 'SubStandard'):
	if (RiskAversion == 'Psychopath'):
		DrivQuality ~= choice({'Poor' : 1.0, 'Normal' : 0.0, 'Excellent' : 0.0})
	elif(RiskAversion == 'Adventurous'):
		DrivQuality ~= choice({'Poor' : 1.0, 'Normal' : 0.0, 'Excellent' : 0.0})
	elif(RiskAversion == 'Normal'):
		DrivQuality ~= choice({'Poor' : 1.0, 'Normal' : 0.0, 'Excellent' : 0.0})
	else:
		DrivQuality ~= choice({'Poor' : 1.0, 'Normal' : 0.0, 'Excellent' : 0.0})
elif(DrivingSkill == 'Normal'):
	if(RiskAversion == 'Psychopath'):
		DrivQuality ~= choice({'Poor' : 0.5, 'Normal' : 0.2, 'Excellent' : 0.30000000000000004})
	elif(RiskAversion == 'Adventurous'):
		DrivQuality ~= choice({'Poor' : 0.3, 'Normal' : 0.4, 'Excellent' : 0.30000000000000004})
	elif(RiskAversion == 'Normal'):
		DrivQuality ~= choice({'Poor' : 0.0, 'Normal' : 1.0, 'Excellent' : 0.0})
	else:
		DrivQuality ~= choice({'Poor' : 0.0, 'Normal' : 0.8, 'Excellent' : 0.19999999999999996})
else:
	if(RiskAversion == 'Psychopath'):
		DrivQuality ~= choice({'Poor' : 0.3, 'Normal' : 0.2, 'Excellent' : 0.5})
	elif(RiskAversion == 'Adventurous'):
		DrivQuality ~= choice({'Poor' : 0.01, 'Normal' : 0.01, 'Excellent' : 0.98})
	elif(RiskAversion == 'Normal'):
		DrivQuality ~= choice({'Poor' : 0.0, 'Normal' : 0.0, 'Excellent' : 1.0})
	else:
		DrivQuality ~= choice({'Poor' : 0.0, 'Normal' : 0.0, 'Excellent' : 1.0})


if (AntiTheft == 'True'):
	if (HomeBase == 'Secure'):
		if (CarValue == 'FiveThou'):
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
		elif(CarValue == 'TenThou'):
			Theft ~= choice({'True' : 2e-06, 'False' : 0.999998})
		elif(CarValue == 'TwentyThou'):
			Theft ~= choice({'True' : 3e-06, 'False' : 0.999997})
		elif(CarValue == 'FiftyThou'):
			Theft ~= choice({'True' : 2e-06, 'False' : 0.999998})
		else:
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(HomeBase == 'City'):
		if(CarValue == 'FiveThou'):
			Theft ~= choice({'True' : 0.0005, 'False' : 0.9995})
		elif(CarValue == 'TenThou'):
			Theft ~= choice({'True' : 0.002, 'False' : 0.998})
		elif(CarValue == 'TwentyThou'):
			Theft ~= choice({'True' : 0.005, 'False' : 0.995})
		elif(CarValue == 'FiftyThou'):
			Theft ~= choice({'True' : 0.005, 'False' : 0.995})
		else:
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(HomeBase == 'Suburb'):
		if(CarValue == 'FiveThou'):
			Theft ~= choice({'True' : 1e-05, 'False' : 0.99999})
		elif(CarValue == 'TenThou'):
			Theft ~= choice({'True' : 0.0001, 'False' : 0.9999})
		elif(CarValue == 'TwentyThou'):
			Theft ~= choice({'True' : 0.0003, 'False' : 0.9997})
		elif(CarValue == 'FiftyThou'):
			Theft ~= choice({'True' : 0.0003, 'False' : 0.9997})
		else:
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	else:
		if(CarValue == 'FiveThou'):
			Theft ~= choice({'True' : 1e-05, 'False' : 0.99999})
		elif(CarValue == 'TenThou'):
			Theft ~= choice({'True' : 2e-05, 'False' : 0.99998})
		elif(CarValue == 'TwentyThou'):
			Theft ~= choice({'True' : 5e-05, 'False' : 0.99995})
		elif(CarValue == 'FiftyThou'):
			Theft ~= choice({'True' : 5e-05, 'False' : 0.99995})
		else:
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
else:
	if(HomeBase == 'Secure'):
		if(CarValue == 'FiveThou'):
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
		elif(CarValue == 'TenThou'):
			Theft ~= choice({'True' : 2e-06, 'False' : 0.999998})
		elif(CarValue == 'TwentyThou'):
			Theft ~= choice({'True' : 3e-06, 'False' : 0.999997})
		elif(CarValue == 'FiftyThou'):
			Theft ~= choice({'True' : 2e-06, 'False' : 0.999998})
		else:
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(HomeBase == 'City'):
		if(CarValue == 'FiveThou'):
			Theft ~= choice({'True' : 0.001, 'False' : 0.999})
		elif(CarValue == 'TenThou'):
			Theft ~= choice({'True' : 0.005, 'False' : 0.995})
		elif(CarValue == 'TwentyThou'):
			Theft ~= choice({'True' : 0.01, 'False' : 0.99})
		elif(CarValue == 'FiftyThou'):
			Theft ~= choice({'True' : 0.01, 'False' : 0.99})
		else:
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	elif(HomeBase == 'Suburb'):
		if(CarValue == 'FiveThou'):
			Theft ~= choice({'True' : 1e-05, 'False' : 0.99999})
		elif(CarValue == 'TenThou'):
			Theft ~= choice({'True' : 0.0002, 'False' : 0.9998})
		elif(CarValue == 'TwentyThou'):
			Theft ~= choice({'True' : 0.0005, 'False' : 0.9995})
		elif(CarValue == 'FiftyThou'):
			Theft ~= choice({'True' : 0.0005, 'False' : 0.9995})
		else:
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
	else:
		if(CarValue == 'FiveThou'):
			Theft ~= choice({'True' : 1e-05, 'False' : 0.99999})
		elif(CarValue == 'TenThou'):
			Theft ~= choice({'True' : 0.0001, 'False' : 0.9999})
		elif(CarValue == 'TwentyThou'):
			Theft ~= choice({'True' : 0.0002, 'False' : 0.9998})
		elif(CarValue == 'FiftyThou'):
			Theft ~= choice({'True' : 0.0002, 'False' : 0.9998})
		else:
			Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})


if (Antilock == 'True'):
	if (Mileage == 'FiveThou'):
		if (DrivQuality == 'Poor'):
			Accident ~= choice({'None' : 0.7, 'Mild' : 0.2, 'Moderate' : 0.07, 'Severe' : 0.030000000000000027})
		elif(DrivQuality == 'Normal'):
			Accident ~= choice({'None' : 0.99, 'Mild' : 0.007, 'Moderate' : 0.002, 'Severe' : 0.0010000000000000009})
		else:
			Accident ~= choice({'None' : 0.999, 'Mild' : 0.0007, 'Moderate' : 0.0002, 'Severe' : 9.999999999998899e-05})
	elif(Mileage == 'TwentyThou'):
		if(DrivQuality == 'Poor'):
			Accident ~= choice({'None' : 0.4, 'Mild' : 0.3, 'Moderate' : 0.2, 'Severe' : 0.10000000000000009})
		elif(DrivQuality == 'Normal'):
			Accident ~= choice({'None' : 0.98, 'Mild' : 0.01, 'Moderate' : 0.005, 'Severe' : 0.0050000000000000044})
		else:
			Accident ~= choice({'None' : 0.995, 'Mild' : 0.003, 'Moderate' : 0.001, 'Severe' : 0.0010000000000000009})
	elif(Mileage == 'FiftyThou'):
		if(DrivQuality == 'Poor'):
			Accident ~= choice({'None' : 0.3, 'Mild' : 0.3, 'Moderate' : 0.2, 'Severe' : 0.19999999999999996})
		elif(DrivQuality == 'Normal'):
			Accident ~= choice({'None' : 0.97, 'Mild' : 0.02, 'Moderate' : 0.007, 'Severe' : 0.0030000000000000027})
		else:
			Accident ~= choice({'None' : 0.99, 'Mild' : 0.007, 'Moderate' : 0.002, 'Severe' : 0.0010000000000000009})
	else:
		if(DrivQuality == 'Poor'):
			Accident ~= choice({'None' : 0.2, 'Mild' : 0.2, 'Moderate' : 0.3, 'Severe' : 0.30000000000000004})
		elif(DrivQuality == 'Normal'):
			Accident ~= choice({'None' : 0.95, 'Mild' : 0.03, 'Moderate' : 0.01, 'Severe' : 0.010000000000000009})
		else:
			Accident ~= choice({'None' : 0.985, 'Mild' : 0.01, 'Moderate' : 0.003, 'Severe' : 0.0020000000000000018})
else:
	if(Mileage == 'FiveThou'):
		if(DrivQuality == 'Poor'):
			Accident ~= choice({'None' : 0.6, 'Mild' : 0.2, 'Moderate' : 0.1, 'Severe' : 0.09999999999999998})
		elif(DrivQuality == 'Normal'):
			Accident ~= choice({'None' : 0.98, 'Mild' : 0.01, 'Moderate' : 0.005, 'Severe' : 0.0050000000000000044})
		else:
			Accident ~= choice({'None' : 0.995, 'Mild' : 0.003, 'Moderate' : 0.001, 'Severe' : 0.0010000000000000009})
	elif(Mileage == 'TwentyThou'):
		if(DrivQuality == 'Poor'):
			Accident ~= choice({'None' : 0.3, 'Mild' : 0.2, 'Moderate' : 0.2, 'Severe' : 0.30000000000000004})
		elif(DrivQuality == 'Normal'):
			Accident ~= choice({'None' : 0.96, 'Mild' : 0.02, 'Moderate' : 0.015, 'Severe' : 0.0050000000000000044})
		else:
			Accident ~= choice({'None' : 0.99, 'Mild' : 0.007, 'Moderate' : 0.002, 'Severe' : 0.0010000000000000009})
	elif(Mileage == 'FiftyThou'):
		if(DrivQuality == 'Poor'):
			Accident ~= choice({'None' : 0.2, 'Mild' : 0.2, 'Moderate' : 0.2, 'Severe' : 0.3999999999999999})
		elif(DrivQuality == 'Normal'):
			Accident ~= choice({'None' : 0.95, 'Mild' : 0.03, 'Moderate' : 0.015, 'Severe' : 0.0050000000000000044})
		else:
			Accident ~= choice({'None' : 0.98, 'Mild' : 0.01, 'Moderate' : 0.005, 'Severe' : 0.0050000000000000044})
	else:
		if(DrivQuality == 'Poor'):
			Accident ~= choice({'None' : 0.1, 'Mild' : 0.1, 'Moderate' : 0.3, 'Severe' : 0.5})
		elif(DrivQuality == 'Normal'):
			Accident ~= choice({'None' : 0.94, 'Mild' : 0.03, 'Moderate' : 0.02, 'Severe' : 0.010000000000000009})
		else:
			Accident ~= choice({'None' : 0.98, 'Mild' : 0.01, 'Moderate' : 0.007, 'Severe' : 0.0030000000000000027})


if (Accident == 'None'):
	ILiCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif(Accident == 'Mild'):
	ILiCost ~= choice({'Thousand' : 0.999, 'TenThou' : 0.000998, 'HundredThou' : 1e-06, 'Million' : 9.999999999177334e-07})
elif(Accident == 'Moderate'):
	ILiCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.05, 'HundredThou' : 0.03, 'Million' : 0.019999999999999907})
else:
	ILiCost ~= choice({'Thousand' : 0.8, 'TenThou' : 0.1, 'HundredThou' : 0.06, 'Million' : 0.040000000000000036})


if (Accident == 'None'):
	if (Age == 'Adolescent'):
		if (Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(Age == 'Adult'):
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
	else:
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif(Accident == 'Mild'):
	if(Age == 'Adolescent'):
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 0.96, 'TenThou' : 0.03, 'HundredThou' : 0.009, 'Million' : 0.0010000000000000009})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 0.98, 'TenThou' : 0.019, 'HundredThou' : 0.0009, 'Million' : 9.999999999998899e-05})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.0099, 'HundredThou' : 9e-05, 'Million' : 9.99999999995449e-06})
		else:
			MedCost ~= choice({'Thousand' : 0.999, 'TenThou' : 0.00099, 'HundredThou' : 9e-06, 'Million' : 9.999999999177334e-07})
	elif(Age == 'Adult'):
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 0.96, 'TenThou' : 0.03, 'HundredThou' : 0.009, 'Million' : 0.0010000000000000009})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 0.98, 'TenThou' : 0.019, 'HundredThou' : 0.0009, 'Million' : 9.999999999998899e-05})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.0099, 'HundredThou' : 9e-05, 'Million' : 9.99999999995449e-06})
		else:
			MedCost ~= choice({'Thousand' : 0.999, 'TenThou' : 0.00099, 'HundredThou' : 9e-06, 'Million' : 9.999999999177334e-07})
	else:
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.07, 'HundredThou' : 0.02, 'Million' : 0.010000000000000009})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.04, 'HundredThou' : 0.007, 'Million' : 0.0030000000000000027})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 0.97, 'TenThou' : 0.025, 'HundredThou' : 0.003, 'Million' : 0.0020000000000000018})
		else:
			MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.007, 'HundredThou' : 0.002, 'Million' : 0.0010000000000000009})
elif(Accident == 'Moderate'):
	if(Age == 'Adolescent'):
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.10000000000000009})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 0.8, 'TenThou' : 0.15, 'HundredThou' : 0.03, 'Million' : 0.019999999999999907})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.02, 'HundredThou' : 0.02, 'Million' : 0.010000000000000009})
		else:
			MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.007, 'HundredThou' : 0.002, 'Million' : 0.0010000000000000009})
	elif(Age == 'Adult'):
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.10000000000000009})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 0.8, 'TenThou' : 0.15, 'HundredThou' : 0.03, 'Million' : 0.019999999999999907})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.02, 'HundredThou' : 0.02, 'Million' : 0.010000000000000009})
		else:
			MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.007, 'HundredThou' : 0.002, 'Million' : 0.0010000000000000009})
	else:
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 0.3, 'TenThou' : 0.3, 'HundredThou' : 0.2, 'Million' : 0.19999999999999996})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.10000000000000009})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.07, 'HundredThou' : 0.02, 'Million' : 0.010000000000000009})
		else:
			MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.03, 'HundredThou' : 0.01, 'Million' : 0.010000000000000009})
else:
	if(Age == 'Adolescent'):
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 0.3, 'TenThou' : 0.3, 'HundredThou' : 0.2, 'Million' : 0.19999999999999996})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.10000000000000009})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.07, 'HundredThou' : 0.02, 'Million' : 0.010000000000000009})
		else:
			MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.03, 'HundredThou' : 0.01, 'Million' : 0.010000000000000009})
	elif(Age == 'Adult'):
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 0.3, 'TenThou' : 0.3, 'HundredThou' : 0.2, 'Million' : 0.19999999999999996})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.10000000000000009})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.07, 'HundredThou' : 0.02, 'Million' : 0.010000000000000009})
		else:
			MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.03, 'HundredThou' : 0.01, 'Million' : 0.010000000000000009})
	else:
		if(Cushioning == 'Poor'):
			MedCost ~= choice({'Thousand' : 0.2, 'TenThou' : 0.2, 'HundredThou' : 0.3, 'Million' : 0.30000000000000004})
		elif(Cushioning == 'Fair'):
			MedCost ~= choice({'Thousand' : 0.3, 'TenThou' : 0.3, 'HundredThou' : 0.2, 'Million' : 0.19999999999999996})
		elif(Cushioning == 'Good'):
			MedCost ~= choice({'Thousand' : 0.6, 'TenThou' : 0.3, 'HundredThou' : 0.07, 'Million' : 0.030000000000000027})
		else:
			MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.05, 'HundredThou' : 0.03, 'Million' : 0.019999999999999907})


if (Accident == 'None'):
	if (RuggedAuto == 'EggShell'):
		OtherCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(RuggedAuto == 'Football'):
		OtherCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
	else:
		OtherCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif(Accident == 'Mild'):
	if(RuggedAuto == 'EggShell'):
		OtherCarCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.005, 'HundredThou' : 0.00499, 'Million' : 9.99999999995449e-06})
	elif(RuggedAuto == 'Football'):
		OtherCarCost ~= choice({'Thousand' : 0.9799657, 'TenThou' : 0.00999965, 'HundredThou' : 0.009984651, 'Million' : 4.999899999991175e-05})
	else:
		OtherCarCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.03, 'HundredThou' : 0.01998, 'Million' : 2.0000000000020002e-05})
elif(Accident == 'Moderate'):
	if(RuggedAuto == 'EggShell'):
		OtherCarCost ~= choice({'Thousand' : 0.6, 'TenThou' : 0.2, 'HundredThou' : 0.19998, 'Million' : 1.999999999990898e-05})
	elif(RuggedAuto == 'Football'):
		OtherCarCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.29997, 'Million' : 2.999999999997449e-05})
	else:
		OtherCarCost ~= choice({'Thousand' : 0.4, 'TenThou' : 0.3, 'HundredThou' : 0.29996, 'Million' : 4.0000000000040004e-05})
else:
	if(RuggedAuto == 'EggShell'):
		OtherCarCost ~= choice({'Thousand' : 0.2, 'TenThou' : 0.4, 'HundredThou' : 0.39996, 'Million' : 3.999999999992898e-05})
	elif(RuggedAuto == 'Football'):
		OtherCarCost ~= choice({'Thousand' : 0.1, 'TenThou' : 0.5, 'HundredThou' : 0.39994, 'Million' : 5.999999999994898e-05})
	else:
		OtherCarCost ~= choice({'Thousand' : 0.005, 'TenThou' : 0.55, 'HundredThou' : 0.4449, 'Million' : 9.999999999998899e-05})


if (Accident == 'None'):
	if (RuggedAuto == 'EggShell'):
		ThisCarDam ~= choice({'None' : 1.0, 'Mild' : 0.0, 'Moderate' : 0.0, 'Severe' : 0.0})
	elif(RuggedAuto == 'Football'):
		ThisCarDam ~= choice({'None' : 1.0, 'Mild' : 0.0, 'Moderate' : 0.0, 'Severe' : 0.0})
	else:
		ThisCarDam ~= choice({'None' : 1.0, 'Mild' : 0.0, 'Moderate' : 0.0, 'Severe' : 0.0})
elif(Accident == 'Mild'):
	if(RuggedAuto == 'EggShell'):
		ThisCarDam ~= choice({'None' : 0.001, 'Mild' : 0.9, 'Moderate' : 0.098, 'Severe' : 0.0010000000000000009})
	elif(RuggedAuto == 'Football'):
		ThisCarDam ~= choice({'None' : 0.2, 'Mild' : 0.75, 'Moderate' : 0.049999, 'Severe' : 1.0000000000287557e-06})
	else:
		ThisCarDam ~= choice({'None' : 0.7, 'Mild' : 0.29, 'Moderate' : 0.009999, 'Severe' : 1.0000000000287557e-06})
elif(Accident == 'Moderate'):
	if(RuggedAuto == 'EggShell'):
		ThisCarDam ~= choice({'None' : 1e-06, 'Mild' : 0.000999, 'Moderate' : 0.7, 'Severe' : 0.29900000000000004})
	elif(RuggedAuto == 'Football'):
		ThisCarDam ~= choice({'None' : 0.001, 'Mild' : 0.099, 'Moderate' : 0.8, 'Severe' : 0.09999999999999998})
	else:
		ThisCarDam ~= choice({'None' : 0.05, 'Mild' : 0.6, 'Moderate' : 0.3, 'Severe' : 0.050000000000000044})
else:
	if(RuggedAuto == 'EggShell'):
		ThisCarDam ~= choice({'None' : 1e-06, 'Mild' : 9e-06, 'Moderate' : 9e-05, 'Severe' : 0.9999})
	elif(RuggedAuto == 'Football'):
		ThisCarDam ~= choice({'None' : 1e-06, 'Mild' : 0.000999, 'Moderate' : 0.009, 'Severe' : 0.99})
	else:
		ThisCarDam ~= choice({'None' : 0.05, 'Mild' : 0.2, 'Moderate' : 0.2, 'Severe' : 0.55})


if (ThisCarDam == 'None'):
	if (CarValue == 'FiveThou'):
		if (Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.2, 'TenThou' : 0.8, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'TenThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.05, 'TenThou' : 0.95, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'TwentyThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.04, 'TenThou' : 0.01, 'HundredThou' : 0.95, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'FiftyThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.04, 'TenThou' : 0.01, 'HundredThou' : 0.95, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
	else:
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.04, 'TenThou' : 0.01, 'HundredThou' : 0.2, 'Million' : 0.75})
		else:
			ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif(ThisCarDam == 'Mild'):
	if(CarValue == 'FiveThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.15, 'TenThou' : 0.85, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.05, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'TenThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.03, 'TenThou' : 0.97, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.05, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'TwentyThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.03, 'TenThou' : 0.02, 'HundredThou' : 0.95, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.01, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'FiftyThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.03, 'TenThou' : 0.02, 'HundredThou' : 0.95, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.01, 'HundredThou' : 0.0, 'Million' : 0.0})
	else:
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.02, 'TenThou' : 0.03, 'HundredThou' : 0.25, 'Million' : 0.7})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.98, 'TenThou' : 0.01, 'HundredThou' : 0.01, 'Million' : 0.0})
elif(ThisCarDam == 'Moderate'):
	if(CarValue == 'FiveThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.05, 'TenThou' : 0.95, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.25, 'TenThou' : 0.75, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'TenThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.01, 'TenThou' : 0.99, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.15, 'TenThou' : 0.85, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'TwentyThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.001, 'TenThou' : 0.001, 'HundredThou' : 0.998, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.01, 'TenThou' : 0.01, 'HundredThou' : 0.98, 'Million' : 0.0})
	elif(CarValue == 'FiftyThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.001, 'TenThou' : 0.001, 'HundredThou' : 0.998, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.005, 'TenThou' : 0.005, 'HundredThou' : 0.99, 'Million' : 0.0})
	else:
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.001, 'TenThou' : 0.001, 'HundredThou' : 0.018, 'Million' : 0.98})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.003, 'TenThou' : 0.003, 'HundredThou' : 0.044, 'Million' : 0.95})
else:
	if(CarValue == 'FiveThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 0.03, 'TenThou' : 0.97, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.05, 'TenThou' : 0.95, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'TenThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 0.999999, 'HundredThou' : 0.0, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.01, 'TenThou' : 0.99, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(CarValue == 'TwentyThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 1e-06, 'HundredThou' : 0.999998, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.005, 'TenThou' : 0.005, 'HundredThou' : 0.99, 'Million' : 0.0})
	elif(CarValue == 'FiftyThou'):
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 1e-06, 'HundredThou' : 0.999998, 'Million' : 0.0})
		else:
			ThisCarCost ~= choice({'Thousand' : 0.001, 'TenThou' : 0.001, 'HundredThou' : 0.998, 'Million' : 0.0})
	else:
		if(Theft == 'True'):
			ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 1e-06, 'HundredThou' : 0.009998, 'Million' : 0.99})
		else:
			ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 1e-06, 'HundredThou' : 0.029998, 'Million' : 0.97})


if (OtherCarCost == 'Thousand'):
	if (ThisCarCost == 'Thousand'):
		PropCost ~= choice({'Thousand' : 0.7, 'TenThou' : 0.3, 'HundredThou' : 0.0, 'Million' : 0.0})
	elif(ThisCarCost == 'TenThou'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.95, 'HundredThou' : 0.05, 'Million' : 0.0})
	elif(ThisCarCost == 'HundredThou'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.98, 'Million' : 0.020000000000000018})
	else:
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
elif(OtherCarCost == 'TenThou'):
	if(ThisCarCost == 'Thousand'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.95, 'HundredThou' : 0.05, 'Million' : 0.0})
	elif(ThisCarCost == 'TenThou'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.6, 'HundredThou' : 0.4, 'Million' : 0.0})
	elif(ThisCarCost == 'HundredThou'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.95, 'Million' : 0.050000000000000044})
	else:
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
elif(OtherCarCost == 'HundredThou'):
	if(ThisCarCost == 'Thousand'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.98, 'Million' : 0.020000000000000018})
	elif(ThisCarCost == 'TenThou'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.8, 'Million' : 0.19999999999999996})
	elif(ThisCarCost == 'HundredThou'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.6, 'Million' : 0.4})
	else:
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
else:
	if(ThisCarCost == 'Thousand'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
	elif(ThisCarCost == 'TenThou'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
	elif(ThisCarCost == 'HundredThou'):
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
	else:
		PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})


'''
compiler = SPPL_Compiler(data)
namespace = compiler.execute_module()
model=namespace.model
Accident = Id('Accident')
Age = Id('Age')
Airbag = Id('Airbag')
AntiTheft = Id('AntiTheft')
Antilock = Id('Antilock')
CarValue = Id('CarValue')
Cushioning = Id('Cushioning')
DrivHist = Id('DrivHist')
DrivQuality = Id('DrivQuality')
DrivingSkill = Id('DrivingSkill')
GoodStudent = Id('GoodStudent')
HomeBase = Id('HomeBase')
ILiCost = Id('ILiCost')
MakeModel = Id('MakeModel')
MedCost = Id('MedCost')
Mileage = Id('Mileage')
OtherCar = Id('OtherCar')
OtherCarCost = Id('OtherCarCost')
PropCost = Id('PropCost')
RiskAversion = Id('RiskAversion')
RuggedAuto = Id('RuggedAuto')
SeniorTrain = Id('SeniorTrain')
SocioEcon = Id('SocioEcon')
Theft = Id('Theft')
ThisCarCost = Id('ThisCarCost')
ThisCarDam = Id('ThisCarDam')
VehicleYear = Id('VehicleYear')
events = [AntiTheft << {'False'},MakeModel << {'Economy'},Theft << {'False'},MakeModel << {'SportsCar'},AntiTheft << {'False'},SocioEcon << {'Middle'},Age << {'Adult'},MakeModel << {'FamilySedan'},ThisCarCost << {'Million'},MedCost << {'Million'},Age << {'Adult'},ThisCarCost << {'TenThou'},Theft << {'True'},Antilock << {'True'},ThisCarCost << {'Million'},DrivHist << {'Many'},SeniorTrain << {'True'},OtherCar << {'False'},Theft << {'True'},OtherCar << {'False'},Airbag << {'True'},VehicleYear << {'Current'},RiskAversion << {'Adventurous'},Theft << {'False'},HomeBase << {'City'},CarValue << {'TwentyThou'},MedCost << {'HundredThou'},HomeBase << {'Rural'},DrivHist << {'Many'},GoodStudent << {'True'},ThisCarDam << {'None'},OtherCarCost << {'TenThou'},Theft << {'False'},Age << {'Senior'},CarValue << {'TwentyThou'},HomeBase << {'Secure'},GoodStudent << {'False'},OtherCarCost << {'Million'},HomeBase << {'Rural'},GoodStudent << {'True'},OtherCar << {'False'},OtherCarCost << {'HundredThou'},SocioEcon << {'Prole'},ILiCost << {'Thousand'},DrivingSkill << {'SubStandard'},SocioEcon << {'Middle'},Cushioning << {'Poor'},OtherCarCost << {'Million'},Theft << {'True'},DrivingSkill << {'Normal'},(Accident << {'Mild'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'TenThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Million'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Mild'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'TenThou'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'Million'}) & (Cushioning << {'Fair'}) & (DrivHist << {'One'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'Luxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'Million'}) & (Cushioning << {'Poor'}) & (DrivHist << {'One'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'City'}) & (ILiCost << {'Million'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Middle'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'One'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'TenThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Million'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Mild'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Million'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'HundredThou'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'Luxury'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Million'}) & (MakeModel << {'Economy'}) & (MedCost << {'Thousand'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'One'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'One'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Million'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'TenThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'City'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Million'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Middle'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'TenThou'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'Million'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Secure'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'HundredThou'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Million'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'City'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Luxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Million'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'Mild'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'City'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Million'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Million'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Million'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'Secure'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'One'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Million'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'Economy'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'Million'}) & (Cushioning << {'Good'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'City'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'Million'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'Million'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'Economy'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Poor'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'Million'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Million'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Million'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'Economy'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Million'}) & (MakeModel << {'Luxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'Million'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Secure'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'})]
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
