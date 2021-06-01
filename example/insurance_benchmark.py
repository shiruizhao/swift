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


if ((Age == 'Adolescent')):
	SocioEcon ~= choice({'Prole' : 0.4, 'Middle' : 0.4, 'UpperMiddle' : 0.19, 'Wealthy' : 0.01})
elif ((Age == 'Adult')):
	SocioEcon ~= choice({'Prole' : 0.4, 'Middle' : 0.4, 'UpperMiddle' : 0.19, 'Wealthy' : 0.01})
else:
	SocioEcon ~= choice({'Prole' : 0.5, 'Middle' : 0.2, 'UpperMiddle' : 0.29, 'Wealthy' : 0.01})


if ((SocioEcon == 'Prole') and (Age == 'Adolescent')):
	GoodStudent ~= choice({'True' : 0.1, 'False' : 0.9})
elif ((SocioEcon == 'Prole') and (Age == 'Adult')):
	GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((SocioEcon == 'Prole') and (Age == 'Senior')):
	GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((SocioEcon == 'Middle') and (Age == 'Adolescent')):
	GoodStudent ~= choice({'True' : 0.2, 'False' : 0.8})
elif ((SocioEcon == 'Middle') and (Age == 'Adult')):
	GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((SocioEcon == 'Middle') and (Age == 'Senior')):
	GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((SocioEcon == 'UpperMiddle') and (Age == 'Adolescent')):
	GoodStudent ~= choice({'True' : 0.5, 'False' : 0.5})
elif ((SocioEcon == 'UpperMiddle') and (Age == 'Adult')):
	GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((SocioEcon == 'UpperMiddle') and (Age == 'Senior')):
	GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((SocioEcon == 'Wealthy') and (Age == 'Adolescent')):
	GoodStudent ~= choice({'True' : 0.4, 'False' : 0.6})
elif ((SocioEcon == 'Wealthy') and (Age == 'Adult')):
	GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})
else:
	GoodStudent ~= choice({'True' : 0.0, 'False' : 1.0})


if ((SocioEcon == 'Prole')):
	OtherCar ~= choice({'True' : 0.5, 'False' : 0.5})
elif ((SocioEcon == 'Middle')):
	OtherCar ~= choice({'True' : 0.8, 'False' : 0.2})
elif ((SocioEcon == 'UpperMiddle')):
	OtherCar ~= choice({'True' : 0.9, 'False' : 0.1})
else:
	OtherCar ~= choice({'True' : 0.95, 'False' : 0.05})


if ((Age == 'Adolescent') and (SocioEcon == 'Prole')):
	RiskAversion ~= choice({'Psychopath' : 0.02, 'Adventurous' : 0.58, 'Normal' : 0.3, 'Cautious' : 0.1})
elif ((Age == 'Adolescent') and (SocioEcon == 'Middle')):
	RiskAversion ~= choice({'Psychopath' : 0.02, 'Adventurous' : 0.38, 'Normal' : 0.5, 'Cautious' : 0.1})
elif ((Age == 'Adolescent') and (SocioEcon == 'UpperMiddle')):
	RiskAversion ~= choice({'Psychopath' : 0.02, 'Adventurous' : 0.48, 'Normal' : 0.4, 'Cautious' : 0.1})
elif ((Age == 'Adolescent') and (SocioEcon == 'Wealthy')):
	RiskAversion ~= choice({'Psychopath' : 0.02, 'Adventurous' : 0.58, 'Normal' : 0.3, 'Cautious' : 0.1})
elif ((Age == 'Adult') and (SocioEcon == 'Prole')):
	RiskAversion ~= choice({'Psychopath' : 0.015, 'Adventurous' : 0.285, 'Normal' : 0.5, 'Cautious' : 0.2})
elif ((Age == 'Adult') and (SocioEcon == 'Middle')):
	RiskAversion ~= choice({'Psychopath' : 0.015, 'Adventurous' : 0.185, 'Normal' : 0.6, 'Cautious' : 0.2})
elif ((Age == 'Adult') and (SocioEcon == 'UpperMiddle')):
	RiskAversion ~= choice({'Psychopath' : 0.015, 'Adventurous' : 0.285, 'Normal' : 0.5, 'Cautious' : 0.2})
elif ((Age == 'Adult') and (SocioEcon == 'Wealthy')):
	RiskAversion ~= choice({'Psychopath' : 0.015, 'Adventurous' : 0.285, 'Normal' : 0.4, 'Cautious' : 0.3})
elif ((Age == 'Senior') and (SocioEcon == 'Prole')):
	RiskAversion ~= choice({'Psychopath' : 0.01, 'Adventurous' : 0.09, 'Normal' : 0.4, 'Cautious' : 0.5})
elif ((Age == 'Senior') and (SocioEcon == 'Middle')):
	RiskAversion ~= choice({'Psychopath' : 0.01, 'Adventurous' : 0.04, 'Normal' : 0.35, 'Cautious' : 0.6})
elif ((Age == 'Senior') and (SocioEcon == 'UpperMiddle')):
	RiskAversion ~= choice({'Psychopath' : 0.01, 'Adventurous' : 0.09, 'Normal' : 0.4, 'Cautious' : 0.5})
else:
	RiskAversion ~= choice({'Psychopath' : 0.01, 'Adventurous' : 0.09, 'Normal' : 0.4, 'Cautious' : 0.5})


if ((Age == 'Adolescent') and (RiskAversion == 'Psychopath')):
	SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((Age == 'Adolescent') and (RiskAversion == 'Adventurous')):
	SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((Age == 'Adolescent') and (RiskAversion == 'Normal')):
	SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((Age == 'Adolescent') and (RiskAversion == 'Cautious')):
	SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((Age == 'Adult') and (RiskAversion == 'Psychopath')):
	SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((Age == 'Adult') and (RiskAversion == 'Adventurous')):
	SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((Age == 'Adult') and (RiskAversion == 'Normal')):
	SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((Age == 'Adult') and (RiskAversion == 'Cautious')):
	SeniorTrain ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((Age == 'Senior') and (RiskAversion == 'Psychopath')):
	SeniorTrain ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((Age == 'Senior') and (RiskAversion == 'Adventurous')):
	SeniorTrain ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((Age == 'Senior') and (RiskAversion == 'Normal')):
	SeniorTrain ~= choice({'True' : 0.3, 'False' : 0.7})
else:
	SeniorTrain ~= choice({'True' : 0.9, 'False' : 0.1})


if ((SocioEcon == 'Prole') and (RiskAversion == 'Psychopath')):
	VehicleYear ~= choice({'Current' : 0.15, 'Older' : 0.85})
elif ((SocioEcon == 'Prole') and (RiskAversion == 'Adventurous')):
	VehicleYear ~= choice({'Current' : 0.15, 'Older' : 0.85})
elif ((SocioEcon == 'Prole') and (RiskAversion == 'Normal')):
	VehicleYear ~= choice({'Current' : 0.15, 'Older' : 0.85})
elif ((SocioEcon == 'Prole') and (RiskAversion == 'Cautious')):
	VehicleYear ~= choice({'Current' : 0.15, 'Older' : 0.85})
elif ((SocioEcon == 'Middle') and (RiskAversion == 'Psychopath')):
	VehicleYear ~= choice({'Current' : 0.3, 'Older' : 0.7})
elif ((SocioEcon == 'Middle') and (RiskAversion == 'Adventurous')):
	VehicleYear ~= choice({'Current' : 0.3, 'Older' : 0.7})
elif ((SocioEcon == 'Middle') and (RiskAversion == 'Normal')):
	VehicleYear ~= choice({'Current' : 0.3, 'Older' : 0.7})
elif ((SocioEcon == 'Middle') and (RiskAversion == 'Cautious')):
	VehicleYear ~= choice({'Current' : 0.3, 'Older' : 0.7})
elif ((SocioEcon == 'UpperMiddle') and (RiskAversion == 'Psychopath')):
	VehicleYear ~= choice({'Current' : 0.8, 'Older' : 0.2})
elif ((SocioEcon == 'UpperMiddle') and (RiskAversion == 'Adventurous')):
	VehicleYear ~= choice({'Current' : 0.8, 'Older' : 0.2})
elif ((SocioEcon == 'UpperMiddle') and (RiskAversion == 'Normal')):
	VehicleYear ~= choice({'Current' : 0.8, 'Older' : 0.2})
elif ((SocioEcon == 'UpperMiddle') and (RiskAversion == 'Cautious')):
	VehicleYear ~= choice({'Current' : 0.8, 'Older' : 0.2})
elif ((SocioEcon == 'Wealthy') and (RiskAversion == 'Psychopath')):
	VehicleYear ~= choice({'Current' : 0.9, 'Older' : 0.1})
elif ((SocioEcon == 'Wealthy') and (RiskAversion == 'Adventurous')):
	VehicleYear ~= choice({'Current' : 0.9, 'Older' : 0.1})
elif ((SocioEcon == 'Wealthy') and (RiskAversion == 'Normal')):
	VehicleYear ~= choice({'Current' : 0.9, 'Older' : 0.1})
else:
	VehicleYear ~= choice({'Current' : 0.9, 'Older' : 0.1})


if ((RiskAversion == 'Psychopath') and (SocioEcon == 'Prole')):
	AntiTheft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((RiskAversion == 'Psychopath') and (SocioEcon == 'Middle')):
	AntiTheft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((RiskAversion == 'Psychopath') and (SocioEcon == 'UpperMiddle')):
	AntiTheft ~= choice({'True' : 0.05, 'False' : 0.95})
elif ((RiskAversion == 'Psychopath') and (SocioEcon == 'Wealthy')):
	AntiTheft ~= choice({'True' : 0.5, 'False' : 0.5})
elif ((RiskAversion == 'Adventurous') and (SocioEcon == 'Prole')):
	AntiTheft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((RiskAversion == 'Adventurous') and (SocioEcon == 'Middle')):
	AntiTheft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((RiskAversion == 'Adventurous') and (SocioEcon == 'UpperMiddle')):
	AntiTheft ~= choice({'True' : 0.2, 'False' : 0.8})
elif ((RiskAversion == 'Adventurous') and (SocioEcon == 'Wealthy')):
	AntiTheft ~= choice({'True' : 0.5, 'False' : 0.5})
elif ((RiskAversion == 'Normal') and (SocioEcon == 'Prole')):
	AntiTheft ~= choice({'True' : 0.1, 'False' : 0.9})
elif ((RiskAversion == 'Normal') and (SocioEcon == 'Middle')):
	AntiTheft ~= choice({'True' : 0.3, 'False' : 0.7})
elif ((RiskAversion == 'Normal') and (SocioEcon == 'UpperMiddle')):
	AntiTheft ~= choice({'True' : 0.9, 'False' : 0.1})
elif ((RiskAversion == 'Normal') and (SocioEcon == 'Wealthy')):
	AntiTheft ~= choice({'True' : 0.8, 'False' : 0.2})
elif ((RiskAversion == 'Cautious') and (SocioEcon == 'Prole')):
	AntiTheft ~= choice({'True' : 0.95, 'False' : 0.05})
elif ((RiskAversion == 'Cautious') and (SocioEcon == 'Middle')):
	AntiTheft ~= choice({'True' : 0.999999, 'False' : 1e-06})
elif ((RiskAversion == 'Cautious') and (SocioEcon == 'UpperMiddle')):
	AntiTheft ~= choice({'True' : 0.999999, 'False' : 1e-06})
else:
	AntiTheft ~= choice({'True' : 0.999999, 'False' : 1e-06})


if ((Age == 'Adolescent') and (SeniorTrain == 'True')):
	DrivingSkill ~= choice({'SubStandard' : 0.5, 'Normal' : 0.45, 'Expert' : 0.05})
elif ((Age == 'Adolescent') and (SeniorTrain == 'False')):
	DrivingSkill ~= choice({'SubStandard' : 0.5, 'Normal' : 0.45, 'Expert' : 0.05})
elif ((Age == 'Adult') and (SeniorTrain == 'True')):
	DrivingSkill ~= choice({'SubStandard' : 0.3, 'Normal' : 0.6, 'Expert' : 0.1})
elif ((Age == 'Adult') and (SeniorTrain == 'False')):
	DrivingSkill ~= choice({'SubStandard' : 0.3, 'Normal' : 0.6, 'Expert' : 0.1})
elif ((Age == 'Senior') and (SeniorTrain == 'True')):
	DrivingSkill ~= choice({'SubStandard' : 0.1, 'Normal' : 0.6, 'Expert' : 0.3})
else:
	DrivingSkill ~= choice({'SubStandard' : 0.4, 'Normal' : 0.5, 'Expert' : 0.1})


if ((RiskAversion == 'Psychopath') and (SocioEcon == 'Prole')):
	HomeBase ~= choice({'Secure' : 1e-06, 'City' : 0.8, 'Suburb' : 0.049999, 'Rural' : 0.15})
elif ((RiskAversion == 'Psychopath') and (SocioEcon == 'Middle')):
	HomeBase ~= choice({'Secure' : 0.15, 'City' : 0.8, 'Suburb' : 0.04, 'Rural' : 0.01})
elif ((RiskAversion == 'Psychopath') and (SocioEcon == 'UpperMiddle')):
	HomeBase ~= choice({'Secure' : 0.35, 'City' : 0.6, 'Suburb' : 0.04, 'Rural' : 0.01})
elif ((RiskAversion == 'Psychopath') and (SocioEcon == 'Wealthy')):
	HomeBase ~= choice({'Secure' : 0.489999, 'City' : 0.5, 'Suburb' : 1e-06, 'Rural' : 0.01})
elif ((RiskAversion == 'Adventurous') and (SocioEcon == 'Prole')):
	HomeBase ~= choice({'Secure' : 1e-06, 'City' : 0.8, 'Suburb' : 0.05, 'Rural' : 0.149999})
elif ((RiskAversion == 'Adventurous') and (SocioEcon == 'Middle')):
	HomeBase ~= choice({'Secure' : 0.01, 'City' : 0.25, 'Suburb' : 0.6, 'Rural' : 0.14})
elif ((RiskAversion == 'Adventurous') and (SocioEcon == 'UpperMiddle')):
	HomeBase ~= choice({'Secure' : 0.2, 'City' : 0.4, 'Suburb' : 0.3, 'Rural' : 0.1})
elif ((RiskAversion == 'Adventurous') and (SocioEcon == 'Wealthy')):
	HomeBase ~= choice({'Secure' : 0.95, 'City' : 1e-06, 'Suburb' : 1e-06, 'Rural' : 0.049998})
elif ((RiskAversion == 'Normal') and (SocioEcon == 'Prole')):
	HomeBase ~= choice({'Secure' : 1e-06, 'City' : 0.8, 'Suburb' : 0.05, 'Rural' : 0.149999})
elif ((RiskAversion == 'Normal') and (SocioEcon == 'Middle')):
	HomeBase ~= choice({'Secure' : 0.299999, 'City' : 1e-06, 'Suburb' : 0.6, 'Rural' : 0.1})
elif ((RiskAversion == 'Normal') and (SocioEcon == 'UpperMiddle')):
	HomeBase ~= choice({'Secure' : 0.5, 'City' : 1e-06, 'Suburb' : 0.4, 'Rural' : 0.099999})
elif ((RiskAversion == 'Normal') and (SocioEcon == 'Wealthy')):
	HomeBase ~= choice({'Secure' : 0.85, 'City' : 1e-06, 'Suburb' : 0.001, 'Rural' : 0.148999})
elif ((RiskAversion == 'Cautious') and (SocioEcon == 'Prole')):
	HomeBase ~= choice({'Secure' : 1e-06, 'City' : 0.8, 'Suburb' : 0.05, 'Rural' : 0.149999})
elif ((RiskAversion == 'Cautious') and (SocioEcon == 'Middle')):
	HomeBase ~= choice({'Secure' : 0.95, 'City' : 1e-06, 'Suburb' : 0.024445, 'Rural' : 0.025554})
elif ((RiskAversion == 'Cautious') and (SocioEcon == 'UpperMiddle')):
	HomeBase ~= choice({'Secure' : 0.999997, 'City' : 1e-06, 'Suburb' : 1e-06, 'Rural' : 1e-06})
else:
	HomeBase ~= choice({'Secure' : 0.999997, 'City' : 1e-06, 'Suburb' : 1e-06, 'Rural' : 1e-06})


if ((SocioEcon == 'Prole') and (RiskAversion == 'Psychopath')):
	MakeModel ~= choice({'SportsCar' : 0.1, 'Economy' : 0.7, 'FamilySedan' : 0.2, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'Prole') and (RiskAversion == 'Adventurous')):
	MakeModel ~= choice({'SportsCar' : 0.1, 'Economy' : 0.7, 'FamilySedan' : 0.2, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'Prole') and (RiskAversion == 'Normal')):
	MakeModel ~= choice({'SportsCar' : 0.1, 'Economy' : 0.7, 'FamilySedan' : 0.2, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'Prole') and (RiskAversion == 'Cautious')):
	MakeModel ~= choice({'SportsCar' : 0.1, 'Economy' : 0.7, 'FamilySedan' : 0.2, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'Middle') and (RiskAversion == 'Psychopath')):
	MakeModel ~= choice({'SportsCar' : 0.15, 'Economy' : 0.2, 'FamilySedan' : 0.65, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'Middle') and (RiskAversion == 'Adventurous')):
	MakeModel ~= choice({'SportsCar' : 0.15, 'Economy' : 0.2, 'FamilySedan' : 0.65, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'Middle') and (RiskAversion == 'Normal')):
	MakeModel ~= choice({'SportsCar' : 0.15, 'Economy' : 0.2, 'FamilySedan' : 0.65, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'Middle') and (RiskAversion == 'Cautious')):
	MakeModel ~= choice({'SportsCar' : 0.15, 'Economy' : 0.2, 'FamilySedan' : 0.65, 'Luxury' : 0.0, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'UpperMiddle') and (RiskAversion == 'Psychopath')):
	MakeModel ~= choice({'SportsCar' : 0.2, 'Economy' : 0.05, 'FamilySedan' : 0.3, 'Luxury' : 0.45, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'UpperMiddle') and (RiskAversion == 'Adventurous')):
	MakeModel ~= choice({'SportsCar' : 0.2, 'Economy' : 0.05, 'FamilySedan' : 0.3, 'Luxury' : 0.45, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'UpperMiddle') and (RiskAversion == 'Normal')):
	MakeModel ~= choice({'SportsCar' : 0.2, 'Economy' : 0.05, 'FamilySedan' : 0.3, 'Luxury' : 0.45, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'UpperMiddle') and (RiskAversion == 'Cautious')):
	MakeModel ~= choice({'SportsCar' : 0.2, 'Economy' : 0.05, 'FamilySedan' : 0.3, 'Luxury' : 0.45, 'SuperLuxury' : 0.0})
elif ((SocioEcon == 'Wealthy') and (RiskAversion == 'Psychopath')):
	MakeModel ~= choice({'SportsCar' : 0.3, 'Economy' : 0.01, 'FamilySedan' : 0.09, 'Luxury' : 0.4, 'SuperLuxury' : 0.2})
elif ((SocioEcon == 'Wealthy') and (RiskAversion == 'Adventurous')):
	MakeModel ~= choice({'SportsCar' : 0.3, 'Economy' : 0.01, 'FamilySedan' : 0.09, 'Luxury' : 0.4, 'SuperLuxury' : 0.2})
elif ((SocioEcon == 'Wealthy') and (RiskAversion == 'Normal')):
	MakeModel ~= choice({'SportsCar' : 0.3, 'Economy' : 0.01, 'FamilySedan' : 0.09, 'Luxury' : 0.4, 'SuperLuxury' : 0.2})
else:
	MakeModel ~= choice({'SportsCar' : 0.3, 'Economy' : 0.01, 'FamilySedan' : 0.09, 'Luxury' : 0.4, 'SuperLuxury' : 0.2})


if ((MakeModel == 'SportsCar') and (VehicleYear == 'Current')):
	RuggedAuto ~= choice({'EggShell' : 0.95, 'Football' : 0.04, 'Tank' : 0.01})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Older')):
	RuggedAuto ~= choice({'EggShell' : 0.95, 'Football' : 0.04, 'Tank' : 0.01})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Current')):
	RuggedAuto ~= choice({'EggShell' : 0.5, 'Football' : 0.5, 'Tank' : 0.0})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Older')):
	RuggedAuto ~= choice({'EggShell' : 0.9, 'Football' : 0.1, 'Tank' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Current')):
	RuggedAuto ~= choice({'EggShell' : 0.2, 'Football' : 0.6, 'Tank' : 0.2})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Older')):
	RuggedAuto ~= choice({'EggShell' : 0.05, 'Football' : 0.55, 'Tank' : 0.4})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Current')):
	RuggedAuto ~= choice({'EggShell' : 0.1, 'Football' : 0.6, 'Tank' : 0.3})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Older')):
	RuggedAuto ~= choice({'EggShell' : 0.1, 'Football' : 0.6, 'Tank' : 0.3})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Current')):
	RuggedAuto ~= choice({'EggShell' : 0.05, 'Football' : 0.55, 'Tank' : 0.4})
else:
	RuggedAuto ~= choice({'EggShell' : 0.05, 'Football' : 0.55, 'Tank' : 0.4})


if ((MakeModel == 'SportsCar') and (VehicleYear == 'Current')):
	Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Older')):
	Airbag ~= choice({'True' : 0.1, 'False' : 0.9})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Current')):
	Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Older')):
	Airbag ~= choice({'True' : 0.05, 'False' : 0.95})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Current')):
	Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Older')):
	Airbag ~= choice({'True' : 0.2, 'False' : 0.8})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Current')):
	Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Older')):
	Airbag ~= choice({'True' : 0.6, 'False' : 0.4})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Current')):
	Airbag ~= choice({'True' : 1.0, 'False' : 0.0})
else:
	Airbag ~= choice({'True' : 0.1, 'False' : 0.9})


if ((MakeModel == 'SportsCar') and (VehicleYear == 'Current')):
	Antilock ~= choice({'True' : 0.9, 'False' : 0.1})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Older')):
	Antilock ~= choice({'True' : 0.1, 'False' : 0.9})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Current')):
	Antilock ~= choice({'True' : 0.001, 'False' : 0.999})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Older')):
	Antilock ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Current')):
	Antilock ~= choice({'True' : 0.4, 'False' : 0.6})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Older')):
	Antilock ~= choice({'True' : 0.0, 'False' : 1.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Current')):
	Antilock ~= choice({'True' : 0.99, 'False' : 0.01})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Older')):
	Antilock ~= choice({'True' : 0.3, 'False' : 0.7})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Current')):
	Antilock ~= choice({'True' : 0.99, 'False' : 0.01})
else:
	Antilock ~= choice({'True' : 0.15, 'False' : 0.85})


if ((MakeModel == 'SportsCar') and (VehicleYear == 'Current') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.8, 'FiftyThou' : 0.09, 'Million' : 0.01})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Current') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.8, 'FiftyThou' : 0.09, 'Million' : 0.01})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Current') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.8, 'FiftyThou' : 0.09, 'Million' : 0.01})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Current') and (Mileage == 'Domino')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.8, 'FiftyThou' : 0.09, 'Million' : 0.01})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Older') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 0.03, 'TenThou' : 0.3, 'TwentyThou' : 0.6, 'FiftyThou' : 0.06, 'Million' : 0.01})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Older') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 0.16, 'TenThou' : 0.5, 'TwentyThou' : 0.3, 'FiftyThou' : 0.03, 'Million' : 0.01})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Older') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 0.4, 'TenThou' : 0.47, 'TwentyThou' : 0.1, 'FiftyThou' : 0.02, 'Million' : 0.01})
elif ((MakeModel == 'SportsCar') and (VehicleYear == 'Older') and (Mileage == 'Domino')):
	CarValue ~= choice({'FiveThou' : 0.9, 'TenThou' : 0.06, 'TwentyThou' : 0.02, 'FiftyThou' : 0.01, 'Million' : 0.01})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Current') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.8, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Current') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.8, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Current') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.8, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Current') and (Mileage == 'Domino')):
	CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.8, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Older') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 0.25, 'TenThou' : 0.7, 'TwentyThou' : 0.05, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Older') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 0.7, 'TenThou' : 0.2999, 'TwentyThou' : 0.0001, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Older') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 0.99, 'TenThou' : 0.009999, 'TwentyThou' : 1e-06, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'Economy') and (VehicleYear == 'Older') and (Mileage == 'Domino')):
	CarValue ~= choice({'FiveThou' : 0.999998, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Current') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.9, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Current') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.9, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Current') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.9, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Current') and (Mileage == 'Domino')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.1, 'TwentyThou' : 0.9, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Older') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 0.2, 'TenThou' : 0.3, 'TwentyThou' : 0.5, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Older') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 0.5, 'TenThou' : 0.3, 'TwentyThou' : 0.2, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Older') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 0.7, 'TenThou' : 0.2, 'TwentyThou' : 0.1, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'FamilySedan') and (VehicleYear == 'Older') and (Mileage == 'Domino')):
	CarValue ~= choice({'FiveThou' : 0.99, 'TenThou' : 0.009999, 'TwentyThou' : 1e-06, 'FiftyThou' : 0.0, 'Million' : 0.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Current') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 1.0, 'Million' : 0.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Current') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 1.0, 'Million' : 0.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Current') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 1.0, 'Million' : 0.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Current') and (Mileage == 'Domino')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 1.0, 'Million' : 0.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Older') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 0.01, 'TenThou' : 0.09, 'TwentyThou' : 0.2, 'FiftyThou' : 0.7, 'Million' : 0.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Older') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 0.05, 'TenThou' : 0.15, 'TwentyThou' : 0.3, 'FiftyThou' : 0.5, 'Million' : 0.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Older') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 0.1, 'TenThou' : 0.3, 'TwentyThou' : 0.3, 'FiftyThou' : 0.3, 'Million' : 0.0})
elif ((MakeModel == 'Luxury') and (VehicleYear == 'Older') and (Mileage == 'Domino')):
	CarValue ~= choice({'FiveThou' : 0.2, 'TenThou' : 0.2, 'TwentyThou' : 0.3, 'FiftyThou' : 0.3, 'Million' : 0.0})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Current') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 0.0, 'Million' : 1.0})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Current') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 0.0, 'Million' : 1.0})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Current') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 0.0, 'Million' : 1.0})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Current') and (Mileage == 'Domino')):
	CarValue ~= choice({'FiveThou' : 0.0, 'TenThou' : 0.0, 'TwentyThou' : 0.0, 'FiftyThou' : 0.0, 'Million' : 1.0})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Older') and (Mileage == 'FiveThou')):
	CarValue ~= choice({'FiveThou' : 1e-06, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 1e-06, 'Million' : 0.999996})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Older') and (Mileage == 'TwentyThou')):
	CarValue ~= choice({'FiveThou' : 1e-06, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 1e-06, 'Million' : 0.999996})
elif ((MakeModel == 'SuperLuxury') and (VehicleYear == 'Older') and (Mileage == 'FiftyThou')):
	CarValue ~= choice({'FiveThou' : 1e-06, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 1e-06, 'Million' : 0.999996})
else:
	CarValue ~= choice({'FiveThou' : 1e-06, 'TenThou' : 1e-06, 'TwentyThou' : 1e-06, 'FiftyThou' : 1e-06, 'Million' : 0.999996})


if ((RuggedAuto == 'EggShell') and (Airbag == 'True')):
	Cushioning ~= choice({'Poor' : 0.5, 'Fair' : 0.3, 'Good' : 0.2, 'Excellent' : 0.0})
elif ((RuggedAuto == 'EggShell') and (Airbag == 'False')):
	Cushioning ~= choice({'Poor' : 0.7, 'Fair' : 0.3, 'Good' : 0.0, 'Excellent' : 0.0})
elif ((RuggedAuto == 'Football') and (Airbag == 'True')):
	Cushioning ~= choice({'Poor' : 0.0, 'Fair' : 0.1, 'Good' : 0.6, 'Excellent' : 0.3})
elif ((RuggedAuto == 'Football') and (Airbag == 'False')):
	Cushioning ~= choice({'Poor' : 0.1, 'Fair' : 0.6, 'Good' : 0.3, 'Excellent' : 0.0})
elif ((RuggedAuto == 'Tank') and (Airbag == 'True')):
	Cushioning ~= choice({'Poor' : 0.0, 'Fair' : 0.0, 'Good' : 0.0, 'Excellent' : 1.0})
else:
	Cushioning ~= choice({'Poor' : 0.0, 'Fair' : 0.0, 'Good' : 0.7, 'Excellent' : 0.3})


if ((DrivingSkill == 'SubStandard') and (RiskAversion == 'Psychopath')):
	DrivHist ~= choice({'Zero' : 0.001, 'One' : 0.004, 'Many' : 0.995})
elif ((DrivingSkill == 'SubStandard') and (RiskAversion == 'Adventurous')):
	DrivHist ~= choice({'Zero' : 0.002, 'One' : 0.008, 'Many' : 0.99})
elif ((DrivingSkill == 'SubStandard') and (RiskAversion == 'Normal')):
	DrivHist ~= choice({'Zero' : 0.03, 'One' : 0.15, 'Many' : 0.82})
elif ((DrivingSkill == 'SubStandard') and (RiskAversion == 'Cautious')):
	DrivHist ~= choice({'Zero' : 0.3, 'One' : 0.3, 'Many' : 0.4})
elif ((DrivingSkill == 'Normal') and (RiskAversion == 'Psychopath')):
	DrivHist ~= choice({'Zero' : 0.1, 'One' : 0.3, 'Many' : 0.6})
elif ((DrivingSkill == 'Normal') and (RiskAversion == 'Adventurous')):
	DrivHist ~= choice({'Zero' : 0.5, 'One' : 0.3, 'Many' : 0.2})
elif ((DrivingSkill == 'Normal') and (RiskAversion == 'Normal')):
	DrivHist ~= choice({'Zero' : 0.9, 'One' : 0.07, 'Many' : 0.03})
elif ((DrivingSkill == 'Normal') and (RiskAversion == 'Cautious')):
	DrivHist ~= choice({'Zero' : 0.95, 'One' : 0.04, 'Many' : 0.01})
elif ((DrivingSkill == 'Expert') and (RiskAversion == 'Psychopath')):
	DrivHist ~= choice({'Zero' : 0.3, 'One' : 0.3, 'Many' : 0.4})
elif ((DrivingSkill == 'Expert') and (RiskAversion == 'Adventurous')):
	DrivHist ~= choice({'Zero' : 0.6, 'One' : 0.3, 'Many' : 0.1})
elif ((DrivingSkill == 'Expert') and (RiskAversion == 'Normal')):
	DrivHist ~= choice({'Zero' : 0.99, 'One' : 0.009999, 'Many' : 1e-06})
else:
	DrivHist ~= choice({'Zero' : 0.999998, 'One' : 1e-06, 'Many' : 1e-06})


if ((DrivingSkill == 'SubStandard') and (RiskAversion == 'Psychopath')):
	DrivQuality ~= choice({'Poor' : 1.0, 'Normal' : 0.0, 'Excellent' : 0.0})
elif ((DrivingSkill == 'SubStandard') and (RiskAversion == 'Adventurous')):
	DrivQuality ~= choice({'Poor' : 1.0, 'Normal' : 0.0, 'Excellent' : 0.0})
elif ((DrivingSkill == 'SubStandard') and (RiskAversion == 'Normal')):
	DrivQuality ~= choice({'Poor' : 1.0, 'Normal' : 0.0, 'Excellent' : 0.0})
elif ((DrivingSkill == 'SubStandard') and (RiskAversion == 'Cautious')):
	DrivQuality ~= choice({'Poor' : 1.0, 'Normal' : 0.0, 'Excellent' : 0.0})
elif ((DrivingSkill == 'Normal') and (RiskAversion == 'Psychopath')):
	DrivQuality ~= choice({'Poor' : 0.5, 'Normal' : 0.2, 'Excellent' : 0.3})
elif ((DrivingSkill == 'Normal') and (RiskAversion == 'Adventurous')):
	DrivQuality ~= choice({'Poor' : 0.3, 'Normal' : 0.4, 'Excellent' : 0.3})
elif ((DrivingSkill == 'Normal') and (RiskAversion == 'Normal')):
	DrivQuality ~= choice({'Poor' : 0.0, 'Normal' : 1.0, 'Excellent' : 0.0})
elif ((DrivingSkill == 'Normal') and (RiskAversion == 'Cautious')):
	DrivQuality ~= choice({'Poor' : 0.0, 'Normal' : 0.8, 'Excellent' : 0.2})
elif ((DrivingSkill == 'Expert') and (RiskAversion == 'Psychopath')):
	DrivQuality ~= choice({'Poor' : 0.3, 'Normal' : 0.2, 'Excellent' : 0.5})
elif ((DrivingSkill == 'Expert') and (RiskAversion == 'Adventurous')):
	DrivQuality ~= choice({'Poor' : 0.01, 'Normal' : 0.01, 'Excellent' : 0.98})
elif ((DrivingSkill == 'Expert') and (RiskAversion == 'Normal')):
	DrivQuality ~= choice({'Poor' : 0.0, 'Normal' : 0.0, 'Excellent' : 1.0})
else:
	DrivQuality ~= choice({'Poor' : 0.0, 'Normal' : 0.0, 'Excellent' : 1.0})


if ((AntiTheft == 'True') and (HomeBase == 'Secure') and (CarValue == 'FiveThou')):
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((AntiTheft == 'True') and (HomeBase == 'Secure') and (CarValue == 'TenThou')):
	Theft ~= choice({'True' : 2e-06, 'False' : 0.999998})
elif ((AntiTheft == 'True') and (HomeBase == 'Secure') and (CarValue == 'TwentyThou')):
	Theft ~= choice({'True' : 3e-06, 'False' : 0.999997})
elif ((AntiTheft == 'True') and (HomeBase == 'Secure') and (CarValue == 'FiftyThou')):
	Theft ~= choice({'True' : 2e-06, 'False' : 0.999998})
elif ((AntiTheft == 'True') and (HomeBase == 'Secure') and (CarValue == 'Million')):
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((AntiTheft == 'True') and (HomeBase == 'City') and (CarValue == 'FiveThou')):
	Theft ~= choice({'True' : 0.0005, 'False' : 0.9995})
elif ((AntiTheft == 'True') and (HomeBase == 'City') and (CarValue == 'TenThou')):
	Theft ~= choice({'True' : 0.002, 'False' : 0.998})
elif ((AntiTheft == 'True') and (HomeBase == 'City') and (CarValue == 'TwentyThou')):
	Theft ~= choice({'True' : 0.005, 'False' : 0.995})
elif ((AntiTheft == 'True') and (HomeBase == 'City') and (CarValue == 'FiftyThou')):
	Theft ~= choice({'True' : 0.005, 'False' : 0.995})
elif ((AntiTheft == 'True') and (HomeBase == 'City') and (CarValue == 'Million')):
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((AntiTheft == 'True') and (HomeBase == 'Suburb') and (CarValue == 'FiveThou')):
	Theft ~= choice({'True' : 1e-05, 'False' : 0.99999})
elif ((AntiTheft == 'True') and (HomeBase == 'Suburb') and (CarValue == 'TenThou')):
	Theft ~= choice({'True' : 0.0001, 'False' : 0.9999})
elif ((AntiTheft == 'True') and (HomeBase == 'Suburb') and (CarValue == 'TwentyThou')):
	Theft ~= choice({'True' : 0.0003, 'False' : 0.9997})
elif ((AntiTheft == 'True') and (HomeBase == 'Suburb') and (CarValue == 'FiftyThou')):
	Theft ~= choice({'True' : 0.0003, 'False' : 0.9997})
elif ((AntiTheft == 'True') and (HomeBase == 'Suburb') and (CarValue == 'Million')):
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((AntiTheft == 'True') and (HomeBase == 'Rural') and (CarValue == 'FiveThou')):
	Theft ~= choice({'True' : 1e-05, 'False' : 0.99999})
elif ((AntiTheft == 'True') and (HomeBase == 'Rural') and (CarValue == 'TenThou')):
	Theft ~= choice({'True' : 2e-05, 'False' : 0.99998})
elif ((AntiTheft == 'True') and (HomeBase == 'Rural') and (CarValue == 'TwentyThou')):
	Theft ~= choice({'True' : 5e-05, 'False' : 0.99995})
elif ((AntiTheft == 'True') and (HomeBase == 'Rural') and (CarValue == 'FiftyThou')):
	Theft ~= choice({'True' : 5e-05, 'False' : 0.99995})
elif ((AntiTheft == 'True') and (HomeBase == 'Rural') and (CarValue == 'Million')):
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((AntiTheft == 'False') and (HomeBase == 'Secure') and (CarValue == 'FiveThou')):
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((AntiTheft == 'False') and (HomeBase == 'Secure') and (CarValue == 'TenThou')):
	Theft ~= choice({'True' : 2e-06, 'False' : 0.999998})
elif ((AntiTheft == 'False') and (HomeBase == 'Secure') and (CarValue == 'TwentyThou')):
	Theft ~= choice({'True' : 3e-06, 'False' : 0.999997})
elif ((AntiTheft == 'False') and (HomeBase == 'Secure') and (CarValue == 'FiftyThou')):
	Theft ~= choice({'True' : 2e-06, 'False' : 0.999998})
elif ((AntiTheft == 'False') and (HomeBase == 'Secure') and (CarValue == 'Million')):
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((AntiTheft == 'False') and (HomeBase == 'City') and (CarValue == 'FiveThou')):
	Theft ~= choice({'True' : 0.001, 'False' : 0.999})
elif ((AntiTheft == 'False') and (HomeBase == 'City') and (CarValue == 'TenThou')):
	Theft ~= choice({'True' : 0.005, 'False' : 0.995})
elif ((AntiTheft == 'False') and (HomeBase == 'City') and (CarValue == 'TwentyThou')):
	Theft ~= choice({'True' : 0.01, 'False' : 0.99})
elif ((AntiTheft == 'False') and (HomeBase == 'City') and (CarValue == 'FiftyThou')):
	Theft ~= choice({'True' : 0.01, 'False' : 0.99})
elif ((AntiTheft == 'False') and (HomeBase == 'City') and (CarValue == 'Million')):
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((AntiTheft == 'False') and (HomeBase == 'Suburb') and (CarValue == 'FiveThou')):
	Theft ~= choice({'True' : 1e-05, 'False' : 0.99999})
elif ((AntiTheft == 'False') and (HomeBase == 'Suburb') and (CarValue == 'TenThou')):
	Theft ~= choice({'True' : 0.0002, 'False' : 0.9998})
elif ((AntiTheft == 'False') and (HomeBase == 'Suburb') and (CarValue == 'TwentyThou')):
	Theft ~= choice({'True' : 0.0005, 'False' : 0.9995})
elif ((AntiTheft == 'False') and (HomeBase == 'Suburb') and (CarValue == 'FiftyThou')):
	Theft ~= choice({'True' : 0.0005, 'False' : 0.9995})
elif ((AntiTheft == 'False') and (HomeBase == 'Suburb') and (CarValue == 'Million')):
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})
elif ((AntiTheft == 'False') and (HomeBase == 'Rural') and (CarValue == 'FiveThou')):
	Theft ~= choice({'True' : 1e-05, 'False' : 0.99999})
elif ((AntiTheft == 'False') and (HomeBase == 'Rural') and (CarValue == 'TenThou')):
	Theft ~= choice({'True' : 0.0001, 'False' : 0.9999})
elif ((AntiTheft == 'False') and (HomeBase == 'Rural') and (CarValue == 'TwentyThou')):
	Theft ~= choice({'True' : 0.0002, 'False' : 0.9998})
elif ((AntiTheft == 'False') and (HomeBase == 'Rural') and (CarValue == 'FiftyThou')):
	Theft ~= choice({'True' : 0.0002, 'False' : 0.9998})
else:
	Theft ~= choice({'True' : 1e-06, 'False' : 0.999999})


if ((Antilock == 'True') and (Mileage == 'FiveThou') and (DrivQuality == 'Poor')):
	Accident ~= choice({'None' : 0.7, 'Mild' : 0.2, 'Moderate' : 0.07, 'Severe' : 0.03})
elif ((Antilock == 'True') and (Mileage == 'FiveThou') and (DrivQuality == 'Normal')):
	Accident ~= choice({'None' : 0.99, 'Mild' : 0.007, 'Moderate' : 0.002, 'Severe' : 0.001})
elif ((Antilock == 'True') and (Mileage == 'FiveThou') and (DrivQuality == 'Excellent')):
	Accident ~= choice({'None' : 0.999, 'Mild' : 0.0007, 'Moderate' : 0.0002, 'Severe' : 0.0001})
elif ((Antilock == 'True') and (Mileage == 'TwentyThou') and (DrivQuality == 'Poor')):
	Accident ~= choice({'None' : 0.4, 'Mild' : 0.3, 'Moderate' : 0.2, 'Severe' : 0.1})
elif ((Antilock == 'True') and (Mileage == 'TwentyThou') and (DrivQuality == 'Normal')):
	Accident ~= choice({'None' : 0.98, 'Mild' : 0.01, 'Moderate' : 0.005, 'Severe' : 0.005})
elif ((Antilock == 'True') and (Mileage == 'TwentyThou') and (DrivQuality == 'Excellent')):
	Accident ~= choice({'None' : 0.995, 'Mild' : 0.003, 'Moderate' : 0.001, 'Severe' : 0.001})
elif ((Antilock == 'True') and (Mileage == 'FiftyThou') and (DrivQuality == 'Poor')):
	Accident ~= choice({'None' : 0.3, 'Mild' : 0.3, 'Moderate' : 0.2, 'Severe' : 0.2})
elif ((Antilock == 'True') and (Mileage == 'FiftyThou') and (DrivQuality == 'Normal')):
	Accident ~= choice({'None' : 0.97, 'Mild' : 0.02, 'Moderate' : 0.007, 'Severe' : 0.003})
elif ((Antilock == 'True') and (Mileage == 'FiftyThou') and (DrivQuality == 'Excellent')):
	Accident ~= choice({'None' : 0.99, 'Mild' : 0.007, 'Moderate' : 0.002, 'Severe' : 0.001})
elif ((Antilock == 'True') and (Mileage == 'Domino') and (DrivQuality == 'Poor')):
	Accident ~= choice({'None' : 0.2, 'Mild' : 0.2, 'Moderate' : 0.3, 'Severe' : 0.3})
elif ((Antilock == 'True') and (Mileage == 'Domino') and (DrivQuality == 'Normal')):
	Accident ~= choice({'None' : 0.95, 'Mild' : 0.03, 'Moderate' : 0.01, 'Severe' : 0.01})
elif ((Antilock == 'True') and (Mileage == 'Domino') and (DrivQuality == 'Excellent')):
	Accident ~= choice({'None' : 0.985, 'Mild' : 0.01, 'Moderate' : 0.003, 'Severe' : 0.002})
elif ((Antilock == 'False') and (Mileage == 'FiveThou') and (DrivQuality == 'Poor')):
	Accident ~= choice({'None' : 0.6, 'Mild' : 0.2, 'Moderate' : 0.1, 'Severe' : 0.1})
elif ((Antilock == 'False') and (Mileage == 'FiveThou') and (DrivQuality == 'Normal')):
	Accident ~= choice({'None' : 0.98, 'Mild' : 0.01, 'Moderate' : 0.005, 'Severe' : 0.005})
elif ((Antilock == 'False') and (Mileage == 'FiveThou') and (DrivQuality == 'Excellent')):
	Accident ~= choice({'None' : 0.995, 'Mild' : 0.003, 'Moderate' : 0.001, 'Severe' : 0.001})
elif ((Antilock == 'False') and (Mileage == 'TwentyThou') and (DrivQuality == 'Poor')):
	Accident ~= choice({'None' : 0.3, 'Mild' : 0.2, 'Moderate' : 0.2, 'Severe' : 0.3})
elif ((Antilock == 'False') and (Mileage == 'TwentyThou') and (DrivQuality == 'Normal')):
	Accident ~= choice({'None' : 0.96, 'Mild' : 0.02, 'Moderate' : 0.015, 'Severe' : 0.005})
elif ((Antilock == 'False') and (Mileage == 'TwentyThou') and (DrivQuality == 'Excellent')):
	Accident ~= choice({'None' : 0.99, 'Mild' : 0.007, 'Moderate' : 0.002, 'Severe' : 0.001})
elif ((Antilock == 'False') and (Mileage == 'FiftyThou') and (DrivQuality == 'Poor')):
	Accident ~= choice({'None' : 0.2, 'Mild' : 0.2, 'Moderate' : 0.2, 'Severe' : 0.4})
elif ((Antilock == 'False') and (Mileage == 'FiftyThou') and (DrivQuality == 'Normal')):
	Accident ~= choice({'None' : 0.95, 'Mild' : 0.03, 'Moderate' : 0.015, 'Severe' : 0.005})
elif ((Antilock == 'False') and (Mileage == 'FiftyThou') and (DrivQuality == 'Excellent')):
	Accident ~= choice({'None' : 0.98, 'Mild' : 0.01, 'Moderate' : 0.005, 'Severe' : 0.005})
elif ((Antilock == 'False') and (Mileage == 'Domino') and (DrivQuality == 'Poor')):
	Accident ~= choice({'None' : 0.1, 'Mild' : 0.1, 'Moderate' : 0.3, 'Severe' : 0.5})
elif ((Antilock == 'False') and (Mileage == 'Domino') and (DrivQuality == 'Normal')):
	Accident ~= choice({'None' : 0.94, 'Mild' : 0.03, 'Moderate' : 0.02, 'Severe' : 0.01})
else:
	Accident ~= choice({'None' : 0.98, 'Mild' : 0.01, 'Moderate' : 0.007, 'Severe' : 0.003})


if ((Accident == 'None')):
	ILiCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'Mild')):
	ILiCost ~= choice({'Thousand' : 0.999, 'TenThou' : 0.000998, 'HundredThou' : 1e-06, 'Million' : 1e-06})
elif ((Accident == 'Moderate')):
	ILiCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.05, 'HundredThou' : 0.03, 'Million' : 0.02})
else:
	ILiCost ~= choice({'Thousand' : 0.8, 'TenThou' : 0.1, 'HundredThou' : 0.06, 'Million' : 0.04})


if ((Accident == 'None') and (Age == 'Adolescent') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Adolescent') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Adolescent') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Adolescent') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Adult') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Adult') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Adult') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Adult') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Senior') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Senior') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Senior') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (Age == 'Senior') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'Mild') and (Age == 'Adolescent') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 0.96, 'TenThou' : 0.03, 'HundredThou' : 0.009, 'Million' : 0.001})
elif ((Accident == 'Mild') and (Age == 'Adolescent') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 0.98, 'TenThou' : 0.019, 'HundredThou' : 0.0009, 'Million' : 0.0001})
elif ((Accident == 'Mild') and (Age == 'Adolescent') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.0099, 'HundredThou' : 9e-05, 'Million' : 1e-05})
elif ((Accident == 'Mild') and (Age == 'Adolescent') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 0.999, 'TenThou' : 0.00099, 'HundredThou' : 9e-06, 'Million' : 1e-06})
elif ((Accident == 'Mild') and (Age == 'Adult') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 0.96, 'TenThou' : 0.03, 'HundredThou' : 0.009, 'Million' : 0.001})
elif ((Accident == 'Mild') and (Age == 'Adult') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 0.98, 'TenThou' : 0.019, 'HundredThou' : 0.0009, 'Million' : 0.0001})
elif ((Accident == 'Mild') and (Age == 'Adult') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.0099, 'HundredThou' : 9e-05, 'Million' : 1e-05})
elif ((Accident == 'Mild') and (Age == 'Adult') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 0.999, 'TenThou' : 0.00099, 'HundredThou' : 9e-06, 'Million' : 1e-06})
elif ((Accident == 'Mild') and (Age == 'Senior') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.07, 'HundredThou' : 0.02, 'Million' : 0.01})
elif ((Accident == 'Mild') and (Age == 'Senior') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.04, 'HundredThou' : 0.007, 'Million' : 0.003})
elif ((Accident == 'Mild') and (Age == 'Senior') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 0.97, 'TenThou' : 0.025, 'HundredThou' : 0.003, 'Million' : 0.002})
elif ((Accident == 'Mild') and (Age == 'Senior') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.007, 'HundredThou' : 0.002, 'Million' : 0.001})
elif ((Accident == 'Moderate') and (Age == 'Adolescent') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.1})
elif ((Accident == 'Moderate') and (Age == 'Adolescent') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 0.8, 'TenThou' : 0.15, 'HundredThou' : 0.03, 'Million' : 0.02})
elif ((Accident == 'Moderate') and (Age == 'Adolescent') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.02, 'HundredThou' : 0.02, 'Million' : 0.01})
elif ((Accident == 'Moderate') and (Age == 'Adolescent') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.007, 'HundredThou' : 0.002, 'Million' : 0.001})
elif ((Accident == 'Moderate') and (Age == 'Adult') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.1})
elif ((Accident == 'Moderate') and (Age == 'Adult') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 0.8, 'TenThou' : 0.15, 'HundredThou' : 0.03, 'Million' : 0.02})
elif ((Accident == 'Moderate') and (Age == 'Adult') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.02, 'HundredThou' : 0.02, 'Million' : 0.01})
elif ((Accident == 'Moderate') and (Age == 'Adult') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.007, 'HundredThou' : 0.002, 'Million' : 0.001})
elif ((Accident == 'Moderate') and (Age == 'Senior') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 0.3, 'TenThou' : 0.3, 'HundredThou' : 0.2, 'Million' : 0.2})
elif ((Accident == 'Moderate') and (Age == 'Senior') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.1})
elif ((Accident == 'Moderate') and (Age == 'Senior') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.07, 'HundredThou' : 0.02, 'Million' : 0.01})
elif ((Accident == 'Moderate') and (Age == 'Senior') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.03, 'HundredThou' : 0.01, 'Million' : 0.01})
elif ((Accident == 'Severe') and (Age == 'Adolescent') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 0.3, 'TenThou' : 0.3, 'HundredThou' : 0.2, 'Million' : 0.2})
elif ((Accident == 'Severe') and (Age == 'Adolescent') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.1})
elif ((Accident == 'Severe') and (Age == 'Adolescent') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.07, 'HundredThou' : 0.02, 'Million' : 0.01})
elif ((Accident == 'Severe') and (Age == 'Adolescent') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.03, 'HundredThou' : 0.01, 'Million' : 0.01})
elif ((Accident == 'Severe') and (Age == 'Adult') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 0.3, 'TenThou' : 0.3, 'HundredThou' : 0.2, 'Million' : 0.2})
elif ((Accident == 'Severe') and (Age == 'Adult') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.2, 'Million' : 0.1})
elif ((Accident == 'Severe') and (Age == 'Adult') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.07, 'HundredThou' : 0.02, 'Million' : 0.01})
elif ((Accident == 'Severe') and (Age == 'Adult') and (Cushioning == 'Excellent')):
	MedCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.03, 'HundredThou' : 0.01, 'Million' : 0.01})
elif ((Accident == 'Severe') and (Age == 'Senior') and (Cushioning == 'Poor')):
	MedCost ~= choice({'Thousand' : 0.2, 'TenThou' : 0.2, 'HundredThou' : 0.3, 'Million' : 0.3})
elif ((Accident == 'Severe') and (Age == 'Senior') and (Cushioning == 'Fair')):
	MedCost ~= choice({'Thousand' : 0.3, 'TenThou' : 0.3, 'HundredThou' : 0.2, 'Million' : 0.2})
elif ((Accident == 'Severe') and (Age == 'Senior') and (Cushioning == 'Good')):
	MedCost ~= choice({'Thousand' : 0.6, 'TenThou' : 0.3, 'HundredThou' : 0.07, 'Million' : 0.03})
else:
	MedCost ~= choice({'Thousand' : 0.9, 'TenThou' : 0.05, 'HundredThou' : 0.03, 'Million' : 0.02})


if ((Accident == 'None') and (RuggedAuto == 'EggShell')):
	OtherCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (RuggedAuto == 'Football')):
	OtherCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'None') and (RuggedAuto == 'Tank')):
	OtherCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((Accident == 'Mild') and (RuggedAuto == 'EggShell')):
	OtherCarCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.005, 'HundredThou' : 0.00499, 'Million' : 1e-05})
elif ((Accident == 'Mild') and (RuggedAuto == 'Football')):
	OtherCarCost ~= choice({'Thousand' : 0.9799657, 'TenThou' : 0.00999965, 'HundredThou' : 0.009984651, 'Million' : 4.999825e-05})
elif ((Accident == 'Mild') and (RuggedAuto == 'Tank')):
	OtherCarCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.03, 'HundredThou' : 0.01998, 'Million' : 2e-05})
elif ((Accident == 'Moderate') and (RuggedAuto == 'EggShell')):
	OtherCarCost ~= choice({'Thousand' : 0.6, 'TenThou' : 0.2, 'HundredThou' : 0.19998, 'Million' : 2e-05})
elif ((Accident == 'Moderate') and (RuggedAuto == 'Football')):
	OtherCarCost ~= choice({'Thousand' : 0.5, 'TenThou' : 0.2, 'HundredThou' : 0.29997, 'Million' : 3e-05})
elif ((Accident == 'Moderate') and (RuggedAuto == 'Tank')):
	OtherCarCost ~= choice({'Thousand' : 0.4, 'TenThou' : 0.3, 'HundredThou' : 0.29996, 'Million' : 4e-05})
elif ((Accident == 'Severe') and (RuggedAuto == 'EggShell')):
	OtherCarCost ~= choice({'Thousand' : 0.2, 'TenThou' : 0.4, 'HundredThou' : 0.39996, 'Million' : 4e-05})
elif ((Accident == 'Severe') and (RuggedAuto == 'Football')):
	OtherCarCost ~= choice({'Thousand' : 0.1, 'TenThou' : 0.5, 'HundredThou' : 0.39994, 'Million' : 6e-05})
else:
	OtherCarCost ~= choice({'Thousand' : 0.005, 'TenThou' : 0.55, 'HundredThou' : 0.4449, 'Million' : 0.0001})


if ((Accident == 'None') and (RuggedAuto == 'EggShell')):
	ThisCarDam ~= choice({'None' : 1.0, 'Mild' : 0.0, 'Moderate' : 0.0, 'Severe' : 0.0})
elif ((Accident == 'None') and (RuggedAuto == 'Football')):
	ThisCarDam ~= choice({'None' : 1.0, 'Mild' : 0.0, 'Moderate' : 0.0, 'Severe' : 0.0})
elif ((Accident == 'None') and (RuggedAuto == 'Tank')):
	ThisCarDam ~= choice({'None' : 1.0, 'Mild' : 0.0, 'Moderate' : 0.0, 'Severe' : 0.0})
elif ((Accident == 'Mild') and (RuggedAuto == 'EggShell')):
	ThisCarDam ~= choice({'None' : 0.001, 'Mild' : 0.9, 'Moderate' : 0.098, 'Severe' : 0.001})
elif ((Accident == 'Mild') and (RuggedAuto == 'Football')):
	ThisCarDam ~= choice({'None' : 0.2, 'Mild' : 0.75, 'Moderate' : 0.049999, 'Severe' : 1e-06})
elif ((Accident == 'Mild') and (RuggedAuto == 'Tank')):
	ThisCarDam ~= choice({'None' : 0.7, 'Mild' : 0.29, 'Moderate' : 0.009999, 'Severe' : 1e-06})
elif ((Accident == 'Moderate') and (RuggedAuto == 'EggShell')):
	ThisCarDam ~= choice({'None' : 1e-06, 'Mild' : 0.000999, 'Moderate' : 0.7, 'Severe' : 0.299})
elif ((Accident == 'Moderate') and (RuggedAuto == 'Football')):
	ThisCarDam ~= choice({'None' : 0.001, 'Mild' : 0.099, 'Moderate' : 0.8, 'Severe' : 0.1})
elif ((Accident == 'Moderate') and (RuggedAuto == 'Tank')):
	ThisCarDam ~= choice({'None' : 0.05, 'Mild' : 0.6, 'Moderate' : 0.3, 'Severe' : 0.05})
elif ((Accident == 'Severe') and (RuggedAuto == 'EggShell')):
	ThisCarDam ~= choice({'None' : 1e-06, 'Mild' : 9e-06, 'Moderate' : 9e-05, 'Severe' : 0.9999})
elif ((Accident == 'Severe') and (RuggedAuto == 'Football')):
	ThisCarDam ~= choice({'None' : 1e-06, 'Mild' : 0.000999, 'Moderate' : 0.009, 'Severe' : 0.99})
else:
	ThisCarDam ~= choice({'None' : 0.05, 'Mild' : 0.2, 'Moderate' : 0.2, 'Severe' : 0.55})


if ((ThisCarDam == 'None') and (CarValue == 'FiveThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.2, 'TenThou' : 0.8, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'None') and (CarValue == 'FiveThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'None') and (CarValue == 'TenThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.05, 'TenThou' : 0.95, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'None') and (CarValue == 'TenThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'None') and (CarValue == 'TwentyThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.04, 'TenThou' : 0.01, 'HundredThou' : 0.95, 'Million' : 0.0})
elif ((ThisCarDam == 'None') and (CarValue == 'TwentyThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'None') and (CarValue == 'FiftyThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.04, 'TenThou' : 0.01, 'HundredThou' : 0.95, 'Million' : 0.0})
elif ((ThisCarDam == 'None') and (CarValue == 'FiftyThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'None') and (CarValue == 'Million') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.04, 'TenThou' : 0.01, 'HundredThou' : 0.2, 'Million' : 0.75})
elif ((ThisCarDam == 'None') and (CarValue == 'Million') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 1.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Mild') and (CarValue == 'FiveThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.15, 'TenThou' : 0.85, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Mild') and (CarValue == 'FiveThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.05, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Mild') and (CarValue == 'TenThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.03, 'TenThou' : 0.97, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Mild') and (CarValue == 'TenThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.95, 'TenThou' : 0.05, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Mild') and (CarValue == 'TwentyThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.03, 'TenThou' : 0.02, 'HundredThou' : 0.95, 'Million' : 0.0})
elif ((ThisCarDam == 'Mild') and (CarValue == 'TwentyThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.01, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Mild') and (CarValue == 'FiftyThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.03, 'TenThou' : 0.02, 'HundredThou' : 0.95, 'Million' : 0.0})
elif ((ThisCarDam == 'Mild') and (CarValue == 'FiftyThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.99, 'TenThou' : 0.01, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Mild') and (CarValue == 'Million') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.02, 'TenThou' : 0.03, 'HundredThou' : 0.25, 'Million' : 0.7})
elif ((ThisCarDam == 'Mild') and (CarValue == 'Million') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.98, 'TenThou' : 0.01, 'HundredThou' : 0.01, 'Million' : 0.0})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'FiveThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.05, 'TenThou' : 0.95, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'FiveThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.25, 'TenThou' : 0.75, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'TenThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.01, 'TenThou' : 0.99, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'TenThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.15, 'TenThou' : 0.85, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'TwentyThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.001, 'TenThou' : 0.001, 'HundredThou' : 0.998, 'Million' : 0.0})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'TwentyThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.01, 'TenThou' : 0.01, 'HundredThou' : 0.98, 'Million' : 0.0})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'FiftyThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.001, 'TenThou' : 0.001, 'HundredThou' : 0.998, 'Million' : 0.0})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'FiftyThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.005, 'TenThou' : 0.005, 'HundredThou' : 0.99, 'Million' : 0.0})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'Million') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.001, 'TenThou' : 0.001, 'HundredThou' : 0.018, 'Million' : 0.98})
elif ((ThisCarDam == 'Moderate') and (CarValue == 'Million') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.003, 'TenThou' : 0.003, 'HundredThou' : 0.044, 'Million' : 0.95})
elif ((ThisCarDam == 'Severe') and (CarValue == 'FiveThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 0.03, 'TenThou' : 0.97, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Severe') and (CarValue == 'FiveThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.05, 'TenThou' : 0.95, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Severe') and (CarValue == 'TenThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 0.999999, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Severe') and (CarValue == 'TenThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.01, 'TenThou' : 0.99, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((ThisCarDam == 'Severe') and (CarValue == 'TwentyThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 1e-06, 'HundredThou' : 0.999998, 'Million' : 0.0})
elif ((ThisCarDam == 'Severe') and (CarValue == 'TwentyThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.005, 'TenThou' : 0.005, 'HundredThou' : 0.99, 'Million' : 0.0})
elif ((ThisCarDam == 'Severe') and (CarValue == 'FiftyThou') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 1e-06, 'HundredThou' : 0.999998, 'Million' : 0.0})
elif ((ThisCarDam == 'Severe') and (CarValue == 'FiftyThou') and (Theft == 'False')):
	ThisCarCost ~= choice({'Thousand' : 0.001, 'TenThou' : 0.001, 'HundredThou' : 0.998, 'Million' : 0.0})
elif ((ThisCarDam == 'Severe') and (CarValue == 'Million') and (Theft == 'True')):
	ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 1e-06, 'HundredThou' : 0.009998, 'Million' : 0.99})
else:
	ThisCarCost ~= choice({'Thousand' : 1e-06, 'TenThou' : 1e-06, 'HundredThou' : 0.029998, 'Million' : 0.97})


if ((OtherCarCost == 'Thousand') and (ThisCarCost == 'Thousand')):
	PropCost ~= choice({'Thousand' : 0.7, 'TenThou' : 0.3, 'HundredThou' : 0.0, 'Million' : 0.0})
elif ((OtherCarCost == 'Thousand') and (ThisCarCost == 'TenThou')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.95, 'HundredThou' : 0.05, 'Million' : 0.0})
elif ((OtherCarCost == 'Thousand') and (ThisCarCost == 'HundredThou')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.98, 'Million' : 0.02})
elif ((OtherCarCost == 'Thousand') and (ThisCarCost == 'Million')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
elif ((OtherCarCost == 'TenThou') and (ThisCarCost == 'Thousand')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.95, 'HundredThou' : 0.05, 'Million' : 0.0})
elif ((OtherCarCost == 'TenThou') and (ThisCarCost == 'TenThou')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.6, 'HundredThou' : 0.4, 'Million' : 0.0})
elif ((OtherCarCost == 'TenThou') and (ThisCarCost == 'HundredThou')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.95, 'Million' : 0.05})
elif ((OtherCarCost == 'TenThou') and (ThisCarCost == 'Million')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
elif ((OtherCarCost == 'HundredThou') and (ThisCarCost == 'Thousand')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.98, 'Million' : 0.02})
elif ((OtherCarCost == 'HundredThou') and (ThisCarCost == 'TenThou')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.8, 'Million' : 0.2})
elif ((OtherCarCost == 'HundredThou') and (ThisCarCost == 'HundredThou')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.6, 'Million' : 0.4})
elif ((OtherCarCost == 'HundredThou') and (ThisCarCost == 'Million')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
elif ((OtherCarCost == 'Million') and (ThisCarCost == 'Thousand')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
elif ((OtherCarCost == 'Million') and (ThisCarCost == 'TenThou')):
	PropCost ~= choice({'Thousand' : 0.0, 'TenThou' : 0.0, 'HundredThou' : 0.0, 'Million' : 1.0})
elif ((OtherCarCost == 'Million') and (ThisCarCost == 'HundredThou')):
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
events = [Cushioning << {'Good'},HomeBase << {'City'},DrivingSkill << {'Normal'},DrivQuality << {'Excellent'},MedCost << {'HundredThou'},OtherCar << {'True'},Mileage << {'FiftyThou'},ThisCarCost << {'Million'},RiskAversion << {'Psychopath'},Antilock << {'False'},OtherCar << {'True'},HomeBase << {'Secure'},ILiCost << {'TenThou'},Theft << {'False'},Cushioning << {'Excellent'},Airbag << {'True'},ILiCost << {'TenThou'},DrivQuality << {'Normal'},ThisCarDam << {'Moderate'},Mileage << {'TwentyThou'},ThisCarDam << {'Mild'},DrivHist << {'Zero'},AntiTheft << {'False'},ThisCarCost << {'Million'},RuggedAuto << {'Football'},CarValue << {'FiveThou'},Accident << {'Mild'},VehicleYear << {'Older'},Age << {'Senior'},DrivQuality << {'Poor'},RiskAversion << {'Cautious'},MakeModel << {'Economy'},GoodStudent << {'False'},MedCost << {'Million'},AntiTheft << {'False'},DrivHist << {'One'},OtherCarCost << {'Million'},OtherCarCost << {'Million'},RuggedAuto << {'EggShell'},OtherCar << {'False'},VehicleYear << {'Older'},OtherCar << {'False'},OtherCarCost << {'Million'},HomeBase << {'Rural'},ThisCarCost << {'HundredThou'},SocioEcon << {'Wealthy'},ILiCost << {'TenThou'},ILiCost << {'Thousand'},OtherCarCost << {'TenThou'},RiskAversion << {'Adventurous'},(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Million'}) & (MakeModel << {'Economy'}) & (MedCost << {'HundredThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'Million'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'Million'}) & (Cushioning << {'Good'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'TenThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'Luxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'Mild'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'Thousand'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'HundredThou'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'Thousand'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Million'}) & (MakeModel << {'Economy'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'TenThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Million'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'Thousand'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'Million'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Million'}) & (MakeModel << {'Economy'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'Luxury'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Secure'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Million'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Million'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Million'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'City'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'Thousand'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Secure'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'TenThou'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'Thousand'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Million'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Million'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'One'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'Million'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Million'}) & (MakeModel << {'Economy'}) & (MedCost << {'TenThou'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'Million'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Million'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'Thousand'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Middle'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Million'}) & (MakeModel << {'Economy'}) & (MedCost << {'Million'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'Million'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'One'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Million'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'One'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'City'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Thousand'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Million'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'True'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Million'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'TenThou'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Million'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'HundredThou'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Luxury'}) & (MedCost << {'HundredThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'False'}) & (OtherCarCost << {'Million'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'Million'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'Million'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'True'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'Million'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Million'}) & (Mileage << {'FiveThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Thousand'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Million'}) & (RiskAversion << {'Cautious'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Moderate'}) & (VehicleYear << {'Current'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'True'}) & (Antilock << {'False'}) & (CarValue << {'FiftyThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'City'}) & (ILiCost << {'Million'}) & (MakeModel << {'SuperLuxury'}) & (MedCost << {'Million'}) & (Mileage << {'Domino'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Million'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'UpperMiddle'}) & (Theft << {'False'}) & (ThisCarCost << {'Million'}) & (ThisCarDam << {'Mild'}) & (VehicleYear << {'Current'}),(Accident << {'Moderate'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'TwentyThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'False'}) & (HomeBase << {'City'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Million'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'Thousand'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'EggShell'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adolescent'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Poor'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Normal'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'True'}) & (HomeBase << {'City'}) & (ILiCost << {'HundredThou'}) & (MakeModel << {'FamilySedan'}) & (MedCost << {'Thousand'}) & (Mileage << {'FiftyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'}),(Accident << {'None'}) & (Age << {'Adult'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'False'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Excellent'}) & (DrivHist << {'Many'}) & (DrivQuality << {'Excellent'}) & (DrivingSkill << {'SubStandard'}) & (GoodStudent << {'False'}) & (HomeBase << {'Rural'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Luxury'}) & (MedCost << {'Million'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'False'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'Thousand'}) & (RiskAversion << {'Psychopath'}) & (RuggedAuto << {'Football'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Prole'}) & (Theft << {'False'}) & (ThisCarCost << {'Thousand'}) & (ThisCarDam << {'Severe'}) & (VehicleYear << {'Older'}),(Accident << {'Mild'}) & (Age << {'Senior'}) & (Airbag << {'True'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'FiveThou'}) & (Cushioning << {'Good'}) & (DrivHist << {'One'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Normal'}) & (GoodStudent << {'True'}) & (HomeBase << {'Suburb'}) & (ILiCost << {'TenThou'}) & (MakeModel << {'Economy'}) & (MedCost << {'TenThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'HundredThou'}) & (PropCost << {'TenThou'}) & (RiskAversion << {'Normal'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'False'}) & (SocioEcon << {'Middle'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Current'}),(Accident << {'Severe'}) & (Age << {'Adult'}) & (Airbag << {'False'}) & (AntiTheft << {'False'}) & (Antilock << {'True'}) & (CarValue << {'TenThou'}) & (Cushioning << {'Fair'}) & (DrivHist << {'Zero'}) & (DrivQuality << {'Poor'}) & (DrivingSkill << {'Expert'}) & (GoodStudent << {'True'}) & (HomeBase << {'Secure'}) & (ILiCost << {'Million'}) & (MakeModel << {'SportsCar'}) & (MedCost << {'HundredThou'}) & (Mileage << {'TwentyThou'}) & (OtherCar << {'True'}) & (OtherCarCost << {'TenThou'}) & (PropCost << {'HundredThou'}) & (RiskAversion << {'Adventurous'}) & (RuggedAuto << {'Tank'}) & (SeniorTrain << {'True'}) & (SocioEcon << {'Wealthy'}) & (Theft << {'True'}) & (ThisCarCost << {'TenThou'}) & (ThisCarDam << {'None'}) & (VehicleYear << {'Older'})]
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
