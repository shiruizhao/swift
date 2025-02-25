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
CBODD_12_00 ~= choice({'15_MG_L' : 0.0,'20_MG_L' : 1.0,'25_MG_L' : 0.0,'30_MG_L' : 0.0})


CBODN_12_00 ~= choice({'5_MG_L' : 0.0,'10_MG_L' : 1.0,'15_MG_L' : 0.0,'20_MG_L' : 0.0})


CKND_12_00 ~= choice({'2_MG_L' : 0.0,'4_MG_L' : 1.0,'6_MG_L' : 0.0})


CKNI_12_00 ~= choice({'20_MG_L' : 1.0/3,'30_MG_L' : 1.0/3,'40_MG_L' : 1.0/3})


if (CKNI_12_00 == '20_MG_L'):
	CKNI_12_15 ~= choice({'20_MG_L' : 0.48, '30_MG_L' : 0.48, '40_MG_L' : 0.040000000000000036})
elif(CKNI_12_00 == '30_MG_L'):
	CKNI_12_15 ~= choice({'20_MG_L' : 0.2, '30_MG_L' : 0.6, '40_MG_L' : 0.19999999999999996})
else:
	CKNI_12_15 ~= choice({'20_MG_L' : 0.04, '30_MG_L' : 0.48, '40_MG_L' : 0.48})


if (CKNI_12_15 == '20_MG_L'):
	CKNI_12_30 ~= choice({'20_MG_L' : 0.48, '30_MG_L' : 0.48, '40_MG_L' : 0.040000000000000036})
elif(CKNI_12_15 == '30_MG_L'):
	CKNI_12_30 ~= choice({'20_MG_L' : 0.2, '30_MG_L' : 0.6, '40_MG_L' : 0.19999999999999996})
else:
	CKNI_12_30 ~= choice({'20_MG_L' : 0.04, '30_MG_L' : 0.48, '40_MG_L' : 0.48})


if (CKNI_12_30 == '20_MG_L'):
	CKNI_12_45 ~= choice({'20_MG_L' : 0.48, '30_MG_L' : 0.48, '40_MG_L' : 0.040000000000000036})
elif(CKNI_12_30 == '30_MG_L'):
	CKNI_12_45 ~= choice({'20_MG_L' : 0.2, '30_MG_L' : 0.6, '40_MG_L' : 0.19999999999999996})
else:
	CKNI_12_45 ~= choice({'20_MG_L' : 0.04, '30_MG_L' : 0.48, '40_MG_L' : 0.48})


CKNN_12_00 ~= choice({'0_5_MG_L' : 0.0,'1_MG_L' : 1.0,'2_MG_L' : 0.0})


if (CKND_12_00 == '2_MG_L'):
	if (CKNN_12_00 == '0_5_MG_L'):
		CKNN_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0})
	elif(CKNN_12_00 == '1_MG_L'):
		CKNN_12_15 ~= choice({'0_5_MG_L' : 0.4459, '1_MG_L' : 0.5541, '2_MG_L' : 0.0})
	else:
		CKNN_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3627, '2_MG_L' : 0.6373})
elif(CKND_12_00 == '4_MG_L'):
	if(CKNN_12_00 == '0_5_MG_L'):
		CKNN_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0})
	elif(CKNN_12_00 == '1_MG_L'):
		CKNN_12_15 ~= choice({'0_5_MG_L' : 0.2499, '1_MG_L' : 0.7501, '2_MG_L' : 0.0})
	else:
		CKNN_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2646, '2_MG_L' : 0.7354})
else:
	if(CKNN_12_00 == '0_5_MG_L'):
		CKNN_12_15 ~= choice({'0_5_MG_L' : 0.8234, '1_MG_L' : 0.1766, '2_MG_L' : 0.0})
	elif(CKNN_12_00 == '1_MG_L'):
		CKNN_12_15 ~= choice({'0_5_MG_L' : 0.0538, '1_MG_L' : 0.9462, '2_MG_L' : 0.0})
	else:
		CKNN_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1666, '2_MG_L' : 0.8334})


CNOD_12_00 ~= choice({'0_5_MG_L' : 0.0,'1_MG_L' : 1.0,'2_MG_L' : 0.0,'4_MG_L' : 0.0})


CNON_12_00 ~= choice({'2_MG_L' : 0.0,'4_MG_L' : 1.0,'6_MG_L' : 0.0,'10_MG_L' : 0.0})


if (CNOD_12_00 == '0_5_MG_L'):
	if (CBODN_12_00 == '5_MG_L'):
		if (CKNN_12_00 == '0_5_MG_L'):
			if (CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9555, '4_MG_L' : 0.0445, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0056, '4_MG_L' : 0.9944, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.055, '6_MG_L' : 0.945, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0767, '10_MG_L' : 0.9233})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9102, '4_MG_L' : 0.0898, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9602, '6_MG_L' : 0.0398, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0096, '6_MG_L' : 0.9904, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.054, '10_MG_L' : 0.946})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8648, '4_MG_L' : 0.1352, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9149, '6_MG_L' : 0.0851, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9822, '10_MG_L' : 0.017800000000000038})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0313, '10_MG_L' : 0.9687})
	elif(CBODN_12_00 == '10_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9618, '4_MG_L' : 0.0382, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0125, '4_MG_L' : 0.9875, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0622, '6_MG_L' : 0.9378, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0804, '10_MG_L' : 0.9196})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9164, '4_MG_L' : 0.0836, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9672, '6_MG_L' : 0.0328, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0169, '6_MG_L' : 0.9831, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0577, '10_MG_L' : 0.9423})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8711, '4_MG_L' : 0.1289, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9219, '6_MG_L' : 0.0781, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9858, '10_MG_L' : 0.01419999999999999})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0351, '10_MG_L' : 0.9649})
	elif(CBODN_12_00 == '15_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9662, '4_MG_L' : 0.0338, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0175, '4_MG_L' : 0.9825, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0674, '6_MG_L' : 0.9326, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0831, '10_MG_L' : 0.9169})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9209, '4_MG_L' : 0.0791, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9722, '6_MG_L' : 0.0278, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.022, '6_MG_L' : 0.978, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0604, '10_MG_L' : 0.9396})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8756, '4_MG_L' : 0.1244, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9268, '6_MG_L' : 0.0732, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9884, '10_MG_L' : 0.011600000000000055})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0377, '10_MG_L' : 0.9623})
	else:
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9696, '4_MG_L' : 0.0304, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0212, '4_MG_L' : 0.9788, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0712, '6_MG_L' : 0.9288, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0851, '10_MG_L' : 0.9149})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9243, '4_MG_L' : 0.0757, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9759, '6_MG_L' : 0.0241, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0259, '6_MG_L' : 0.9741, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0624, '10_MG_L' : 0.9376})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8789, '4_MG_L' : 0.1211, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9306, '6_MG_L' : 0.0694, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9903, '10_MG_L' : 0.009700000000000042})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0397, '10_MG_L' : 0.9603})
elif(CNOD_12_00 == '1_MG_L'):
	if(CBODN_12_00 == '5_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9432, '4_MG_L' : 0.0568, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9933, '6_MG_L' : 0.0067, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0427, '6_MG_L' : 0.9573, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0706, '10_MG_L' : 0.9294})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8979, '4_MG_L' : 0.1021, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.948, '6_MG_L' : 0.052, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9987, '10_MG_L' : 0.0012999999999999678})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0479, '10_MG_L' : 0.9521})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8526, '4_MG_L' : 0.1474, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9026, '6_MG_L' : 0.0974, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.976, '10_MG_L' : 0.02400000000000002})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0252, '10_MG_L' : 0.9748})
	elif(CBODN_12_00 == '10_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9495, '4_MG_L' : 0.0505, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0003, '4_MG_L' : 0.9997, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.05, '6_MG_L' : 0.95, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0743, '10_MG_L' : 0.9257})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9042, '4_MG_L' : 0.0958, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9549, '6_MG_L' : 0.0451, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0046, '6_MG_L' : 0.9954, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0516, '10_MG_L' : 0.9484})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8588, '4_MG_L' : 0.1412, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9096, '6_MG_L' : 0.0904, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9796, '10_MG_L' : 0.020399999999999974})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0289, '10_MG_L' : 0.9711})
	elif(CBODN_12_00 == '15_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.954, '4_MG_L' : 0.046, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0052, '4_MG_L' : 0.9948, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0551, '6_MG_L' : 0.9449, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0769, '10_MG_L' : 0.9231})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9086, '4_MG_L' : 0.0914, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9599, '6_MG_L' : 0.0401, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0098, '6_MG_L' : 0.9902, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0543, '10_MG_L' : 0.9457})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8633, '4_MG_L' : 0.1367, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9146, '6_MG_L' : 0.0854, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9822, '10_MG_L' : 0.017800000000000038})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0316, '10_MG_L' : 0.9684})
	else:
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9573, '4_MG_L' : 0.0427, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.009, '4_MG_L' : 0.991, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.059, '6_MG_L' : 0.941, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0789, '10_MG_L' : 0.9211})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.912, '4_MG_L' : 0.088, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9636, '6_MG_L' : 0.0364, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0137, '6_MG_L' : 0.9863, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0563, '10_MG_L' : 0.9437})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8667, '4_MG_L' : 0.1333, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9183, '6_MG_L' : 0.0817, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9842, '10_MG_L' : 0.015800000000000036})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0336, '10_MG_L' : 0.9664})
elif(CNOD_12_00 == '2_MG_L'):
	if(CBODN_12_00 == '5_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9187, '4_MG_L' : 0.0813, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9688, '6_MG_L' : 0.0312, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0182, '6_MG_L' : 0.9818, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0583, '10_MG_L' : 0.9417})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8734, '4_MG_L' : 0.1266, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9235, '6_MG_L' : 0.0765, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9864, '10_MG_L' : 0.013599999999999945})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0356, '10_MG_L' : 0.9644})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8281, '4_MG_L' : 0.1719, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8781, '6_MG_L' : 0.1219, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9638, '10_MG_L' : 0.03620000000000001})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.013, '10_MG_L' : 0.987})
	elif(CBODN_12_00 == '10_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.925, '4_MG_L' : 0.075, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9758, '6_MG_L' : 0.0242, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0254, '6_MG_L' : 0.9746, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.062, '10_MG_L' : 0.938})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8797, '4_MG_L' : 0.1203, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9304, '6_MG_L' : 0.0696, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9901, '10_MG_L' : 0.00990000000000002})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0394, '10_MG_L' : 0.9606})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8343, '4_MG_L' : 0.1657, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8851, '6_MG_L' : 0.1149, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9674, '10_MG_L' : 0.03259999999999996})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0167, '10_MG_L' : 0.9833})
	elif(CBODN_12_00 == '15_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9295, '4_MG_L' : 0.0705, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9807, '6_MG_L' : 0.0193, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0306, '6_MG_L' : 0.9694, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0647, '10_MG_L' : 0.9353})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8841, '4_MG_L' : 0.1159, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9354, '6_MG_L' : 0.0646, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9926, '10_MG_L' : 0.007399999999999962})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.042, '10_MG_L' : 0.958})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8388, '4_MG_L' : 0.1612, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8901, '6_MG_L' : 0.1099, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.97, '10_MG_L' : 0.030000000000000027})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0194, '10_MG_L' : 0.9806})
	else:
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.9328, '4_MG_L' : 0.0672, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9845, '6_MG_L' : 0.0155, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0345, '6_MG_L' : 0.9655, '10_MG_L' : 0.0})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0667, '10_MG_L' : 0.9333})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8875, '4_MG_L' : 0.1125, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9391, '6_MG_L' : 0.0609, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9946, '10_MG_L' : 0.00539999999999996})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.044, '10_MG_L' : 0.956})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8422, '4_MG_L' : 0.1578, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8938, '6_MG_L' : 0.1062, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9719, '10_MG_L' : 0.028100000000000014})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0214, '10_MG_L' : 0.9786})
else:
	if(CBODN_12_00 == '5_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8697, '4_MG_L' : 0.1303, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9198, '6_MG_L' : 0.0802, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9846, '10_MG_L' : 0.01539999999999997})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0338, '10_MG_L' : 0.9662})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8244, '4_MG_L' : 0.1756, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8744, '6_MG_L' : 0.1256, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9619, '10_MG_L' : 0.03810000000000002})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0111, '10_MG_L' : 0.9889})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.779, '4_MG_L' : 0.221, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8291, '6_MG_L' : 0.1709, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9393, '10_MG_L' : 0.060699999999999976})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})
	elif(CBODN_12_00 == '10_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.876, '4_MG_L' : 0.124, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9267, '6_MG_L' : 0.0733, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9882, '10_MG_L' : 0.011800000000000033})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0375, '10_MG_L' : 0.9625})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8306, '4_MG_L' : 0.1694, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8814, '6_MG_L' : 0.1186, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9655, '10_MG_L' : 0.034499999999999975})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0149, '10_MG_L' : 0.9851})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.7853, '4_MG_L' : 0.2147, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8361, '6_MG_L' : 0.1639, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9429, '10_MG_L' : 0.05710000000000004})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})
	elif(CBODN_12_00 == '15_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8804, '4_MG_L' : 0.1196, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9317, '6_MG_L' : 0.0683, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9908, '10_MG_L' : 0.009199999999999986})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0402, '10_MG_L' : 0.9598})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8351, '4_MG_L' : 0.1649, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8864, '6_MG_L' : 0.1136, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9681, '10_MG_L' : 0.03190000000000004})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0175, '10_MG_L' : 0.9825})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.7898, '4_MG_L' : 0.2102, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.841, '6_MG_L' : 0.159, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9455, '10_MG_L' : 0.05449999999999999})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})
	else:
		if(CKNN_12_00 == '0_5_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8838, '4_MG_L' : 0.1162, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9354, '6_MG_L' : 0.0646, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9927, '10_MG_L' : 0.007299999999999973})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0422, '10_MG_L' : 0.9578})
		elif(CKNN_12_00 == '1_MG_L'):
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.8385, '4_MG_L' : 0.1615, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8901, '6_MG_L' : 0.1099, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9701, '10_MG_L' : 0.029900000000000038})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0195, '10_MG_L' : 0.9805})
		else:
			if(CNON_12_00 == '2_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.7931, '4_MG_L' : 0.2069, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '4_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8448, '6_MG_L' : 0.1552, '10_MG_L' : 0.0})
			elif(CNON_12_00 == '6_MG_L'):
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9474, '10_MG_L' : 0.05259999999999998})
			else:
				CNON_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})


C_NI_12_00 ~= choice({'3' : 0.25,'4' : 0.25,'5' : 0.25,'6' : 0.25})


if (C_NI_12_00 == '3'):
	C_NI_12_15 ~= choice({'3' : 0.5, '4' : 0.4, '5' : 0.1, '6' : 0.0})
elif(C_NI_12_00 == '4'):
	C_NI_12_15 ~= choice({'3' : 0.2, '4' : 0.55, '5' : 0.2, '6' : 0.050000000000000044})
elif(C_NI_12_00 == '5'):
	C_NI_12_15 ~= choice({'3' : 0.1, '4' : 0.3, '5' : 0.5, '6' : 0.09999999999999998})
else:
	C_NI_12_15 ~= choice({'3' : 0.0, '4' : 0.15, '5' : 0.25, '6' : 0.6})


if (C_NI_12_15 == '3'):
	C_NI_12_30 ~= choice({'3' : 0.5, '4' : 0.4, '5' : 0.1, '6' : 0.0})
elif(C_NI_12_15 == '4'):
	C_NI_12_30 ~= choice({'3' : 0.2, '4' : 0.55, '5' : 0.2, '6' : 0.050000000000000044})
elif(C_NI_12_15 == '5'):
	C_NI_12_30 ~= choice({'3' : 0.1, '4' : 0.3, '5' : 0.5, '6' : 0.09999999999999998})
else:
	C_NI_12_30 ~= choice({'3' : 0.0, '4' : 0.15, '5' : 0.25, '6' : 0.6})


if (C_NI_12_30 == '3'):
	C_NI_12_45 ~= choice({'3' : 0.5, '4' : 0.4, '5' : 0.1, '6' : 0.0})
elif(C_NI_12_30 == '4'):
	C_NI_12_45 ~= choice({'3' : 0.2, '4' : 0.55, '5' : 0.2, '6' : 0.050000000000000044})
elif(C_NI_12_30 == '5'):
	C_NI_12_45 ~= choice({'3' : 0.1, '4' : 0.3, '5' : 0.5, '6' : 0.09999999999999998})
else:
	C_NI_12_45 ~= choice({'3' : 0.0, '4' : 0.15, '5' : 0.25, '6' : 0.6})


if (C_NI_12_00 == '3'):
	if (CKNI_12_00 == '20_MG_L'):
		if (CBODD_12_00 == '15_MG_L'):
			if (CNOD_12_00 == '0_5_MG_L'):
				if (CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9791, '20_MG_L' : 0.0209, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9951, '20_MG_L' : 0.0049, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0903, '20_MG_L' : 0.9097, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0585, '20_MG_L' : 0.9415, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0268, '20_MG_L' : 0.9732, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.109, '20_MG_L' : 0.891, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0773, '20_MG_L' : 0.9227, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0455, '20_MG_L' : 0.9545, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.124, '20_MG_L' : 0.876, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0923, '20_MG_L' : 0.9077, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0605, '20_MG_L' : 0.9395, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.134, '20_MG_L' : 0.866, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.1023, '20_MG_L' : 0.8977, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0705, '20_MG_L' : 0.9295, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1362, '25_MG_L' : 0.8638, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1045, '25_MG_L' : 0.8955, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0727, '25_MG_L' : 0.9273, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.157, '25_MG_L' : 0.843, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1253, '25_MG_L' : 0.8747, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0935, '25_MG_L' : 0.9065, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1737, '25_MG_L' : 0.8263, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.142, '25_MG_L' : 0.858, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1102, '25_MG_L' : 0.8898, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1848, '25_MG_L' : 0.8152, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1531, '25_MG_L' : 0.8469, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1213, '25_MG_L' : 0.8787, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1809, '30_MG_L' : 0.8190999999999999})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1491, '30_MG_L' : 0.8509})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1174, '30_MG_L' : 0.8826})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2034, '30_MG_L' : 0.7966})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1716, '30_MG_L' : 0.8284})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1399, '30_MG_L' : 0.8601})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2214, '30_MG_L' : 0.7786})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1896, '30_MG_L' : 0.8104})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1579, '30_MG_L' : 0.8421})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2334, '30_MG_L' : 0.7666})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2016, '30_MG_L' : 0.7984})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1699, '30_MG_L' : 0.8301000000000001})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
	elif(CKNI_12_00 == '30_MG_L'):
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9949, '20_MG_L' : 0.0051, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9632, '20_MG_L' : 0.0368, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9314, '20_MG_L' : 0.0686, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8997, '20_MG_L' : 0.1003, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9793, '20_MG_L' : 0.0207, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9475, '20_MG_L' : 0.0525, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9158, '20_MG_L' : 0.0842, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9921, '20_MG_L' : 0.0079, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9604, '20_MG_L' : 0.0396, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9286, '20_MG_L' : 0.0714, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9689, '20_MG_L' : 0.0311, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9372, '20_MG_L' : 0.0628, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0426, '20_MG_L' : 0.9574, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0109, '20_MG_L' : 0.9891, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9792, '25_MG_L' : 0.0208, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9474, '25_MG_L' : 0.0526, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0614, '20_MG_L' : 0.9386, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0296, '20_MG_L' : 0.9704, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9979, '25_MG_L' : 0.0021, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9662, '25_MG_L' : 0.0338, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0764, '20_MG_L' : 0.9236, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0446, '20_MG_L' : 0.9554, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0129, '20_MG_L' : 0.9871, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9812, '25_MG_L' : 0.0188, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0864, '20_MG_L' : 0.9136, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0546, '20_MG_L' : 0.9454, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0229, '20_MG_L' : 0.9771, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9912, '25_MG_L' : 0.0088, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0886, '25_MG_L' : 0.9114, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0568, '25_MG_L' : 0.9432, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0251, '25_MG_L' : 0.9749, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9933, '30_MG_L' : 0.006700000000000039})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1094, '25_MG_L' : 0.8906, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0777, '25_MG_L' : 0.9223, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0459, '25_MG_L' : 0.9541, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0142, '25_MG_L' : 0.9858, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1261, '25_MG_L' : 0.8739, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0943, '25_MG_L' : 0.9057, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0626, '25_MG_L' : 0.9374, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0308, '25_MG_L' : 0.9692, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1372, '25_MG_L' : 0.8628, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1054, '25_MG_L' : 0.8946, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0737, '25_MG_L' : 0.9263, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.042, '25_MG_L' : 0.958, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1333, '30_MG_L' : 0.8667})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1015, '30_MG_L' : 0.8985})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0698, '30_MG_L' : 0.9302})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.038, '30_MG_L' : 0.962})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1558, '30_MG_L' : 0.8442000000000001})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.124, '30_MG_L' : 0.876})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0923, '30_MG_L' : 0.9077})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0605, '30_MG_L' : 0.9395})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1738, '30_MG_L' : 0.8262})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.142, '30_MG_L' : 0.858})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1103, '30_MG_L' : 0.8897})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0785, '30_MG_L' : 0.9215})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1858, '30_MG_L' : 0.8142})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.154, '30_MG_L' : 0.846})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1223, '30_MG_L' : 0.8777})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0905, '30_MG_L' : 0.9095})
	else:
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
elif(C_NI_12_00 == '4'):
	if(CKNI_12_00 == '20_MG_L'):
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9791, '20_MG_L' : 0.0209, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9951, '20_MG_L' : 0.0049, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0585, '20_MG_L' : 0.9415, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0268, '20_MG_L' : 0.9732, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0773, '20_MG_L' : 0.9227, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0455, '20_MG_L' : 0.9545, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0923, '20_MG_L' : 0.9077, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0605, '20_MG_L' : 0.9395, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.1023, '20_MG_L' : 0.8977, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0705, '20_MG_L' : 0.9295, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1045, '25_MG_L' : 0.8955, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0727, '25_MG_L' : 0.9273, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1253, '25_MG_L' : 0.8747, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0935, '25_MG_L' : 0.9065, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.142, '25_MG_L' : 0.858, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1102, '25_MG_L' : 0.8898, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1531, '25_MG_L' : 0.8469, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1213, '25_MG_L' : 0.8787, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1491, '30_MG_L' : 0.8509})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1174, '30_MG_L' : 0.8826})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1716, '30_MG_L' : 0.8284})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1399, '30_MG_L' : 0.8601})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1896, '30_MG_L' : 0.8104})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1579, '30_MG_L' : 0.8421})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2016, '30_MG_L' : 0.7984})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1699, '30_MG_L' : 0.8301000000000001})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
	elif(CKNI_12_00 == '30_MG_L'):
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
	else:
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8203, '20_MG_L' : 0.1797, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7886, '20_MG_L' : 0.2114, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8364, '20_MG_L' : 0.1636, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8046, '20_MG_L' : 0.1954, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8493, '20_MG_L' : 0.1507, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8175, '20_MG_L' : 0.1825, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8578, '20_MG_L' : 0.1422, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8261, '20_MG_L' : 0.1739, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.868, '25_MG_L' : 0.132, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8363, '25_MG_L' : 0.1637, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8868, '25_MG_L' : 0.1132, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.855, '25_MG_L' : 0.145, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9018, '25_MG_L' : 0.0982, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.87, '25_MG_L' : 0.13, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9118, '25_MG_L' : 0.0882, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.88, '25_MG_L' : 0.12, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.914, '30_MG_L' : 0.08599999999999997})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8822, '30_MG_L' : 0.11780000000000002})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9348, '30_MG_L' : 0.06520000000000004})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9031, '30_MG_L' : 0.09689999999999999})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9515, '30_MG_L' : 0.04849999999999999})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9197, '30_MG_L' : 0.08030000000000004})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9626, '30_MG_L' : 0.03739999999999999})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9308, '30_MG_L' : 0.06920000000000004})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0112, '30_MG_L' : 0.9888})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
elif(C_NI_12_00 == '5'):
	if(CKNI_12_00 == '20_MG_L'):
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9791, '20_MG_L' : 0.0209, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9951, '20_MG_L' : 0.0049, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0268, '20_MG_L' : 0.9732, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0455, '20_MG_L' : 0.9545, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0605, '20_MG_L' : 0.9395, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0705, '20_MG_L' : 0.9295, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0727, '25_MG_L' : 0.9273, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0935, '25_MG_L' : 0.9065, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1102, '25_MG_L' : 0.8898, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1213, '25_MG_L' : 0.8787, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1174, '30_MG_L' : 0.8826})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1399, '30_MG_L' : 0.8601})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1579, '30_MG_L' : 0.8421})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1699, '30_MG_L' : 0.8301000000000001})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
	elif(CKNI_12_00 == '30_MG_L'):
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8997, '20_MG_L' : 0.1003, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8679, '20_MG_L' : 0.1321, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8362, '20_MG_L' : 0.1638, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8045, '20_MG_L' : 0.1955, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9158, '20_MG_L' : 0.0842, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.884, '20_MG_L' : 0.116, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8523, '20_MG_L' : 0.1477, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8205, '20_MG_L' : 0.1795, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9286, '20_MG_L' : 0.0714, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8969, '20_MG_L' : 0.1031, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8651, '20_MG_L' : 0.1349, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8334, '20_MG_L' : 0.1666, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9372, '20_MG_L' : 0.0628, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9054, '20_MG_L' : 0.0946, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8737, '20_MG_L' : 0.1263, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.842, '20_MG_L' : 0.158, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9474, '25_MG_L' : 0.0526, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9157, '25_MG_L' : 0.0843, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8839, '25_MG_L' : 0.1161, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8522, '25_MG_L' : 0.1478, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9662, '25_MG_L' : 0.0338, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9344, '25_MG_L' : 0.0656, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9027, '25_MG_L' : 0.0973, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8709, '25_MG_L' : 0.1291, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9812, '25_MG_L' : 0.0188, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9494, '25_MG_L' : 0.0506, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9177, '25_MG_L' : 0.0823, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8859, '25_MG_L' : 0.1141, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9912, '25_MG_L' : 0.0088, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9594, '25_MG_L' : 0.0406, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9277, '25_MG_L' : 0.0723, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8959, '25_MG_L' : 0.1041, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9933, '30_MG_L' : 0.006700000000000039})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9616, '30_MG_L' : 0.03839999999999999})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9298, '30_MG_L' : 0.07020000000000004})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8981, '30_MG_L' : 0.10189999999999999})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0142, '25_MG_L' : 0.9858, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9824, '30_MG_L' : 0.01759999999999995})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9507, '30_MG_L' : 0.04930000000000001})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9189, '30_MG_L' : 0.08109999999999995})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0308, '25_MG_L' : 0.9692, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9991, '30_MG_L' : 0.0009000000000000119})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9673, '30_MG_L' : 0.03269999999999995})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9356, '30_MG_L' : 0.06440000000000001})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.042, '25_MG_L' : 0.958, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0102, '25_MG_L' : 0.9898, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9785, '30_MG_L' : 0.021499999999999964})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9467, '30_MG_L' : 0.053300000000000014})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.038, '30_MG_L' : 0.962})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0063, '30_MG_L' : 0.9937})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0605, '30_MG_L' : 0.9395})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0288, '30_MG_L' : 0.9712})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0785, '30_MG_L' : 0.9215})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0468, '30_MG_L' : 0.9532})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.015, '30_MG_L' : 0.985})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0905, '30_MG_L' : 0.9095})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0588, '30_MG_L' : 0.9412})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.027, '30_MG_L' : 0.973})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
	else:
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8203, '20_MG_L' : 0.1797, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7886, '20_MG_L' : 0.2114, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7568, '20_MG_L' : 0.2432, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7251, '20_MG_L' : 0.2749, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8364, '20_MG_L' : 0.1636, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8046, '20_MG_L' : 0.1954, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7729, '20_MG_L' : 0.2271, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7412, '20_MG_L' : 0.2588, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8493, '20_MG_L' : 0.1507, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8175, '20_MG_L' : 0.1825, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7858, '20_MG_L' : 0.2142, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.754, '20_MG_L' : 0.246, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8578, '20_MG_L' : 0.1422, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8261, '20_MG_L' : 0.1739, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7943, '20_MG_L' : 0.2057, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7626, '20_MG_L' : 0.2374, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.868, '25_MG_L' : 0.132, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8363, '25_MG_L' : 0.1637, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8045, '25_MG_L' : 0.1955, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7728, '25_MG_L' : 0.2272, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8868, '25_MG_L' : 0.1132, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.855, '25_MG_L' : 0.145, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8233, '25_MG_L' : 0.1767, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7916, '25_MG_L' : 0.2084, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9018, '25_MG_L' : 0.0982, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.87, '25_MG_L' : 0.13, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8383, '25_MG_L' : 0.1617, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8066, '25_MG_L' : 0.1934, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9118, '25_MG_L' : 0.0882, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.88, '25_MG_L' : 0.12, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8483, '25_MG_L' : 0.1517, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8166, '25_MG_L' : 0.1834, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.914, '30_MG_L' : 0.08599999999999997})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8822, '30_MG_L' : 0.11780000000000002})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8505, '30_MG_L' : 0.14949999999999997})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8187, '30_MG_L' : 0.18130000000000002})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9348, '30_MG_L' : 0.06520000000000004})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9031, '30_MG_L' : 0.09689999999999999})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8713, '30_MG_L' : 0.12870000000000004})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8396, '30_MG_L' : 0.1604})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9515, '30_MG_L' : 0.04849999999999999})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9197, '30_MG_L' : 0.08030000000000004})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.888, '30_MG_L' : 0.11199999999999999})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8562, '30_MG_L' : 0.14380000000000004})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9626, '30_MG_L' : 0.03739999999999999})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9308, '30_MG_L' : 0.06920000000000004})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8991, '30_MG_L' : 0.10089999999999999})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8673, '30_MG_L' : 0.13270000000000004})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0112, '30_MG_L' : 0.9888})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
else:
	if(CKNI_12_00 == '20_MG_L'):
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
	elif(CKNI_12_00 == '30_MG_L'):
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8203, '20_MG_L' : 0.1797, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7886, '20_MG_L' : 0.2114, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7568, '20_MG_L' : 0.2432, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8364, '20_MG_L' : 0.1636, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8046, '20_MG_L' : 0.1954, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7729, '20_MG_L' : 0.2271, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8493, '20_MG_L' : 0.1507, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8175, '20_MG_L' : 0.1825, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7858, '20_MG_L' : 0.2142, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8578, '20_MG_L' : 0.1422, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.8261, '20_MG_L' : 0.1739, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7943, '20_MG_L' : 0.2057, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.868, '25_MG_L' : 0.132, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8363, '25_MG_L' : 0.1637, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8045, '25_MG_L' : 0.1955, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8868, '25_MG_L' : 0.1132, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.855, '25_MG_L' : 0.145, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8233, '25_MG_L' : 0.1767, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9018, '25_MG_L' : 0.0982, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.87, '25_MG_L' : 0.13, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8383, '25_MG_L' : 0.1617, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9118, '25_MG_L' : 0.0882, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.88, '25_MG_L' : 0.12, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8483, '25_MG_L' : 0.1517, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.914, '30_MG_L' : 0.08599999999999997})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8822, '30_MG_L' : 0.11780000000000002})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8505, '30_MG_L' : 0.14949999999999997})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9348, '30_MG_L' : 0.06520000000000004})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9031, '30_MG_L' : 0.09689999999999999})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8713, '30_MG_L' : 0.12870000000000004})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9515, '30_MG_L' : 0.04849999999999999})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9197, '30_MG_L' : 0.08030000000000004})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.888, '30_MG_L' : 0.11199999999999999})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9626, '30_MG_L' : 0.03739999999999999})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9308, '30_MG_L' : 0.06920000000000004})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8991, '30_MG_L' : 0.10089999999999999})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0112, '30_MG_L' : 0.9888})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
	else:
		if(CBODD_12_00 == '15_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7568, '20_MG_L' : 0.2432, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7251, '20_MG_L' : 0.2749, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.6933, '20_MG_L' : 0.3067, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.6616, '20_MG_L' : 0.3384, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7729, '20_MG_L' : 0.2271, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7412, '20_MG_L' : 0.2588, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7094, '20_MG_L' : 0.2906, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.6777, '20_MG_L' : 0.3223, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7858, '20_MG_L' : 0.2142, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.754, '20_MG_L' : 0.246, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7223, '20_MG_L' : 0.2777, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.6905, '20_MG_L' : 0.3095, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7943, '20_MG_L' : 0.2057, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7626, '20_MG_L' : 0.2374, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.7308, '20_MG_L' : 0.2692, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.6991, '20_MG_L' : 0.3009, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '20_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8045, '25_MG_L' : 0.1955, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7728, '25_MG_L' : 0.2272, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7411, '25_MG_L' : 0.2589, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7093, '25_MG_L' : 0.2907, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8233, '25_MG_L' : 0.1767, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7916, '25_MG_L' : 0.2084, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7598, '25_MG_L' : 0.2402, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7281, '25_MG_L' : 0.2719, '30_MG_L' : 0.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8383, '25_MG_L' : 0.1617, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8066, '25_MG_L' : 0.1934, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7748, '25_MG_L' : 0.2252, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7431, '25_MG_L' : 0.2569, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8483, '25_MG_L' : 0.1517, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8166, '25_MG_L' : 0.1834, '30_MG_L' : 0.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7848, '25_MG_L' : 0.2152, '30_MG_L' : 0.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7531, '25_MG_L' : 0.2469, '30_MG_L' : 0.0})
		elif(CBODD_12_00 == '25_MG_L'):
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8505, '30_MG_L' : 0.14949999999999997})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8187, '30_MG_L' : 0.18130000000000002})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.787, '30_MG_L' : 0.21299999999999997})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.7552, '30_MG_L' : 0.24480000000000002})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8713, '30_MG_L' : 0.12870000000000004})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8396, '30_MG_L' : 0.1604})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8078, '30_MG_L' : 0.19220000000000004})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.7761, '30_MG_L' : 0.2239})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.888, '30_MG_L' : 0.11199999999999999})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8562, '30_MG_L' : 0.14380000000000004})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8245, '30_MG_L' : 0.1755})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.7927, '30_MG_L' : 0.20730000000000004})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8991, '30_MG_L' : 0.10089999999999999})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8673, '30_MG_L' : 0.13270000000000004})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8356, '30_MG_L' : 0.1644})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8039, '30_MG_L' : 0.19610000000000005})
		else:
			if(CNOD_12_00 == '0_5_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '1_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_00 == '2_MG_L'):
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_00 == '5_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '10_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_00 == '15_MG_L'):
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_15 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})


if (CBODD_12_00 == '15_MG_L'):
	if (CBODN_12_00 == '5_MG_L'):
		if (CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.9557, '10_MG_L' : 0.0443, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.9561, '10_MG_L' : 0.0439, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.9562, '10_MG_L' : 0.0438, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.9564, '10_MG_L' : 0.0436, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_00 == '10_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0406, '10_MG_L' : 0.9594, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0412, '10_MG_L' : 0.9588, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0414, '10_MG_L' : 0.9586, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0416, '10_MG_L' : 0.9584, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_00 == '15_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.1152, '15_MG_L' : 0.8848, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.116, '15_MG_L' : 0.884, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.1163, '15_MG_L' : 0.8837, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.1166, '15_MG_L' : 0.8834, '20_MG_L' : 0.0})
	else:
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1835, '20_MG_L' : 0.8165})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1844, '20_MG_L' : 0.8156})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1847, '20_MG_L' : 0.8153})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.185, '20_MG_L' : 0.815})
elif(CBODD_12_00 == '20_MG_L'):
	if(CBODN_12_00 == '5_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.9067, '10_MG_L' : 0.0933, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.9071, '10_MG_L' : 0.0929, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.9072, '10_MG_L' : 0.0928, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.9073, '10_MG_L' : 0.0927, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_00 == '10_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9916, '15_MG_L' : 0.0084, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9922, '15_MG_L' : 0.0078, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9924, '15_MG_L' : 0.0076, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9926, '15_MG_L' : 0.0074, '20_MG_L' : 0.0})
	elif(CBODN_12_00 == '15_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0662, '15_MG_L' : 0.9338, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.067, '15_MG_L' : 0.933, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0673, '15_MG_L' : 0.9327, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0676, '15_MG_L' : 0.9324, '20_MG_L' : 0.0})
	else:
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1344, '20_MG_L' : 0.8656})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1354, '20_MG_L' : 0.8646})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1357, '20_MG_L' : 0.8643000000000001})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.136, '20_MG_L' : 0.864})
elif(CBODD_12_00 == '25_MG_L'):
	if(CBODN_12_00 == '5_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.8577, '10_MG_L' : 0.1423, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.8581, '10_MG_L' : 0.1419, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.8582, '10_MG_L' : 0.1418, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.8583, '10_MG_L' : 0.1417, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_00 == '10_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9426, '15_MG_L' : 0.0574, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9432, '15_MG_L' : 0.0568, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9434, '15_MG_L' : 0.0566, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9436, '15_MG_L' : 0.0564, '20_MG_L' : 0.0})
	elif(CBODN_12_00 == '15_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0172, '15_MG_L' : 0.9828, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.018, '15_MG_L' : 0.982, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0183, '15_MG_L' : 0.9817, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0185, '15_MG_L' : 0.9815, '20_MG_L' : 0.0})
	else:
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0854, '20_MG_L' : 0.9146})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0863, '20_MG_L' : 0.9137})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0867, '20_MG_L' : 0.9133})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.087, '20_MG_L' : 0.913})
else:
	if(CBODN_12_00 == '5_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.8087, '10_MG_L' : 0.1913, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.809, '10_MG_L' : 0.191, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.8092, '10_MG_L' : 0.1908, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.8093, '10_MG_L' : 0.1907, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_00 == '10_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8936, '15_MG_L' : 0.1064, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8942, '15_MG_L' : 0.1058, '20_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8944, '15_MG_L' : 0.1056, '20_MG_L' : 0.0})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8946, '15_MG_L' : 0.1054, '20_MG_L' : 0.0})
	elif(CBODN_12_00 == '15_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.9682, '20_MG_L' : 0.03180000000000005})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.969, '20_MG_L' : 0.031000000000000028})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.9693, '20_MG_L' : 0.03069999999999995})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.9695, '20_MG_L' : 0.03049999999999997})
	else:
		if(CNON_12_00 == '2_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0364, '20_MG_L' : 0.9636})
		elif(CNON_12_00 == '4_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0373, '20_MG_L' : 0.9627})
		elif(CNON_12_00 == '6_MG_L'):
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0377, '20_MG_L' : 0.9623})
		else:
			CBODN_12_15 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.038, '20_MG_L' : 0.962})


if (CBODD_12_15 == '15_MG_L'):
	if (CBODN_12_15 == '5_MG_L'):
		if (CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.9557, '10_MG_L' : 0.0443, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.9561, '10_MG_L' : 0.0439, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.9562, '10_MG_L' : 0.0438, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.9564, '10_MG_L' : 0.0436, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_15 == '10_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0406, '10_MG_L' : 0.9594, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0412, '10_MG_L' : 0.9588, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0414, '10_MG_L' : 0.9586, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0416, '10_MG_L' : 0.9584, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_15 == '15_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.1152, '15_MG_L' : 0.8848, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.116, '15_MG_L' : 0.884, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.1163, '15_MG_L' : 0.8837, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.1166, '15_MG_L' : 0.8834, '20_MG_L' : 0.0})
	else:
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1835, '20_MG_L' : 0.8165})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1844, '20_MG_L' : 0.8156})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1847, '20_MG_L' : 0.8153})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.185, '20_MG_L' : 0.815})
elif(CBODD_12_15 == '20_MG_L'):
	if(CBODN_12_15 == '5_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.9067, '10_MG_L' : 0.0933, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.9071, '10_MG_L' : 0.0929, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.9072, '10_MG_L' : 0.0928, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.9073, '10_MG_L' : 0.0927, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_15 == '10_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9916, '15_MG_L' : 0.0084, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9922, '15_MG_L' : 0.0078, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9924, '15_MG_L' : 0.0076, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9926, '15_MG_L' : 0.0074, '20_MG_L' : 0.0})
	elif(CBODN_12_15 == '15_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0662, '15_MG_L' : 0.9338, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.067, '15_MG_L' : 0.933, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0673, '15_MG_L' : 0.9327, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0676, '15_MG_L' : 0.9324, '20_MG_L' : 0.0})
	else:
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1344, '20_MG_L' : 0.8656})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1354, '20_MG_L' : 0.8646})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1357, '20_MG_L' : 0.8643000000000001})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.136, '20_MG_L' : 0.864})
elif(CBODD_12_15 == '25_MG_L'):
	if(CBODN_12_15 == '5_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.8577, '10_MG_L' : 0.1423, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.8581, '10_MG_L' : 0.1419, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.8582, '10_MG_L' : 0.1418, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.8583, '10_MG_L' : 0.1417, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_15 == '10_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9426, '15_MG_L' : 0.0574, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9432, '15_MG_L' : 0.0568, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9434, '15_MG_L' : 0.0566, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9436, '15_MG_L' : 0.0564, '20_MG_L' : 0.0})
	elif(CBODN_12_15 == '15_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0172, '15_MG_L' : 0.9828, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.018, '15_MG_L' : 0.982, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0183, '15_MG_L' : 0.9817, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0185, '15_MG_L' : 0.9815, '20_MG_L' : 0.0})
	else:
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0854, '20_MG_L' : 0.9146})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0863, '20_MG_L' : 0.9137})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0867, '20_MG_L' : 0.9133})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.087, '20_MG_L' : 0.913})
else:
	if(CBODN_12_15 == '5_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.8087, '10_MG_L' : 0.1913, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.809, '10_MG_L' : 0.191, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.8092, '10_MG_L' : 0.1908, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.8093, '10_MG_L' : 0.1907, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_15 == '10_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8936, '15_MG_L' : 0.1064, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8942, '15_MG_L' : 0.1058, '20_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8944, '15_MG_L' : 0.1056, '20_MG_L' : 0.0})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8946, '15_MG_L' : 0.1054, '20_MG_L' : 0.0})
	elif(CBODN_12_15 == '15_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.9682, '20_MG_L' : 0.03180000000000005})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.969, '20_MG_L' : 0.031000000000000028})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.9693, '20_MG_L' : 0.03069999999999995})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.9695, '20_MG_L' : 0.03049999999999997})
	else:
		if(CNON_12_15 == '2_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0364, '20_MG_L' : 0.9636})
		elif(CNON_12_15 == '4_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0373, '20_MG_L' : 0.9627})
		elif(CNON_12_15 == '6_MG_L'):
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0377, '20_MG_L' : 0.9623})
		else:
			CBODN_12_30 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.038, '20_MG_L' : 0.962})


if (CKNI_12_00 == '20_MG_L'):
	if (CKND_12_00 == '2_MG_L'):
		if (CKNN_12_00 == '0_5_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.9524, '4_MG_L' : 0.0476, '6_MG_L' : 0.0})
		elif(CKNN_12_00 == '1_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.9444, '4_MG_L' : 0.0556, '6_MG_L' : 0.0})
		else:
			CKND_12_15 ~= choice({'2_MG_L' : 0.9286, '4_MG_L' : 0.0714, '6_MG_L' : 0.0})
	elif(CKND_12_00 == '4_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9921, '6_MG_L' : 0.007900000000000018})
		elif(CKNN_12_00 == '1_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9841, '6_MG_L' : 0.015900000000000025})
		else:
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9683, '6_MG_L' : 0.03169999999999995})
	else:
		if(CKNN_12_00 == '0_5_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0317, '6_MG_L' : 0.9683})
		elif(CKNN_12_00 == '1_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0238, '6_MG_L' : 0.9762})
		else:
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0079, '6_MG_L' : 0.9921})
elif(CKNI_12_00 == '30_MG_L'):
	if(CKND_12_00 == '2_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.9127, '4_MG_L' : 0.0873, '6_MG_L' : 0.0})
		elif(CKNN_12_00 == '1_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.9048, '4_MG_L' : 0.0952, '6_MG_L' : 0.0})
		else:
			CKND_12_15 ~= choice({'2_MG_L' : 0.8889, '4_MG_L' : 0.1111, '6_MG_L' : 0.0})
	elif(CKND_12_00 == '4_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9524, '6_MG_L' : 0.047599999999999976})
		elif(CKNN_12_00 == '1_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9444, '6_MG_L' : 0.05559999999999998})
		else:
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9286, '6_MG_L' : 0.07140000000000002})
	else:
		if(CKNN_12_00 == '0_5_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		elif(CKNN_12_00 == '1_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		else:
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
else:
	if(CKND_12_00 == '2_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.873, '4_MG_L' : 0.127, '6_MG_L' : 0.0})
		elif(CKNN_12_00 == '1_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.8651, '4_MG_L' : 0.1349, '6_MG_L' : 0.0})
		else:
			CKND_12_15 ~= choice({'2_MG_L' : 0.8492, '4_MG_L' : 0.1508, '6_MG_L' : 0.0})
	elif(CKND_12_00 == '4_MG_L'):
		if(CKNN_12_00 == '0_5_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9127, '6_MG_L' : 0.08730000000000004})
		elif(CKNN_12_00 == '1_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9048, '6_MG_L' : 0.09519999999999995})
		else:
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8889, '6_MG_L' : 0.11109999999999998})
	else:
		if(CKNN_12_00 == '0_5_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		elif(CKNN_12_00 == '1_MG_L'):
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		else:
			CKND_12_15 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})


if (CKNI_12_15 == '20_MG_L'):
	if (CKND_12_15 == '2_MG_L'):
		if (CKNN_12_15 == '0_5_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.9524, '4_MG_L' : 0.0476, '6_MG_L' : 0.0})
		elif(CKNN_12_15 == '1_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.9444, '4_MG_L' : 0.0556, '6_MG_L' : 0.0})
		else:
			CKND_12_30 ~= choice({'2_MG_L' : 0.9286, '4_MG_L' : 0.0714, '6_MG_L' : 0.0})
	elif(CKND_12_15 == '4_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9921, '6_MG_L' : 0.007900000000000018})
		elif(CKNN_12_15 == '1_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9841, '6_MG_L' : 0.015900000000000025})
		else:
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9683, '6_MG_L' : 0.03169999999999995})
	else:
		if(CKNN_12_15 == '0_5_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0317, '6_MG_L' : 0.9683})
		elif(CKNN_12_15 == '1_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0238, '6_MG_L' : 0.9762})
		else:
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0079, '6_MG_L' : 0.9921})
elif(CKNI_12_15 == '30_MG_L'):
	if(CKND_12_15 == '2_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.9127, '4_MG_L' : 0.0873, '6_MG_L' : 0.0})
		elif(CKNN_12_15 == '1_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.9048, '4_MG_L' : 0.0952, '6_MG_L' : 0.0})
		else:
			CKND_12_30 ~= choice({'2_MG_L' : 0.8889, '4_MG_L' : 0.1111, '6_MG_L' : 0.0})
	elif(CKND_12_15 == '4_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9524, '6_MG_L' : 0.047599999999999976})
		elif(CKNN_12_15 == '1_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9444, '6_MG_L' : 0.05559999999999998})
		else:
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9286, '6_MG_L' : 0.07140000000000002})
	else:
		if(CKNN_12_15 == '0_5_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		elif(CKNN_12_15 == '1_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		else:
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
else:
	if(CKND_12_15 == '2_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.873, '4_MG_L' : 0.127, '6_MG_L' : 0.0})
		elif(CKNN_12_15 == '1_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.8651, '4_MG_L' : 0.1349, '6_MG_L' : 0.0})
		else:
			CKND_12_30 ~= choice({'2_MG_L' : 0.8492, '4_MG_L' : 0.1508, '6_MG_L' : 0.0})
	elif(CKND_12_15 == '4_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9127, '6_MG_L' : 0.08730000000000004})
		elif(CKNN_12_15 == '1_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9048, '6_MG_L' : 0.09519999999999995})
		else:
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8889, '6_MG_L' : 0.11109999999999998})
	else:
		if(CKNN_12_15 == '0_5_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		elif(CKNN_12_15 == '1_MG_L'):
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		else:
			CKND_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})


if (CKND_12_15 == '2_MG_L'):
	if (CKNN_12_15 == '0_5_MG_L'):
		CKNN_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0})
	elif(CKNN_12_15 == '1_MG_L'):
		CKNN_12_30 ~= choice({'0_5_MG_L' : 0.4459, '1_MG_L' : 0.5541, '2_MG_L' : 0.0})
	else:
		CKNN_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3627, '2_MG_L' : 0.6373})
elif(CKND_12_15 == '4_MG_L'):
	if(CKNN_12_15 == '0_5_MG_L'):
		CKNN_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0})
	elif(CKNN_12_15 == '1_MG_L'):
		CKNN_12_30 ~= choice({'0_5_MG_L' : 0.2499, '1_MG_L' : 0.7501, '2_MG_L' : 0.0})
	else:
		CKNN_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2646, '2_MG_L' : 0.7354})
else:
	if(CKNN_12_15 == '0_5_MG_L'):
		CKNN_12_30 ~= choice({'0_5_MG_L' : 0.8234, '1_MG_L' : 0.1766, '2_MG_L' : 0.0})
	elif(CKNN_12_15 == '1_MG_L'):
		CKNN_12_30 ~= choice({'0_5_MG_L' : 0.0538, '1_MG_L' : 0.9462, '2_MG_L' : 0.0})
	else:
		CKNN_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1666, '2_MG_L' : 0.8334})


if (CKND_12_30 == '2_MG_L'):
	if (CKNN_12_30 == '0_5_MG_L'):
		CKNN_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0})
	elif(CKNN_12_30 == '1_MG_L'):
		CKNN_12_45 ~= choice({'0_5_MG_L' : 0.4459, '1_MG_L' : 0.5541, '2_MG_L' : 0.0})
	else:
		CKNN_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3627, '2_MG_L' : 0.6373})
elif(CKND_12_30 == '4_MG_L'):
	if(CKNN_12_30 == '0_5_MG_L'):
		CKNN_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0})
	elif(CKNN_12_30 == '1_MG_L'):
		CKNN_12_45 ~= choice({'0_5_MG_L' : 0.2499, '1_MG_L' : 0.7501, '2_MG_L' : 0.0})
	else:
		CKNN_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2646, '2_MG_L' : 0.7354})
else:
	if(CKNN_12_30 == '0_5_MG_L'):
		CKNN_12_45 ~= choice({'0_5_MG_L' : 0.8234, '1_MG_L' : 0.1766, '2_MG_L' : 0.0})
	elif(CKNN_12_30 == '1_MG_L'):
		CKNN_12_45 ~= choice({'0_5_MG_L' : 0.0538, '1_MG_L' : 0.9462, '2_MG_L' : 0.0})
	else:
		CKNN_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1666, '2_MG_L' : 0.8334})


if (CBODD_12_00 == '15_MG_L'):
	if (CNOD_12_00 == '0_5_MG_L'):
		if (CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.8893, '1_MG_L' : 0.1107, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.6354, '1_MG_L' : 0.3646, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_00 == '1_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.3675, '1_MG_L' : 0.6325, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.2405, '1_MG_L' : 0.7595, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.1135, '1_MG_L' : 0.8865, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.9298, '2_MG_L' : 0.0702, '4_MG_L' : 0.0})
	elif(CNOD_12_00 == '2_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2972, '2_MG_L' : 0.7028, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2338, '2_MG_L' : 0.7662, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1703, '2_MG_L' : 0.8297, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0433, '2_MG_L' : 0.9567, '4_MG_L' : 0.0})
	else:
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2129, '4_MG_L' : 0.7871})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1812, '4_MG_L' : 0.8188})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1494, '4_MG_L' : 0.8506})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0859, '4_MG_L' : 0.9141})
elif(CBODD_12_00 == '20_MG_L'):
	if(CNOD_12_00 == '0_5_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.9816, '1_MG_L' : 0.0184, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.7276, '1_MG_L' : 0.2724, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_00 == '1_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.4905, '1_MG_L' : 0.5095, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.3635, '1_MG_L' : 0.6365, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.2366, '1_MG_L' : 0.7634, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.9913, '2_MG_L' : 0.0087, '4_MG_L' : 0.0})
	elif(CNOD_12_00 == '2_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3711, '2_MG_L' : 0.6289, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3076, '2_MG_L' : 0.6924, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2441, '2_MG_L' : 0.7559, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1171, '2_MG_L' : 0.8829, '4_MG_L' : 0.0})
	else:
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2539, '4_MG_L' : 0.7461})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2222, '4_MG_L' : 0.7778})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1904, '4_MG_L' : 0.8096})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1269, '4_MG_L' : 0.8731})
elif(CBODD_12_00 == '25_MG_L'):
	if(CNOD_12_00 == '0_5_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.7994, '1_MG_L' : 0.2006, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_00 == '1_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.5862, '1_MG_L' : 0.4138, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.4592, '1_MG_L' : 0.5408, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.3322, '1_MG_L' : 0.6678, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0783, '1_MG_L' : 0.9217, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_00 == '2_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.4285, '2_MG_L' : 0.5715, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.365, '2_MG_L' : 0.635, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3015, '2_MG_L' : 0.6985, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1745, '2_MG_L' : 0.8255, '4_MG_L' : 0.0})
	else:
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2858, '4_MG_L' : 0.7142})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2541, '4_MG_L' : 0.7459})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2223, '4_MG_L' : 0.7777000000000001})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1588, '4_MG_L' : 0.8412})
else:
	if(CNOD_12_00 == '0_5_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.8568, '1_MG_L' : 0.1432, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_00 == '1_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.6627, '1_MG_L' : 0.3373, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.5358, '1_MG_L' : 0.4642, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.4088, '1_MG_L' : 0.5912, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.1548, '1_MG_L' : 0.8452, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_00 == '2_MG_L'):
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.4744, '2_MG_L' : 0.5256, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.4109, '2_MG_L' : 0.5891, '4_MG_L' : 0.0})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3474, '2_MG_L' : 0.6526, '4_MG_L' : 0.0})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2204, '2_MG_L' : 0.7796, '4_MG_L' : 0.0})
	else:
		if(CNON_12_00 == '2_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.3113, '4_MG_L' : 0.6887})
		elif(CNON_12_00 == '4_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2796, '4_MG_L' : 0.7203999999999999})
		elif(CNON_12_00 == '6_MG_L'):
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2478, '4_MG_L' : 0.7522})
		else:
			CNOD_12_15 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1843, '4_MG_L' : 0.8157})


if (CBODD_12_15 == '15_MG_L'):
	if (CNOD_12_15 == '0_5_MG_L'):
		if (CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.8893, '1_MG_L' : 0.1107, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.6354, '1_MG_L' : 0.3646, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_15 == '1_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.3675, '1_MG_L' : 0.6325, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.2405, '1_MG_L' : 0.7595, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.1135, '1_MG_L' : 0.8865, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.9298, '2_MG_L' : 0.0702, '4_MG_L' : 0.0})
	elif(CNOD_12_15 == '2_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2972, '2_MG_L' : 0.7028, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2338, '2_MG_L' : 0.7662, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1703, '2_MG_L' : 0.8297, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0433, '2_MG_L' : 0.9567, '4_MG_L' : 0.0})
	else:
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2129, '4_MG_L' : 0.7871})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1812, '4_MG_L' : 0.8188})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1494, '4_MG_L' : 0.8506})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0859, '4_MG_L' : 0.9141})
elif(CBODD_12_15 == '20_MG_L'):
	if(CNOD_12_15 == '0_5_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.9816, '1_MG_L' : 0.0184, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.7276, '1_MG_L' : 0.2724, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_15 == '1_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.4905, '1_MG_L' : 0.5095, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.3635, '1_MG_L' : 0.6365, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.2366, '1_MG_L' : 0.7634, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.9913, '2_MG_L' : 0.0087, '4_MG_L' : 0.0})
	elif(CNOD_12_15 == '2_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3711, '2_MG_L' : 0.6289, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3076, '2_MG_L' : 0.6924, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2441, '2_MG_L' : 0.7559, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1171, '2_MG_L' : 0.8829, '4_MG_L' : 0.0})
	else:
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2539, '4_MG_L' : 0.7461})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2222, '4_MG_L' : 0.7778})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1904, '4_MG_L' : 0.8096})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1269, '4_MG_L' : 0.8731})
elif(CBODD_12_15 == '25_MG_L'):
	if(CNOD_12_15 == '0_5_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.7994, '1_MG_L' : 0.2006, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_15 == '1_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.5862, '1_MG_L' : 0.4138, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.4592, '1_MG_L' : 0.5408, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.3322, '1_MG_L' : 0.6678, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0783, '1_MG_L' : 0.9217, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_15 == '2_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.4285, '2_MG_L' : 0.5715, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.365, '2_MG_L' : 0.635, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3015, '2_MG_L' : 0.6985, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1745, '2_MG_L' : 0.8255, '4_MG_L' : 0.0})
	else:
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2858, '4_MG_L' : 0.7142})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2541, '4_MG_L' : 0.7459})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2223, '4_MG_L' : 0.7777000000000001})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1588, '4_MG_L' : 0.8412})
else:
	if(CNOD_12_15 == '0_5_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.8568, '1_MG_L' : 0.1432, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_15 == '1_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.6627, '1_MG_L' : 0.3373, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.5358, '1_MG_L' : 0.4642, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.4088, '1_MG_L' : 0.5912, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.1548, '1_MG_L' : 0.8452, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_15 == '2_MG_L'):
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.4744, '2_MG_L' : 0.5256, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.4109, '2_MG_L' : 0.5891, '4_MG_L' : 0.0})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3474, '2_MG_L' : 0.6526, '4_MG_L' : 0.0})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2204, '2_MG_L' : 0.7796, '4_MG_L' : 0.0})
	else:
		if(CNON_12_15 == '2_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.3113, '4_MG_L' : 0.6887})
		elif(CNON_12_15 == '4_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2796, '4_MG_L' : 0.7203999999999999})
		elif(CNON_12_15 == '6_MG_L'):
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2478, '4_MG_L' : 0.7522})
		else:
			CNOD_12_30 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1843, '4_MG_L' : 0.8157})


if (CNOD_12_15 == '0_5_MG_L'):
	if (CBODN_12_15 == '5_MG_L'):
		if (CKNN_12_15 == '0_5_MG_L'):
			if (CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9555, '4_MG_L' : 0.0445, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0056, '4_MG_L' : 0.9944, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.055, '6_MG_L' : 0.945, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0767, '10_MG_L' : 0.9233})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9102, '4_MG_L' : 0.0898, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9602, '6_MG_L' : 0.0398, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0096, '6_MG_L' : 0.9904, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.054, '10_MG_L' : 0.946})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8648, '4_MG_L' : 0.1352, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9149, '6_MG_L' : 0.0851, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9822, '10_MG_L' : 0.017800000000000038})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0313, '10_MG_L' : 0.9687})
	elif(CBODN_12_15 == '10_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9618, '4_MG_L' : 0.0382, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0125, '4_MG_L' : 0.9875, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0622, '6_MG_L' : 0.9378, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0804, '10_MG_L' : 0.9196})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9164, '4_MG_L' : 0.0836, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9672, '6_MG_L' : 0.0328, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0169, '6_MG_L' : 0.9831, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0577, '10_MG_L' : 0.9423})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8711, '4_MG_L' : 0.1289, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9219, '6_MG_L' : 0.0781, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9858, '10_MG_L' : 0.01419999999999999})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0351, '10_MG_L' : 0.9649})
	elif(CBODN_12_15 == '15_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9662, '4_MG_L' : 0.0338, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0175, '4_MG_L' : 0.9825, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0674, '6_MG_L' : 0.9326, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0831, '10_MG_L' : 0.9169})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9209, '4_MG_L' : 0.0791, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9722, '6_MG_L' : 0.0278, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.022, '6_MG_L' : 0.978, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0604, '10_MG_L' : 0.9396})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8756, '4_MG_L' : 0.1244, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9268, '6_MG_L' : 0.0732, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9884, '10_MG_L' : 0.011600000000000055})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0377, '10_MG_L' : 0.9623})
	else:
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9696, '4_MG_L' : 0.0304, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0212, '4_MG_L' : 0.9788, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0712, '6_MG_L' : 0.9288, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0851, '10_MG_L' : 0.9149})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9243, '4_MG_L' : 0.0757, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9759, '6_MG_L' : 0.0241, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0259, '6_MG_L' : 0.9741, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0624, '10_MG_L' : 0.9376})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8789, '4_MG_L' : 0.1211, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9306, '6_MG_L' : 0.0694, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9903, '10_MG_L' : 0.009700000000000042})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0397, '10_MG_L' : 0.9603})
elif(CNOD_12_15 == '1_MG_L'):
	if(CBODN_12_15 == '5_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9432, '4_MG_L' : 0.0568, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9933, '6_MG_L' : 0.0067, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0427, '6_MG_L' : 0.9573, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0706, '10_MG_L' : 0.9294})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8979, '4_MG_L' : 0.1021, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.948, '6_MG_L' : 0.052, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9987, '10_MG_L' : 0.0012999999999999678})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0479, '10_MG_L' : 0.9521})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8526, '4_MG_L' : 0.1474, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9026, '6_MG_L' : 0.0974, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.976, '10_MG_L' : 0.02400000000000002})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0252, '10_MG_L' : 0.9748})
	elif(CBODN_12_15 == '10_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9495, '4_MG_L' : 0.0505, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0003, '4_MG_L' : 0.9997, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.05, '6_MG_L' : 0.95, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0743, '10_MG_L' : 0.9257})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9042, '4_MG_L' : 0.0958, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9549, '6_MG_L' : 0.0451, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0046, '6_MG_L' : 0.9954, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0516, '10_MG_L' : 0.9484})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8588, '4_MG_L' : 0.1412, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9096, '6_MG_L' : 0.0904, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9796, '10_MG_L' : 0.020399999999999974})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0289, '10_MG_L' : 0.9711})
	elif(CBODN_12_15 == '15_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.954, '4_MG_L' : 0.046, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0052, '4_MG_L' : 0.9948, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0551, '6_MG_L' : 0.9449, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0769, '10_MG_L' : 0.9231})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9086, '4_MG_L' : 0.0914, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9599, '6_MG_L' : 0.0401, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0098, '6_MG_L' : 0.9902, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0543, '10_MG_L' : 0.9457})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8633, '4_MG_L' : 0.1367, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9146, '6_MG_L' : 0.0854, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9822, '10_MG_L' : 0.017800000000000038})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0316, '10_MG_L' : 0.9684})
	else:
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9573, '4_MG_L' : 0.0427, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.009, '4_MG_L' : 0.991, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.059, '6_MG_L' : 0.941, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0789, '10_MG_L' : 0.9211})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.912, '4_MG_L' : 0.088, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9636, '6_MG_L' : 0.0364, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0137, '6_MG_L' : 0.9863, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0563, '10_MG_L' : 0.9437})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8667, '4_MG_L' : 0.1333, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9183, '6_MG_L' : 0.0817, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9842, '10_MG_L' : 0.015800000000000036})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0336, '10_MG_L' : 0.9664})
elif(CNOD_12_15 == '2_MG_L'):
	if(CBODN_12_15 == '5_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9187, '4_MG_L' : 0.0813, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9688, '6_MG_L' : 0.0312, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0182, '6_MG_L' : 0.9818, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0583, '10_MG_L' : 0.9417})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8734, '4_MG_L' : 0.1266, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9235, '6_MG_L' : 0.0765, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9864, '10_MG_L' : 0.013599999999999945})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0356, '10_MG_L' : 0.9644})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8281, '4_MG_L' : 0.1719, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8781, '6_MG_L' : 0.1219, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9638, '10_MG_L' : 0.03620000000000001})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.013, '10_MG_L' : 0.987})
	elif(CBODN_12_15 == '10_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.925, '4_MG_L' : 0.075, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9758, '6_MG_L' : 0.0242, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0254, '6_MG_L' : 0.9746, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.062, '10_MG_L' : 0.938})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8797, '4_MG_L' : 0.1203, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9304, '6_MG_L' : 0.0696, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9901, '10_MG_L' : 0.00990000000000002})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0394, '10_MG_L' : 0.9606})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8343, '4_MG_L' : 0.1657, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8851, '6_MG_L' : 0.1149, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9674, '10_MG_L' : 0.03259999999999996})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0167, '10_MG_L' : 0.9833})
	elif(CBODN_12_15 == '15_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9295, '4_MG_L' : 0.0705, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9807, '6_MG_L' : 0.0193, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0306, '6_MG_L' : 0.9694, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0647, '10_MG_L' : 0.9353})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8841, '4_MG_L' : 0.1159, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9354, '6_MG_L' : 0.0646, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9926, '10_MG_L' : 0.007399999999999962})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.042, '10_MG_L' : 0.958})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8388, '4_MG_L' : 0.1612, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8901, '6_MG_L' : 0.1099, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.97, '10_MG_L' : 0.030000000000000027})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0194, '10_MG_L' : 0.9806})
	else:
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.9328, '4_MG_L' : 0.0672, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9845, '6_MG_L' : 0.0155, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0345, '6_MG_L' : 0.9655, '10_MG_L' : 0.0})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0667, '10_MG_L' : 0.9333})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8875, '4_MG_L' : 0.1125, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9391, '6_MG_L' : 0.0609, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9946, '10_MG_L' : 0.00539999999999996})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.044, '10_MG_L' : 0.956})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8422, '4_MG_L' : 0.1578, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8938, '6_MG_L' : 0.1062, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9719, '10_MG_L' : 0.028100000000000014})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0214, '10_MG_L' : 0.9786})
else:
	if(CBODN_12_15 == '5_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8697, '4_MG_L' : 0.1303, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9198, '6_MG_L' : 0.0802, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9846, '10_MG_L' : 0.01539999999999997})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0338, '10_MG_L' : 0.9662})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8244, '4_MG_L' : 0.1756, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8744, '6_MG_L' : 0.1256, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9619, '10_MG_L' : 0.03810000000000002})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0111, '10_MG_L' : 0.9889})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.779, '4_MG_L' : 0.221, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8291, '6_MG_L' : 0.1709, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9393, '10_MG_L' : 0.060699999999999976})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})
	elif(CBODN_12_15 == '10_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.876, '4_MG_L' : 0.124, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9267, '6_MG_L' : 0.0733, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9882, '10_MG_L' : 0.011800000000000033})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0375, '10_MG_L' : 0.9625})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8306, '4_MG_L' : 0.1694, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8814, '6_MG_L' : 0.1186, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9655, '10_MG_L' : 0.034499999999999975})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0149, '10_MG_L' : 0.9851})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.7853, '4_MG_L' : 0.2147, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8361, '6_MG_L' : 0.1639, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9429, '10_MG_L' : 0.05710000000000004})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})
	elif(CBODN_12_15 == '15_MG_L'):
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8804, '4_MG_L' : 0.1196, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9317, '6_MG_L' : 0.0683, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9908, '10_MG_L' : 0.009199999999999986})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0402, '10_MG_L' : 0.9598})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8351, '4_MG_L' : 0.1649, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8864, '6_MG_L' : 0.1136, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9681, '10_MG_L' : 0.03190000000000004})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0175, '10_MG_L' : 0.9825})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.7898, '4_MG_L' : 0.2102, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.841, '6_MG_L' : 0.159, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9455, '10_MG_L' : 0.05449999999999999})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})
	else:
		if(CKNN_12_15 == '0_5_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8838, '4_MG_L' : 0.1162, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9354, '6_MG_L' : 0.0646, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9927, '10_MG_L' : 0.007299999999999973})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0422, '10_MG_L' : 0.9578})
		elif(CKNN_12_15 == '1_MG_L'):
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.8385, '4_MG_L' : 0.1615, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8901, '6_MG_L' : 0.1099, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9701, '10_MG_L' : 0.029900000000000038})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0195, '10_MG_L' : 0.9805})
		else:
			if(CNON_12_15 == '2_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.7931, '4_MG_L' : 0.2069, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '4_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8448, '6_MG_L' : 0.1552, '10_MG_L' : 0.0})
			elif(CNON_12_15 == '6_MG_L'):
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9474, '10_MG_L' : 0.05259999999999998})
			else:
				CNON_12_30 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})


if (CNOD_12_30 == '0_5_MG_L'):
	if (CBODN_12_30 == '5_MG_L'):
		if (CKNN_12_30 == '0_5_MG_L'):
			if (CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9555, '4_MG_L' : 0.0445, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0056, '4_MG_L' : 0.9944, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.055, '6_MG_L' : 0.945, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0767, '10_MG_L' : 0.9233})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9102, '4_MG_L' : 0.0898, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9602, '6_MG_L' : 0.0398, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0096, '6_MG_L' : 0.9904, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.054, '10_MG_L' : 0.946})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8648, '4_MG_L' : 0.1352, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9149, '6_MG_L' : 0.0851, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9822, '10_MG_L' : 0.017800000000000038})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0313, '10_MG_L' : 0.9687})
	elif(CBODN_12_30 == '10_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9618, '4_MG_L' : 0.0382, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0125, '4_MG_L' : 0.9875, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0622, '6_MG_L' : 0.9378, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0804, '10_MG_L' : 0.9196})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9164, '4_MG_L' : 0.0836, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9672, '6_MG_L' : 0.0328, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0169, '6_MG_L' : 0.9831, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0577, '10_MG_L' : 0.9423})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8711, '4_MG_L' : 0.1289, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9219, '6_MG_L' : 0.0781, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9858, '10_MG_L' : 0.01419999999999999})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0351, '10_MG_L' : 0.9649})
	elif(CBODN_12_30 == '15_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9662, '4_MG_L' : 0.0338, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0175, '4_MG_L' : 0.9825, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0674, '6_MG_L' : 0.9326, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0831, '10_MG_L' : 0.9169})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9209, '4_MG_L' : 0.0791, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9722, '6_MG_L' : 0.0278, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.022, '6_MG_L' : 0.978, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0604, '10_MG_L' : 0.9396})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8756, '4_MG_L' : 0.1244, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9268, '6_MG_L' : 0.0732, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9884, '10_MG_L' : 0.011600000000000055})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0377, '10_MG_L' : 0.9623})
	else:
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9696, '4_MG_L' : 0.0304, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0212, '4_MG_L' : 0.9788, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0712, '6_MG_L' : 0.9288, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0851, '10_MG_L' : 0.9149})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9243, '4_MG_L' : 0.0757, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9759, '6_MG_L' : 0.0241, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0259, '6_MG_L' : 0.9741, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0624, '10_MG_L' : 0.9376})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8789, '4_MG_L' : 0.1211, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9306, '6_MG_L' : 0.0694, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9903, '10_MG_L' : 0.009700000000000042})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0397, '10_MG_L' : 0.9603})
elif(CNOD_12_30 == '1_MG_L'):
	if(CBODN_12_30 == '5_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9432, '4_MG_L' : 0.0568, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9933, '6_MG_L' : 0.0067, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0427, '6_MG_L' : 0.9573, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0706, '10_MG_L' : 0.9294})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8979, '4_MG_L' : 0.1021, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.948, '6_MG_L' : 0.052, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9987, '10_MG_L' : 0.0012999999999999678})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0479, '10_MG_L' : 0.9521})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8526, '4_MG_L' : 0.1474, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9026, '6_MG_L' : 0.0974, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.976, '10_MG_L' : 0.02400000000000002})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0252, '10_MG_L' : 0.9748})
	elif(CBODN_12_30 == '10_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9495, '4_MG_L' : 0.0505, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0003, '4_MG_L' : 0.9997, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.05, '6_MG_L' : 0.95, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0743, '10_MG_L' : 0.9257})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9042, '4_MG_L' : 0.0958, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9549, '6_MG_L' : 0.0451, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0046, '6_MG_L' : 0.9954, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0516, '10_MG_L' : 0.9484})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8588, '4_MG_L' : 0.1412, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9096, '6_MG_L' : 0.0904, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9796, '10_MG_L' : 0.020399999999999974})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0289, '10_MG_L' : 0.9711})
	elif(CBODN_12_30 == '15_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.954, '4_MG_L' : 0.046, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0052, '4_MG_L' : 0.9948, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0551, '6_MG_L' : 0.9449, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0769, '10_MG_L' : 0.9231})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9086, '4_MG_L' : 0.0914, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9599, '6_MG_L' : 0.0401, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0098, '6_MG_L' : 0.9902, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0543, '10_MG_L' : 0.9457})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8633, '4_MG_L' : 0.1367, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9146, '6_MG_L' : 0.0854, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9822, '10_MG_L' : 0.017800000000000038})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0316, '10_MG_L' : 0.9684})
	else:
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9573, '4_MG_L' : 0.0427, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.009, '4_MG_L' : 0.991, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.059, '6_MG_L' : 0.941, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0789, '10_MG_L' : 0.9211})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.912, '4_MG_L' : 0.088, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9636, '6_MG_L' : 0.0364, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0137, '6_MG_L' : 0.9863, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0563, '10_MG_L' : 0.9437})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8667, '4_MG_L' : 0.1333, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9183, '6_MG_L' : 0.0817, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9842, '10_MG_L' : 0.015800000000000036})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0336, '10_MG_L' : 0.9664})
elif(CNOD_12_30 == '2_MG_L'):
	if(CBODN_12_30 == '5_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9187, '4_MG_L' : 0.0813, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9688, '6_MG_L' : 0.0312, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0182, '6_MG_L' : 0.9818, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0583, '10_MG_L' : 0.9417})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8734, '4_MG_L' : 0.1266, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9235, '6_MG_L' : 0.0765, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9864, '10_MG_L' : 0.013599999999999945})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0356, '10_MG_L' : 0.9644})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8281, '4_MG_L' : 0.1719, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8781, '6_MG_L' : 0.1219, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9638, '10_MG_L' : 0.03620000000000001})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.013, '10_MG_L' : 0.987})
	elif(CBODN_12_30 == '10_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.925, '4_MG_L' : 0.075, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9758, '6_MG_L' : 0.0242, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0254, '6_MG_L' : 0.9746, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.062, '10_MG_L' : 0.938})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8797, '4_MG_L' : 0.1203, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9304, '6_MG_L' : 0.0696, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9901, '10_MG_L' : 0.00990000000000002})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0394, '10_MG_L' : 0.9606})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8343, '4_MG_L' : 0.1657, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8851, '6_MG_L' : 0.1149, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9674, '10_MG_L' : 0.03259999999999996})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0167, '10_MG_L' : 0.9833})
	elif(CBODN_12_30 == '15_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9295, '4_MG_L' : 0.0705, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9807, '6_MG_L' : 0.0193, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0306, '6_MG_L' : 0.9694, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0647, '10_MG_L' : 0.9353})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8841, '4_MG_L' : 0.1159, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9354, '6_MG_L' : 0.0646, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9926, '10_MG_L' : 0.007399999999999962})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.042, '10_MG_L' : 0.958})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8388, '4_MG_L' : 0.1612, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8901, '6_MG_L' : 0.1099, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.97, '10_MG_L' : 0.030000000000000027})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0194, '10_MG_L' : 0.9806})
	else:
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.9328, '4_MG_L' : 0.0672, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9845, '6_MG_L' : 0.0155, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0345, '6_MG_L' : 0.9655, '10_MG_L' : 0.0})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0667, '10_MG_L' : 0.9333})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8875, '4_MG_L' : 0.1125, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9391, '6_MG_L' : 0.0609, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9946, '10_MG_L' : 0.00539999999999996})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.044, '10_MG_L' : 0.956})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8422, '4_MG_L' : 0.1578, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8938, '6_MG_L' : 0.1062, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9719, '10_MG_L' : 0.028100000000000014})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0214, '10_MG_L' : 0.9786})
else:
	if(CBODN_12_30 == '5_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8697, '4_MG_L' : 0.1303, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9198, '6_MG_L' : 0.0802, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9846, '10_MG_L' : 0.01539999999999997})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0338, '10_MG_L' : 0.9662})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8244, '4_MG_L' : 0.1756, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8744, '6_MG_L' : 0.1256, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9619, '10_MG_L' : 0.03810000000000002})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0111, '10_MG_L' : 0.9889})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.779, '4_MG_L' : 0.221, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8291, '6_MG_L' : 0.1709, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9393, '10_MG_L' : 0.060699999999999976})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})
	elif(CBODN_12_30 == '10_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.876, '4_MG_L' : 0.124, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9267, '6_MG_L' : 0.0733, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9882, '10_MG_L' : 0.011800000000000033})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0375, '10_MG_L' : 0.9625})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8306, '4_MG_L' : 0.1694, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8814, '6_MG_L' : 0.1186, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9655, '10_MG_L' : 0.034499999999999975})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0149, '10_MG_L' : 0.9851})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.7853, '4_MG_L' : 0.2147, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8361, '6_MG_L' : 0.1639, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9429, '10_MG_L' : 0.05710000000000004})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})
	elif(CBODN_12_30 == '15_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8804, '4_MG_L' : 0.1196, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9317, '6_MG_L' : 0.0683, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9908, '10_MG_L' : 0.009199999999999986})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0402, '10_MG_L' : 0.9598})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8351, '4_MG_L' : 0.1649, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8864, '6_MG_L' : 0.1136, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9681, '10_MG_L' : 0.03190000000000004})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0175, '10_MG_L' : 0.9825})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.7898, '4_MG_L' : 0.2102, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.841, '6_MG_L' : 0.159, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9455, '10_MG_L' : 0.05449999999999999})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})
	else:
		if(CKNN_12_30 == '0_5_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8838, '4_MG_L' : 0.1162, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9354, '6_MG_L' : 0.0646, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9927, '10_MG_L' : 0.007299999999999973})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0422, '10_MG_L' : 0.9578})
		elif(CKNN_12_30 == '1_MG_L'):
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.8385, '4_MG_L' : 0.1615, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8901, '6_MG_L' : 0.1099, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9701, '10_MG_L' : 0.029900000000000038})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0195, '10_MG_L' : 0.9805})
		else:
			if(CNON_12_30 == '2_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.7931, '4_MG_L' : 0.2069, '6_MG_L' : 0.0, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '4_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8448, '6_MG_L' : 0.1552, '10_MG_L' : 0.0})
			elif(CNON_12_30 == '6_MG_L'):
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.9474, '10_MG_L' : 0.05259999999999998})
			else:
				CNON_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 0.0, '10_MG_L' : 1.0})


if (C_NI_12_15 == '3'):
	if (CKNI_12_15 == '20_MG_L'):
		if (CBODD_12_15 == '15_MG_L'):
			if (CNOD_12_15 == '0_5_MG_L'):
				if (CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9791, '20_MG_L' : 0.0209, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9951, '20_MG_L' : 0.0049, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0903, '20_MG_L' : 0.9097, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0585, '20_MG_L' : 0.9415, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0268, '20_MG_L' : 0.9732, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.109, '20_MG_L' : 0.891, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0773, '20_MG_L' : 0.9227, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0455, '20_MG_L' : 0.9545, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.124, '20_MG_L' : 0.876, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0923, '20_MG_L' : 0.9077, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0605, '20_MG_L' : 0.9395, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.134, '20_MG_L' : 0.866, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.1023, '20_MG_L' : 0.8977, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0705, '20_MG_L' : 0.9295, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1362, '25_MG_L' : 0.8638, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1045, '25_MG_L' : 0.8955, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0727, '25_MG_L' : 0.9273, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.157, '25_MG_L' : 0.843, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1253, '25_MG_L' : 0.8747, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0935, '25_MG_L' : 0.9065, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1737, '25_MG_L' : 0.8263, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.142, '25_MG_L' : 0.858, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1102, '25_MG_L' : 0.8898, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1848, '25_MG_L' : 0.8152, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1531, '25_MG_L' : 0.8469, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1213, '25_MG_L' : 0.8787, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1809, '30_MG_L' : 0.8190999999999999})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1491, '30_MG_L' : 0.8509})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1174, '30_MG_L' : 0.8826})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2034, '30_MG_L' : 0.7966})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1716, '30_MG_L' : 0.8284})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1399, '30_MG_L' : 0.8601})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2214, '30_MG_L' : 0.7786})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1896, '30_MG_L' : 0.8104})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1579, '30_MG_L' : 0.8421})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2334, '30_MG_L' : 0.7666})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2016, '30_MG_L' : 0.7984})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1699, '30_MG_L' : 0.8301000000000001})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
	elif(CKNI_12_15 == '30_MG_L'):
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9949, '20_MG_L' : 0.0051, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9632, '20_MG_L' : 0.0368, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9314, '20_MG_L' : 0.0686, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8997, '20_MG_L' : 0.1003, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9793, '20_MG_L' : 0.0207, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9475, '20_MG_L' : 0.0525, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9158, '20_MG_L' : 0.0842, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9921, '20_MG_L' : 0.0079, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9604, '20_MG_L' : 0.0396, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9286, '20_MG_L' : 0.0714, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9689, '20_MG_L' : 0.0311, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9372, '20_MG_L' : 0.0628, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0426, '20_MG_L' : 0.9574, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0109, '20_MG_L' : 0.9891, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9792, '25_MG_L' : 0.0208, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9474, '25_MG_L' : 0.0526, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0614, '20_MG_L' : 0.9386, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0296, '20_MG_L' : 0.9704, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9979, '25_MG_L' : 0.0021, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9662, '25_MG_L' : 0.0338, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0764, '20_MG_L' : 0.9236, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0446, '20_MG_L' : 0.9554, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0129, '20_MG_L' : 0.9871, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9812, '25_MG_L' : 0.0188, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0864, '20_MG_L' : 0.9136, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0546, '20_MG_L' : 0.9454, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0229, '20_MG_L' : 0.9771, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9912, '25_MG_L' : 0.0088, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0886, '25_MG_L' : 0.9114, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0568, '25_MG_L' : 0.9432, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0251, '25_MG_L' : 0.9749, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9933, '30_MG_L' : 0.006700000000000039})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1094, '25_MG_L' : 0.8906, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0777, '25_MG_L' : 0.9223, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0459, '25_MG_L' : 0.9541, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0142, '25_MG_L' : 0.9858, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1261, '25_MG_L' : 0.8739, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0943, '25_MG_L' : 0.9057, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0626, '25_MG_L' : 0.9374, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0308, '25_MG_L' : 0.9692, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1372, '25_MG_L' : 0.8628, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1054, '25_MG_L' : 0.8946, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0737, '25_MG_L' : 0.9263, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.042, '25_MG_L' : 0.958, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1333, '30_MG_L' : 0.8667})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1015, '30_MG_L' : 0.8985})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0698, '30_MG_L' : 0.9302})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.038, '30_MG_L' : 0.962})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1558, '30_MG_L' : 0.8442000000000001})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.124, '30_MG_L' : 0.876})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0923, '30_MG_L' : 0.9077})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0605, '30_MG_L' : 0.9395})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1738, '30_MG_L' : 0.8262})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.142, '30_MG_L' : 0.858})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1103, '30_MG_L' : 0.8897})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0785, '30_MG_L' : 0.9215})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1858, '30_MG_L' : 0.8142})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.154, '30_MG_L' : 0.846})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1223, '30_MG_L' : 0.8777})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0905, '30_MG_L' : 0.9095})
	else:
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
elif(C_NI_12_15 == '4'):
	if(CKNI_12_15 == '20_MG_L'):
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9791, '20_MG_L' : 0.0209, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9951, '20_MG_L' : 0.0049, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0585, '20_MG_L' : 0.9415, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0268, '20_MG_L' : 0.9732, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0773, '20_MG_L' : 0.9227, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0455, '20_MG_L' : 0.9545, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0923, '20_MG_L' : 0.9077, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0605, '20_MG_L' : 0.9395, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.1023, '20_MG_L' : 0.8977, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0705, '20_MG_L' : 0.9295, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1045, '25_MG_L' : 0.8955, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0727, '25_MG_L' : 0.9273, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1253, '25_MG_L' : 0.8747, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0935, '25_MG_L' : 0.9065, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.142, '25_MG_L' : 0.858, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1102, '25_MG_L' : 0.8898, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1531, '25_MG_L' : 0.8469, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1213, '25_MG_L' : 0.8787, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1491, '30_MG_L' : 0.8509})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1174, '30_MG_L' : 0.8826})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1716, '30_MG_L' : 0.8284})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1399, '30_MG_L' : 0.8601})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1896, '30_MG_L' : 0.8104})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1579, '30_MG_L' : 0.8421})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2016, '30_MG_L' : 0.7984})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1699, '30_MG_L' : 0.8301000000000001})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
	elif(CKNI_12_15 == '30_MG_L'):
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
	else:
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8203, '20_MG_L' : 0.1797, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7886, '20_MG_L' : 0.2114, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8364, '20_MG_L' : 0.1636, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8046, '20_MG_L' : 0.1954, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8493, '20_MG_L' : 0.1507, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8175, '20_MG_L' : 0.1825, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8578, '20_MG_L' : 0.1422, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8261, '20_MG_L' : 0.1739, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.868, '25_MG_L' : 0.132, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8363, '25_MG_L' : 0.1637, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8868, '25_MG_L' : 0.1132, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.855, '25_MG_L' : 0.145, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9018, '25_MG_L' : 0.0982, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.87, '25_MG_L' : 0.13, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9118, '25_MG_L' : 0.0882, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.88, '25_MG_L' : 0.12, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.914, '30_MG_L' : 0.08599999999999997})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8822, '30_MG_L' : 0.11780000000000002})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9348, '30_MG_L' : 0.06520000000000004})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9031, '30_MG_L' : 0.09689999999999999})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9515, '30_MG_L' : 0.04849999999999999})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9197, '30_MG_L' : 0.08030000000000004})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9626, '30_MG_L' : 0.03739999999999999})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9308, '30_MG_L' : 0.06920000000000004})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0112, '30_MG_L' : 0.9888})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
elif(C_NI_12_15 == '5'):
	if(CKNI_12_15 == '20_MG_L'):
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9791, '20_MG_L' : 0.0209, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9951, '20_MG_L' : 0.0049, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0268, '20_MG_L' : 0.9732, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0455, '20_MG_L' : 0.9545, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0605, '20_MG_L' : 0.9395, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0705, '20_MG_L' : 0.9295, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0727, '25_MG_L' : 0.9273, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0935, '25_MG_L' : 0.9065, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1102, '25_MG_L' : 0.8898, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1213, '25_MG_L' : 0.8787, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1174, '30_MG_L' : 0.8826})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1399, '30_MG_L' : 0.8601})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1579, '30_MG_L' : 0.8421})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1699, '30_MG_L' : 0.8301000000000001})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
	elif(CKNI_12_15 == '30_MG_L'):
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8997, '20_MG_L' : 0.1003, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8679, '20_MG_L' : 0.1321, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8362, '20_MG_L' : 0.1638, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8045, '20_MG_L' : 0.1955, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9158, '20_MG_L' : 0.0842, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.884, '20_MG_L' : 0.116, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8523, '20_MG_L' : 0.1477, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8205, '20_MG_L' : 0.1795, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9286, '20_MG_L' : 0.0714, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8969, '20_MG_L' : 0.1031, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8651, '20_MG_L' : 0.1349, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8334, '20_MG_L' : 0.1666, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9372, '20_MG_L' : 0.0628, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9054, '20_MG_L' : 0.0946, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8737, '20_MG_L' : 0.1263, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.842, '20_MG_L' : 0.158, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9474, '25_MG_L' : 0.0526, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9157, '25_MG_L' : 0.0843, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8839, '25_MG_L' : 0.1161, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8522, '25_MG_L' : 0.1478, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9662, '25_MG_L' : 0.0338, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9344, '25_MG_L' : 0.0656, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9027, '25_MG_L' : 0.0973, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8709, '25_MG_L' : 0.1291, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9812, '25_MG_L' : 0.0188, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9494, '25_MG_L' : 0.0506, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9177, '25_MG_L' : 0.0823, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8859, '25_MG_L' : 0.1141, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9912, '25_MG_L' : 0.0088, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9594, '25_MG_L' : 0.0406, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9277, '25_MG_L' : 0.0723, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8959, '25_MG_L' : 0.1041, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9933, '30_MG_L' : 0.006700000000000039})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9616, '30_MG_L' : 0.03839999999999999})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9298, '30_MG_L' : 0.07020000000000004})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8981, '30_MG_L' : 0.10189999999999999})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0142, '25_MG_L' : 0.9858, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9824, '30_MG_L' : 0.01759999999999995})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9507, '30_MG_L' : 0.04930000000000001})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9189, '30_MG_L' : 0.08109999999999995})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0308, '25_MG_L' : 0.9692, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9991, '30_MG_L' : 0.0009000000000000119})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9673, '30_MG_L' : 0.03269999999999995})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9356, '30_MG_L' : 0.06440000000000001})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.042, '25_MG_L' : 0.958, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0102, '25_MG_L' : 0.9898, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9785, '30_MG_L' : 0.021499999999999964})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9467, '30_MG_L' : 0.053300000000000014})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.038, '30_MG_L' : 0.962})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0063, '30_MG_L' : 0.9937})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0605, '30_MG_L' : 0.9395})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0288, '30_MG_L' : 0.9712})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0785, '30_MG_L' : 0.9215})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0468, '30_MG_L' : 0.9532})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.015, '30_MG_L' : 0.985})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0905, '30_MG_L' : 0.9095})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0588, '30_MG_L' : 0.9412})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.027, '30_MG_L' : 0.973})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
	else:
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8203, '20_MG_L' : 0.1797, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7886, '20_MG_L' : 0.2114, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7568, '20_MG_L' : 0.2432, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7251, '20_MG_L' : 0.2749, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8364, '20_MG_L' : 0.1636, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8046, '20_MG_L' : 0.1954, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7729, '20_MG_L' : 0.2271, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7412, '20_MG_L' : 0.2588, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8493, '20_MG_L' : 0.1507, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8175, '20_MG_L' : 0.1825, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7858, '20_MG_L' : 0.2142, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.754, '20_MG_L' : 0.246, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8578, '20_MG_L' : 0.1422, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8261, '20_MG_L' : 0.1739, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7943, '20_MG_L' : 0.2057, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7626, '20_MG_L' : 0.2374, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.868, '25_MG_L' : 0.132, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8363, '25_MG_L' : 0.1637, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8045, '25_MG_L' : 0.1955, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7728, '25_MG_L' : 0.2272, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8868, '25_MG_L' : 0.1132, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.855, '25_MG_L' : 0.145, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8233, '25_MG_L' : 0.1767, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7916, '25_MG_L' : 0.2084, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9018, '25_MG_L' : 0.0982, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.87, '25_MG_L' : 0.13, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8383, '25_MG_L' : 0.1617, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8066, '25_MG_L' : 0.1934, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9118, '25_MG_L' : 0.0882, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.88, '25_MG_L' : 0.12, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8483, '25_MG_L' : 0.1517, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8166, '25_MG_L' : 0.1834, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.914, '30_MG_L' : 0.08599999999999997})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8822, '30_MG_L' : 0.11780000000000002})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8505, '30_MG_L' : 0.14949999999999997})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8187, '30_MG_L' : 0.18130000000000002})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9348, '30_MG_L' : 0.06520000000000004})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9031, '30_MG_L' : 0.09689999999999999})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8713, '30_MG_L' : 0.12870000000000004})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8396, '30_MG_L' : 0.1604})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9515, '30_MG_L' : 0.04849999999999999})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9197, '30_MG_L' : 0.08030000000000004})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.888, '30_MG_L' : 0.11199999999999999})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8562, '30_MG_L' : 0.14380000000000004})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9626, '30_MG_L' : 0.03739999999999999})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9308, '30_MG_L' : 0.06920000000000004})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8991, '30_MG_L' : 0.10089999999999999})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8673, '30_MG_L' : 0.13270000000000004})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0112, '30_MG_L' : 0.9888})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
else:
	if(CKNI_12_15 == '20_MG_L'):
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
	elif(CKNI_12_15 == '30_MG_L'):
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8203, '20_MG_L' : 0.1797, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7886, '20_MG_L' : 0.2114, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7568, '20_MG_L' : 0.2432, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8364, '20_MG_L' : 0.1636, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8046, '20_MG_L' : 0.1954, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7729, '20_MG_L' : 0.2271, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8493, '20_MG_L' : 0.1507, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8175, '20_MG_L' : 0.1825, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7858, '20_MG_L' : 0.2142, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8578, '20_MG_L' : 0.1422, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.8261, '20_MG_L' : 0.1739, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7943, '20_MG_L' : 0.2057, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.868, '25_MG_L' : 0.132, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8363, '25_MG_L' : 0.1637, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8045, '25_MG_L' : 0.1955, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8868, '25_MG_L' : 0.1132, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.855, '25_MG_L' : 0.145, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8233, '25_MG_L' : 0.1767, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9018, '25_MG_L' : 0.0982, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.87, '25_MG_L' : 0.13, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8383, '25_MG_L' : 0.1617, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9118, '25_MG_L' : 0.0882, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.88, '25_MG_L' : 0.12, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8483, '25_MG_L' : 0.1517, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.914, '30_MG_L' : 0.08599999999999997})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8822, '30_MG_L' : 0.11780000000000002})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8505, '30_MG_L' : 0.14949999999999997})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9348, '30_MG_L' : 0.06520000000000004})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9031, '30_MG_L' : 0.09689999999999999})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8713, '30_MG_L' : 0.12870000000000004})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9515, '30_MG_L' : 0.04849999999999999})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9197, '30_MG_L' : 0.08030000000000004})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.888, '30_MG_L' : 0.11199999999999999})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9626, '30_MG_L' : 0.03739999999999999})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9308, '30_MG_L' : 0.06920000000000004})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8991, '30_MG_L' : 0.10089999999999999})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0112, '30_MG_L' : 0.9888})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
	else:
		if(CBODD_12_15 == '15_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7568, '20_MG_L' : 0.2432, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7251, '20_MG_L' : 0.2749, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.6933, '20_MG_L' : 0.3067, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.6616, '20_MG_L' : 0.3384, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7729, '20_MG_L' : 0.2271, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7412, '20_MG_L' : 0.2588, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7094, '20_MG_L' : 0.2906, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.6777, '20_MG_L' : 0.3223, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7858, '20_MG_L' : 0.2142, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.754, '20_MG_L' : 0.246, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7223, '20_MG_L' : 0.2777, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.6905, '20_MG_L' : 0.3095, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7943, '20_MG_L' : 0.2057, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7626, '20_MG_L' : 0.2374, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.7308, '20_MG_L' : 0.2692, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.6991, '20_MG_L' : 0.3009, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '20_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8045, '25_MG_L' : 0.1955, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7728, '25_MG_L' : 0.2272, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7411, '25_MG_L' : 0.2589, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7093, '25_MG_L' : 0.2907, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8233, '25_MG_L' : 0.1767, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7916, '25_MG_L' : 0.2084, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7598, '25_MG_L' : 0.2402, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7281, '25_MG_L' : 0.2719, '30_MG_L' : 0.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8383, '25_MG_L' : 0.1617, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8066, '25_MG_L' : 0.1934, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7748, '25_MG_L' : 0.2252, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7431, '25_MG_L' : 0.2569, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8483, '25_MG_L' : 0.1517, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8166, '25_MG_L' : 0.1834, '30_MG_L' : 0.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7848, '25_MG_L' : 0.2152, '30_MG_L' : 0.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7531, '25_MG_L' : 0.2469, '30_MG_L' : 0.0})
		elif(CBODD_12_15 == '25_MG_L'):
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8505, '30_MG_L' : 0.14949999999999997})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8187, '30_MG_L' : 0.18130000000000002})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.787, '30_MG_L' : 0.21299999999999997})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.7552, '30_MG_L' : 0.24480000000000002})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8713, '30_MG_L' : 0.12870000000000004})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8396, '30_MG_L' : 0.1604})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8078, '30_MG_L' : 0.19220000000000004})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.7761, '30_MG_L' : 0.2239})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.888, '30_MG_L' : 0.11199999999999999})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8562, '30_MG_L' : 0.14380000000000004})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8245, '30_MG_L' : 0.1755})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.7927, '30_MG_L' : 0.20730000000000004})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8991, '30_MG_L' : 0.10089999999999999})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8673, '30_MG_L' : 0.13270000000000004})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8356, '30_MG_L' : 0.1644})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8039, '30_MG_L' : 0.19610000000000005})
		else:
			if(CNOD_12_15 == '0_5_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '1_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_15 == '2_MG_L'):
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_15 == '5_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '10_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_15 == '15_MG_L'):
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_30 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})


if (C_NI_12_30 == '3'):
	if (CKNI_12_30 == '20_MG_L'):
		if (CBODD_12_30 == '15_MG_L'):
			if (CNOD_12_30 == '0_5_MG_L'):
				if (CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9791, '20_MG_L' : 0.0209, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9951, '20_MG_L' : 0.0049, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0903, '20_MG_L' : 0.9097, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0585, '20_MG_L' : 0.9415, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0268, '20_MG_L' : 0.9732, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.109, '20_MG_L' : 0.891, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0773, '20_MG_L' : 0.9227, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0455, '20_MG_L' : 0.9545, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.124, '20_MG_L' : 0.876, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0923, '20_MG_L' : 0.9077, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0605, '20_MG_L' : 0.9395, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.134, '20_MG_L' : 0.866, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.1023, '20_MG_L' : 0.8977, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0705, '20_MG_L' : 0.9295, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1362, '25_MG_L' : 0.8638, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1045, '25_MG_L' : 0.8955, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0727, '25_MG_L' : 0.9273, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.157, '25_MG_L' : 0.843, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1253, '25_MG_L' : 0.8747, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0935, '25_MG_L' : 0.9065, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1737, '25_MG_L' : 0.8263, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.142, '25_MG_L' : 0.858, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1102, '25_MG_L' : 0.8898, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1848, '25_MG_L' : 0.8152, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1531, '25_MG_L' : 0.8469, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1213, '25_MG_L' : 0.8787, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1809, '30_MG_L' : 0.8190999999999999})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1491, '30_MG_L' : 0.8509})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1174, '30_MG_L' : 0.8826})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2034, '30_MG_L' : 0.7966})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1716, '30_MG_L' : 0.8284})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1399, '30_MG_L' : 0.8601})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2214, '30_MG_L' : 0.7786})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1896, '30_MG_L' : 0.8104})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1579, '30_MG_L' : 0.8421})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2334, '30_MG_L' : 0.7666})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2016, '30_MG_L' : 0.7984})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1699, '30_MG_L' : 0.8301000000000001})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
	elif(CKNI_12_30 == '30_MG_L'):
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9949, '20_MG_L' : 0.0051, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9632, '20_MG_L' : 0.0368, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9314, '20_MG_L' : 0.0686, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8997, '20_MG_L' : 0.1003, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9793, '20_MG_L' : 0.0207, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9475, '20_MG_L' : 0.0525, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9158, '20_MG_L' : 0.0842, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9921, '20_MG_L' : 0.0079, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9604, '20_MG_L' : 0.0396, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9286, '20_MG_L' : 0.0714, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9689, '20_MG_L' : 0.0311, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9372, '20_MG_L' : 0.0628, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0426, '20_MG_L' : 0.9574, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0109, '20_MG_L' : 0.9891, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9792, '25_MG_L' : 0.0208, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9474, '25_MG_L' : 0.0526, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0614, '20_MG_L' : 0.9386, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0296, '20_MG_L' : 0.9704, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9979, '25_MG_L' : 0.0021, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9662, '25_MG_L' : 0.0338, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0764, '20_MG_L' : 0.9236, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0446, '20_MG_L' : 0.9554, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0129, '20_MG_L' : 0.9871, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9812, '25_MG_L' : 0.0188, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0864, '20_MG_L' : 0.9136, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0546, '20_MG_L' : 0.9454, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0229, '20_MG_L' : 0.9771, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9912, '25_MG_L' : 0.0088, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0886, '25_MG_L' : 0.9114, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0568, '25_MG_L' : 0.9432, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0251, '25_MG_L' : 0.9749, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9933, '30_MG_L' : 0.006700000000000039})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1094, '25_MG_L' : 0.8906, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0777, '25_MG_L' : 0.9223, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0459, '25_MG_L' : 0.9541, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0142, '25_MG_L' : 0.9858, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1261, '25_MG_L' : 0.8739, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0943, '25_MG_L' : 0.9057, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0626, '25_MG_L' : 0.9374, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0308, '25_MG_L' : 0.9692, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1372, '25_MG_L' : 0.8628, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1054, '25_MG_L' : 0.8946, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0737, '25_MG_L' : 0.9263, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.042, '25_MG_L' : 0.958, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1333, '30_MG_L' : 0.8667})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1015, '30_MG_L' : 0.8985})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0698, '30_MG_L' : 0.9302})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.038, '30_MG_L' : 0.962})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1558, '30_MG_L' : 0.8442000000000001})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.124, '30_MG_L' : 0.876})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0923, '30_MG_L' : 0.9077})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0605, '30_MG_L' : 0.9395})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1738, '30_MG_L' : 0.8262})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.142, '30_MG_L' : 0.858})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1103, '30_MG_L' : 0.8897})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0785, '30_MG_L' : 0.9215})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1858, '30_MG_L' : 0.8142})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.154, '30_MG_L' : 0.846})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1223, '30_MG_L' : 0.8777})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0905, '30_MG_L' : 0.9095})
	else:
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
elif(C_NI_12_30 == '4'):
	if(CKNI_12_30 == '20_MG_L'):
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9791, '20_MG_L' : 0.0209, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9951, '20_MG_L' : 0.0049, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0585, '20_MG_L' : 0.9415, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0268, '20_MG_L' : 0.9732, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0773, '20_MG_L' : 0.9227, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0455, '20_MG_L' : 0.9545, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0923, '20_MG_L' : 0.9077, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0605, '20_MG_L' : 0.9395, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.1023, '20_MG_L' : 0.8977, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0705, '20_MG_L' : 0.9295, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1045, '25_MG_L' : 0.8955, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0727, '25_MG_L' : 0.9273, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1253, '25_MG_L' : 0.8747, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0935, '25_MG_L' : 0.9065, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.142, '25_MG_L' : 0.858, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1102, '25_MG_L' : 0.8898, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1531, '25_MG_L' : 0.8469, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1213, '25_MG_L' : 0.8787, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1491, '30_MG_L' : 0.8509})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1174, '30_MG_L' : 0.8826})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1716, '30_MG_L' : 0.8284})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1399, '30_MG_L' : 0.8601})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1896, '30_MG_L' : 0.8104})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1579, '30_MG_L' : 0.8421})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.2016, '30_MG_L' : 0.7984})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1699, '30_MG_L' : 0.8301000000000001})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
	elif(CKNI_12_30 == '30_MG_L'):
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
	else:
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8203, '20_MG_L' : 0.1797, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7886, '20_MG_L' : 0.2114, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8364, '20_MG_L' : 0.1636, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8046, '20_MG_L' : 0.1954, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8493, '20_MG_L' : 0.1507, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8175, '20_MG_L' : 0.1825, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8578, '20_MG_L' : 0.1422, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8261, '20_MG_L' : 0.1739, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.868, '25_MG_L' : 0.132, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8363, '25_MG_L' : 0.1637, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8868, '25_MG_L' : 0.1132, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.855, '25_MG_L' : 0.145, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9018, '25_MG_L' : 0.0982, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.87, '25_MG_L' : 0.13, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9118, '25_MG_L' : 0.0882, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.88, '25_MG_L' : 0.12, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.914, '30_MG_L' : 0.08599999999999997})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8822, '30_MG_L' : 0.11780000000000002})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9348, '30_MG_L' : 0.06520000000000004})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9031, '30_MG_L' : 0.09689999999999999})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9515, '30_MG_L' : 0.04849999999999999})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9197, '30_MG_L' : 0.08030000000000004})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9626, '30_MG_L' : 0.03739999999999999})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9308, '30_MG_L' : 0.06920000000000004})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0112, '30_MG_L' : 0.9888})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
elif(C_NI_12_30 == '5'):
	if(CKNI_12_30 == '20_MG_L'):
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9791, '20_MG_L' : 0.0209, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9951, '20_MG_L' : 0.0049, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 1.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0268, '20_MG_L' : 0.9732, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0455, '20_MG_L' : 0.9545, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0605, '20_MG_L' : 0.9395, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0705, '20_MG_L' : 0.9295, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0727, '25_MG_L' : 0.9273, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0935, '25_MG_L' : 0.9065, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1102, '25_MG_L' : 0.8898, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.1213, '25_MG_L' : 0.8787, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1174, '30_MG_L' : 0.8826})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1399, '30_MG_L' : 0.8601})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1579, '30_MG_L' : 0.8421})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1699, '30_MG_L' : 0.8301000000000001})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
	elif(CKNI_12_30 == '30_MG_L'):
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8997, '20_MG_L' : 0.1003, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8679, '20_MG_L' : 0.1321, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8362, '20_MG_L' : 0.1638, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8045, '20_MG_L' : 0.1955, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9158, '20_MG_L' : 0.0842, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.884, '20_MG_L' : 0.116, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8523, '20_MG_L' : 0.1477, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8205, '20_MG_L' : 0.1795, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9286, '20_MG_L' : 0.0714, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8969, '20_MG_L' : 0.1031, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8651, '20_MG_L' : 0.1349, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8334, '20_MG_L' : 0.1666, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9372, '20_MG_L' : 0.0628, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9054, '20_MG_L' : 0.0946, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8737, '20_MG_L' : 0.1263, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.842, '20_MG_L' : 0.158, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9474, '25_MG_L' : 0.0526, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9157, '25_MG_L' : 0.0843, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8839, '25_MG_L' : 0.1161, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8522, '25_MG_L' : 0.1478, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9662, '25_MG_L' : 0.0338, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9344, '25_MG_L' : 0.0656, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9027, '25_MG_L' : 0.0973, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8709, '25_MG_L' : 0.1291, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9812, '25_MG_L' : 0.0188, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9494, '25_MG_L' : 0.0506, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9177, '25_MG_L' : 0.0823, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8859, '25_MG_L' : 0.1141, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9912, '25_MG_L' : 0.0088, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9594, '25_MG_L' : 0.0406, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9277, '25_MG_L' : 0.0723, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8959, '25_MG_L' : 0.1041, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9933, '30_MG_L' : 0.006700000000000039})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9616, '30_MG_L' : 0.03839999999999999})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9298, '30_MG_L' : 0.07020000000000004})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8981, '30_MG_L' : 0.10189999999999999})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0142, '25_MG_L' : 0.9858, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9824, '30_MG_L' : 0.01759999999999995})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9507, '30_MG_L' : 0.04930000000000001})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9189, '30_MG_L' : 0.08109999999999995})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0308, '25_MG_L' : 0.9692, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9991, '30_MG_L' : 0.0009000000000000119})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9673, '30_MG_L' : 0.03269999999999995})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9356, '30_MG_L' : 0.06440000000000001})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.042, '25_MG_L' : 0.958, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0102, '25_MG_L' : 0.9898, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9785, '30_MG_L' : 0.021499999999999964})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9467, '30_MG_L' : 0.053300000000000014})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.038, '30_MG_L' : 0.962})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0063, '30_MG_L' : 0.9937})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0605, '30_MG_L' : 0.9395})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0288, '30_MG_L' : 0.9712})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0785, '30_MG_L' : 0.9215})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0468, '30_MG_L' : 0.9532})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.015, '30_MG_L' : 0.985})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0905, '30_MG_L' : 0.9095})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0588, '30_MG_L' : 0.9412})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.027, '30_MG_L' : 0.973})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
	else:
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8203, '20_MG_L' : 0.1797, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7886, '20_MG_L' : 0.2114, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7568, '20_MG_L' : 0.2432, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7251, '20_MG_L' : 0.2749, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8364, '20_MG_L' : 0.1636, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8046, '20_MG_L' : 0.1954, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7729, '20_MG_L' : 0.2271, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7412, '20_MG_L' : 0.2588, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8493, '20_MG_L' : 0.1507, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8175, '20_MG_L' : 0.1825, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7858, '20_MG_L' : 0.2142, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.754, '20_MG_L' : 0.246, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8578, '20_MG_L' : 0.1422, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8261, '20_MG_L' : 0.1739, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7943, '20_MG_L' : 0.2057, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7626, '20_MG_L' : 0.2374, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.868, '25_MG_L' : 0.132, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8363, '25_MG_L' : 0.1637, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8045, '25_MG_L' : 0.1955, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7728, '25_MG_L' : 0.2272, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8868, '25_MG_L' : 0.1132, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.855, '25_MG_L' : 0.145, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8233, '25_MG_L' : 0.1767, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7916, '25_MG_L' : 0.2084, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9018, '25_MG_L' : 0.0982, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.87, '25_MG_L' : 0.13, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8383, '25_MG_L' : 0.1617, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8066, '25_MG_L' : 0.1934, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9118, '25_MG_L' : 0.0882, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.88, '25_MG_L' : 0.12, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8483, '25_MG_L' : 0.1517, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8166, '25_MG_L' : 0.1834, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.914, '30_MG_L' : 0.08599999999999997})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8822, '30_MG_L' : 0.11780000000000002})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8505, '30_MG_L' : 0.14949999999999997})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8187, '30_MG_L' : 0.18130000000000002})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9348, '30_MG_L' : 0.06520000000000004})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9031, '30_MG_L' : 0.09689999999999999})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8713, '30_MG_L' : 0.12870000000000004})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8396, '30_MG_L' : 0.1604})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9515, '30_MG_L' : 0.04849999999999999})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9197, '30_MG_L' : 0.08030000000000004})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.888, '30_MG_L' : 0.11199999999999999})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8562, '30_MG_L' : 0.14380000000000004})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9626, '30_MG_L' : 0.03739999999999999})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9308, '30_MG_L' : 0.06920000000000004})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8991, '30_MG_L' : 0.10089999999999999})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8673, '30_MG_L' : 0.13270000000000004})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0112, '30_MG_L' : 0.9888})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
else:
	if(CKNI_12_30 == '20_MG_L'):
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9473, '20_MG_L' : 0.0527, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9156, '20_MG_L' : 0.0844, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8838, '20_MG_L' : 0.1162, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9634, '20_MG_L' : 0.0366, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9316, '20_MG_L' : 0.0684, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8999, '20_MG_L' : 0.1001, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9762, '20_MG_L' : 0.0238, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9445, '20_MG_L' : 0.0555, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9127, '20_MG_L' : 0.0873, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9848, '20_MG_L' : 0.0152, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9531, '20_MG_L' : 0.0469, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.9213, '20_MG_L' : 0.0787, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.995, '25_MG_L' : 0.005, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9633, '25_MG_L' : 0.0367, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9315, '25_MG_L' : 0.0685, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0138, '20_MG_L' : 0.9862, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.982, '25_MG_L' : 0.018, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9503, '25_MG_L' : 0.0497, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0288, '20_MG_L' : 0.9712, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.997, '25_MG_L' : 0.003, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9653, '25_MG_L' : 0.0347, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0388, '20_MG_L' : 0.9612, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.007, '20_MG_L' : 0.993, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9753, '25_MG_L' : 0.0247, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.041, '25_MG_L' : 0.959, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0092, '25_MG_L' : 0.9908, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9775, '30_MG_L' : 0.022499999999999964})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0618, '25_MG_L' : 0.9382, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.03, '25_MG_L' : 0.97, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9983, '30_MG_L' : 0.0017000000000000348})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0785, '25_MG_L' : 0.9215, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0467, '25_MG_L' : 0.9533, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.015, '25_MG_L' : 0.985, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0896, '25_MG_L' : 0.9104, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0578, '25_MG_L' : 0.9422, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0261, '25_MG_L' : 0.9739, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0856, '30_MG_L' : 0.9144})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0539, '30_MG_L' : 0.9460999999999999})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0221, '30_MG_L' : 0.9779})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1081, '30_MG_L' : 0.8919})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0764, '30_MG_L' : 0.9236})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0446, '30_MG_L' : 0.9554})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1261, '30_MG_L' : 0.8739})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0944, '30_MG_L' : 0.9056})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0626, '30_MG_L' : 0.9374})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1381, '30_MG_L' : 0.8619})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.1064, '30_MG_L' : 0.8936})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0746, '30_MG_L' : 0.9254})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
	elif(CKNI_12_30 == '30_MG_L'):
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8521, '20_MG_L' : 0.1479, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8203, '20_MG_L' : 0.1797, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7886, '20_MG_L' : 0.2114, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7568, '20_MG_L' : 0.2432, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8681, '20_MG_L' : 0.1319, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8364, '20_MG_L' : 0.1636, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8046, '20_MG_L' : 0.1954, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7729, '20_MG_L' : 0.2271, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.881, '20_MG_L' : 0.119, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8493, '20_MG_L' : 0.1507, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8175, '20_MG_L' : 0.1825, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7858, '20_MG_L' : 0.2142, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8896, '20_MG_L' : 0.1104, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8578, '20_MG_L' : 0.1422, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.8261, '20_MG_L' : 0.1739, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7943, '20_MG_L' : 0.2057, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8998, '25_MG_L' : 0.1002, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.868, '25_MG_L' : 0.132, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8363, '25_MG_L' : 0.1637, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8045, '25_MG_L' : 0.1955, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9185, '25_MG_L' : 0.0815, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8868, '25_MG_L' : 0.1132, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.855, '25_MG_L' : 0.145, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8233, '25_MG_L' : 0.1767, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9335, '25_MG_L' : 0.0665, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9018, '25_MG_L' : 0.0982, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.87, '25_MG_L' : 0.13, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8383, '25_MG_L' : 0.1617, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9435, '25_MG_L' : 0.0565, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.9118, '25_MG_L' : 0.0882, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.88, '25_MG_L' : 0.12, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8483, '25_MG_L' : 0.1517, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9457, '30_MG_L' : 0.054300000000000015})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.914, '30_MG_L' : 0.08599999999999997})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8822, '30_MG_L' : 0.11780000000000002})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8505, '30_MG_L' : 0.14949999999999997})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9666, '30_MG_L' : 0.033399999999999985})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9348, '30_MG_L' : 0.06520000000000004})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9031, '30_MG_L' : 0.09689999999999999})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8713, '30_MG_L' : 0.12870000000000004})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9832, '30_MG_L' : 0.016800000000000037})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9515, '30_MG_L' : 0.04849999999999999})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9197, '30_MG_L' : 0.08030000000000004})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.888, '30_MG_L' : 0.11199999999999999})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9943, '30_MG_L' : 0.005700000000000038})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9626, '30_MG_L' : 0.03739999999999999})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.9308, '30_MG_L' : 0.06920000000000004})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8991, '30_MG_L' : 0.10089999999999999})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0129, '30_MG_L' : 0.9871})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0309, '30_MG_L' : 0.9691})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0429, '30_MG_L' : 0.9571})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0112, '30_MG_L' : 0.9888})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
	else:
		if(CBODD_12_30 == '15_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7568, '20_MG_L' : 0.2432, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7251, '20_MG_L' : 0.2749, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.6933, '20_MG_L' : 0.3067, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.6616, '20_MG_L' : 0.3384, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7729, '20_MG_L' : 0.2271, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7412, '20_MG_L' : 0.2588, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7094, '20_MG_L' : 0.2906, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.6777, '20_MG_L' : 0.3223, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7858, '20_MG_L' : 0.2142, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.754, '20_MG_L' : 0.246, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7223, '20_MG_L' : 0.2777, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.6905, '20_MG_L' : 0.3095, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7943, '20_MG_L' : 0.2057, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7626, '20_MG_L' : 0.2374, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.7308, '20_MG_L' : 0.2692, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.6991, '20_MG_L' : 0.3009, '25_MG_L' : 0.0, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '20_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8045, '25_MG_L' : 0.1955, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7728, '25_MG_L' : 0.2272, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7411, '25_MG_L' : 0.2589, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7093, '25_MG_L' : 0.2907, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8233, '25_MG_L' : 0.1767, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7916, '25_MG_L' : 0.2084, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7598, '25_MG_L' : 0.2402, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7281, '25_MG_L' : 0.2719, '30_MG_L' : 0.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8383, '25_MG_L' : 0.1617, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8066, '25_MG_L' : 0.1934, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7748, '25_MG_L' : 0.2252, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7431, '25_MG_L' : 0.2569, '30_MG_L' : 0.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8483, '25_MG_L' : 0.1517, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.8166, '25_MG_L' : 0.1834, '30_MG_L' : 0.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7848, '25_MG_L' : 0.2152, '30_MG_L' : 0.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.7531, '25_MG_L' : 0.2469, '30_MG_L' : 0.0})
		elif(CBODD_12_30 == '25_MG_L'):
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8505, '30_MG_L' : 0.14949999999999997})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8187, '30_MG_L' : 0.18130000000000002})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.787, '30_MG_L' : 0.21299999999999997})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.7552, '30_MG_L' : 0.24480000000000002})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8713, '30_MG_L' : 0.12870000000000004})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8396, '30_MG_L' : 0.1604})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8078, '30_MG_L' : 0.19220000000000004})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.7761, '30_MG_L' : 0.2239})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.888, '30_MG_L' : 0.11199999999999999})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8562, '30_MG_L' : 0.14380000000000004})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8245, '30_MG_L' : 0.1755})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.7927, '30_MG_L' : 0.20730000000000004})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8991, '30_MG_L' : 0.10089999999999999})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8673, '30_MG_L' : 0.13270000000000004})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8356, '30_MG_L' : 0.1644})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.8039, '30_MG_L' : 0.19610000000000005})
		else:
			if(CNOD_12_30 == '0_5_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '1_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			elif(CNOD_12_30 == '2_MG_L'):
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
			else:
				if(CBODN_12_30 == '5_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '10_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				elif(CBODN_12_30 == '15_MG_L'):
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})
				else:
					CBODD_12_45 ~= choice({'15_MG_L' : 0.0, '20_MG_L' : 0.0, '25_MG_L' : 0.0, '30_MG_L' : 1.0})


if (CBODD_12_30 == '15_MG_L'):
	if (CBODN_12_30 == '5_MG_L'):
		if (CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.9557, '10_MG_L' : 0.0443, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.9561, '10_MG_L' : 0.0439, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.9562, '10_MG_L' : 0.0438, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.9564, '10_MG_L' : 0.0436, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_30 == '10_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0406, '10_MG_L' : 0.9594, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0412, '10_MG_L' : 0.9588, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0414, '10_MG_L' : 0.9586, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0416, '10_MG_L' : 0.9584, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_30 == '15_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.1152, '15_MG_L' : 0.8848, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.116, '15_MG_L' : 0.884, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.1163, '15_MG_L' : 0.8837, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.1166, '15_MG_L' : 0.8834, '20_MG_L' : 0.0})
	else:
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1835, '20_MG_L' : 0.8165})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1844, '20_MG_L' : 0.8156})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1847, '20_MG_L' : 0.8153})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.185, '20_MG_L' : 0.815})
elif(CBODD_12_30 == '20_MG_L'):
	if(CBODN_12_30 == '5_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.9067, '10_MG_L' : 0.0933, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.9071, '10_MG_L' : 0.0929, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.9072, '10_MG_L' : 0.0928, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.9073, '10_MG_L' : 0.0927, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_30 == '10_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9916, '15_MG_L' : 0.0084, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9922, '15_MG_L' : 0.0078, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9924, '15_MG_L' : 0.0076, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9926, '15_MG_L' : 0.0074, '20_MG_L' : 0.0})
	elif(CBODN_12_30 == '15_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0662, '15_MG_L' : 0.9338, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.067, '15_MG_L' : 0.933, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0673, '15_MG_L' : 0.9327, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0676, '15_MG_L' : 0.9324, '20_MG_L' : 0.0})
	else:
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1344, '20_MG_L' : 0.8656})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1354, '20_MG_L' : 0.8646})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.1357, '20_MG_L' : 0.8643000000000001})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.136, '20_MG_L' : 0.864})
elif(CBODD_12_30 == '25_MG_L'):
	if(CBODN_12_30 == '5_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.8577, '10_MG_L' : 0.1423, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.8581, '10_MG_L' : 0.1419, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.8582, '10_MG_L' : 0.1418, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.8583, '10_MG_L' : 0.1417, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_30 == '10_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9426, '15_MG_L' : 0.0574, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9432, '15_MG_L' : 0.0568, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9434, '15_MG_L' : 0.0566, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.9436, '15_MG_L' : 0.0564, '20_MG_L' : 0.0})
	elif(CBODN_12_30 == '15_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0172, '15_MG_L' : 0.9828, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.018, '15_MG_L' : 0.982, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0183, '15_MG_L' : 0.9817, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0185, '15_MG_L' : 0.9815, '20_MG_L' : 0.0})
	else:
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0854, '20_MG_L' : 0.9146})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0863, '20_MG_L' : 0.9137})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0867, '20_MG_L' : 0.9133})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.087, '20_MG_L' : 0.913})
else:
	if(CBODN_12_30 == '5_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.8087, '10_MG_L' : 0.1913, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.809, '10_MG_L' : 0.191, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.8092, '10_MG_L' : 0.1908, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.8093, '10_MG_L' : 0.1907, '15_MG_L' : 0.0, '20_MG_L' : 0.0})
	elif(CBODN_12_30 == '10_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8936, '15_MG_L' : 0.1064, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8942, '15_MG_L' : 0.1058, '20_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8944, '15_MG_L' : 0.1056, '20_MG_L' : 0.0})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.8946, '15_MG_L' : 0.1054, '20_MG_L' : 0.0})
	elif(CBODN_12_30 == '15_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.9682, '20_MG_L' : 0.03180000000000005})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.969, '20_MG_L' : 0.031000000000000028})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.9693, '20_MG_L' : 0.03069999999999995})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.9695, '20_MG_L' : 0.03049999999999997})
	else:
		if(CNON_12_30 == '2_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0364, '20_MG_L' : 0.9636})
		elif(CNON_12_30 == '4_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0373, '20_MG_L' : 0.9627})
		elif(CNON_12_30 == '6_MG_L'):
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.0377, '20_MG_L' : 0.9623})
		else:
			CBODN_12_45 ~= choice({'5_MG_L' : 0.0, '10_MG_L' : 0.0, '15_MG_L' : 0.038, '20_MG_L' : 0.962})


if (CKNI_12_30 == '20_MG_L'):
	if (CKND_12_30 == '2_MG_L'):
		if (CKNN_12_30 == '0_5_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.9524, '4_MG_L' : 0.0476, '6_MG_L' : 0.0})
		elif(CKNN_12_30 == '1_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.9444, '4_MG_L' : 0.0556, '6_MG_L' : 0.0})
		else:
			CKND_12_45 ~= choice({'2_MG_L' : 0.9286, '4_MG_L' : 0.0714, '6_MG_L' : 0.0})
	elif(CKND_12_30 == '4_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9921, '6_MG_L' : 0.007900000000000018})
		elif(CKNN_12_30 == '1_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9841, '6_MG_L' : 0.015900000000000025})
		else:
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9683, '6_MG_L' : 0.03169999999999995})
	else:
		if(CKNN_12_30 == '0_5_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0317, '6_MG_L' : 0.9683})
		elif(CKNN_12_30 == '1_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0238, '6_MG_L' : 0.9762})
		else:
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0079, '6_MG_L' : 0.9921})
elif(CKNI_12_30 == '30_MG_L'):
	if(CKND_12_30 == '2_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.9127, '4_MG_L' : 0.0873, '6_MG_L' : 0.0})
		elif(CKNN_12_30 == '1_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.9048, '4_MG_L' : 0.0952, '6_MG_L' : 0.0})
		else:
			CKND_12_45 ~= choice({'2_MG_L' : 0.8889, '4_MG_L' : 0.1111, '6_MG_L' : 0.0})
	elif(CKND_12_30 == '4_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9524, '6_MG_L' : 0.047599999999999976})
		elif(CKNN_12_30 == '1_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9444, '6_MG_L' : 0.05559999999999998})
		else:
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9286, '6_MG_L' : 0.07140000000000002})
	else:
		if(CKNN_12_30 == '0_5_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		elif(CKNN_12_30 == '1_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		else:
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
else:
	if(CKND_12_30 == '2_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.873, '4_MG_L' : 0.127, '6_MG_L' : 0.0})
		elif(CKNN_12_30 == '1_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.8651, '4_MG_L' : 0.1349, '6_MG_L' : 0.0})
		else:
			CKND_12_45 ~= choice({'2_MG_L' : 0.8492, '4_MG_L' : 0.1508, '6_MG_L' : 0.0})
	elif(CKND_12_30 == '4_MG_L'):
		if(CKNN_12_30 == '0_5_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9127, '6_MG_L' : 0.08730000000000004})
		elif(CKNN_12_30 == '1_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.9048, '6_MG_L' : 0.09519999999999995})
		else:
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.8889, '6_MG_L' : 0.11109999999999998})
	else:
		if(CKNN_12_30 == '0_5_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		elif(CKNN_12_30 == '1_MG_L'):
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})
		else:
			CKND_12_45 ~= choice({'2_MG_L' : 0.0, '4_MG_L' : 0.0, '6_MG_L' : 1.0})


if (CBODD_12_30 == '15_MG_L'):
	if (CNOD_12_30 == '0_5_MG_L'):
		if (CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.8893, '1_MG_L' : 0.1107, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.6354, '1_MG_L' : 0.3646, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_30 == '1_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.3675, '1_MG_L' : 0.6325, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.2405, '1_MG_L' : 0.7595, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.1135, '1_MG_L' : 0.8865, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.9298, '2_MG_L' : 0.0702, '4_MG_L' : 0.0})
	elif(CNOD_12_30 == '2_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2972, '2_MG_L' : 0.7028, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2338, '2_MG_L' : 0.7662, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1703, '2_MG_L' : 0.8297, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0433, '2_MG_L' : 0.9567, '4_MG_L' : 0.0})
	else:
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2129, '4_MG_L' : 0.7871})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1812, '4_MG_L' : 0.8188})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1494, '4_MG_L' : 0.8506})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0859, '4_MG_L' : 0.9141})
elif(CBODD_12_30 == '20_MG_L'):
	if(CNOD_12_30 == '0_5_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.9816, '1_MG_L' : 0.0184, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.7276, '1_MG_L' : 0.2724, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_30 == '1_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.4905, '1_MG_L' : 0.5095, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.3635, '1_MG_L' : 0.6365, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.2366, '1_MG_L' : 0.7634, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.9913, '2_MG_L' : 0.0087, '4_MG_L' : 0.0})
	elif(CNOD_12_30 == '2_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3711, '2_MG_L' : 0.6289, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3076, '2_MG_L' : 0.6924, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2441, '2_MG_L' : 0.7559, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1171, '2_MG_L' : 0.8829, '4_MG_L' : 0.0})
	else:
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2539, '4_MG_L' : 0.7461})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2222, '4_MG_L' : 0.7778})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1904, '4_MG_L' : 0.8096})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1269, '4_MG_L' : 0.8731})
elif(CBODD_12_30 == '25_MG_L'):
	if(CNOD_12_30 == '0_5_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.7994, '1_MG_L' : 0.2006, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_30 == '1_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.5862, '1_MG_L' : 0.4138, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.4592, '1_MG_L' : 0.5408, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.3322, '1_MG_L' : 0.6678, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0783, '1_MG_L' : 0.9217, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_30 == '2_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.4285, '2_MG_L' : 0.5715, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.365, '2_MG_L' : 0.635, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3015, '2_MG_L' : 0.6985, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.1745, '2_MG_L' : 0.8255, '4_MG_L' : 0.0})
	else:
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2858, '4_MG_L' : 0.7142})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2541, '4_MG_L' : 0.7459})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2223, '4_MG_L' : 0.7777000000000001})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1588, '4_MG_L' : 0.8412})
else:
	if(CNOD_12_30 == '0_5_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 1.0, '1_MG_L' : 0.0, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.8568, '1_MG_L' : 0.1432, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_30 == '1_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.6627, '1_MG_L' : 0.3373, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.5358, '1_MG_L' : 0.4642, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.4088, '1_MG_L' : 0.5912, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.1548, '1_MG_L' : 0.8452, '2_MG_L' : 0.0, '4_MG_L' : 0.0})
	elif(CNOD_12_30 == '2_MG_L'):
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.4744, '2_MG_L' : 0.5256, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.4109, '2_MG_L' : 0.5891, '4_MG_L' : 0.0})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.3474, '2_MG_L' : 0.6526, '4_MG_L' : 0.0})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.2204, '2_MG_L' : 0.7796, '4_MG_L' : 0.0})
	else:
		if(CNON_12_30 == '2_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.3113, '4_MG_L' : 0.6887})
		elif(CNON_12_30 == '4_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2796, '4_MG_L' : 0.7203999999999999})
		elif(CNON_12_30 == '6_MG_L'):
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.2478, '4_MG_L' : 0.7522})
		else:
			CNOD_12_45 ~= choice({'0_5_MG_L' : 0.0, '1_MG_L' : 0.0, '2_MG_L' : 0.1843, '4_MG_L' : 0.8157})


'''
compiler = SPPL_Compiler(data)
namespace = compiler.execute_module()
model=namespace.model
CBODD_12_00 = Id('CBODD_12_00')
CBODD_12_15 = Id('CBODD_12_15')
CBODD_12_30 = Id('CBODD_12_30')
CBODD_12_45 = Id('CBODD_12_45')
CBODN_12_00 = Id('CBODN_12_00')
CBODN_12_15 = Id('CBODN_12_15')
CBODN_12_30 = Id('CBODN_12_30')
CBODN_12_45 = Id('CBODN_12_45')
CKND_12_00 = Id('CKND_12_00')
CKND_12_15 = Id('CKND_12_15')
CKND_12_30 = Id('CKND_12_30')
CKND_12_45 = Id('CKND_12_45')
CKNI_12_00 = Id('CKNI_12_00')
CKNI_12_15 = Id('CKNI_12_15')
CKNI_12_30 = Id('CKNI_12_30')
CKNI_12_45 = Id('CKNI_12_45')
CKNN_12_00 = Id('CKNN_12_00')
CKNN_12_15 = Id('CKNN_12_15')
CKNN_12_30 = Id('CKNN_12_30')
CKNN_12_45 = Id('CKNN_12_45')
CNOD_12_00 = Id('CNOD_12_00')
CNOD_12_15 = Id('CNOD_12_15')
CNOD_12_30 = Id('CNOD_12_30')
CNOD_12_45 = Id('CNOD_12_45')
CNON_12_00 = Id('CNON_12_00')
CNON_12_15 = Id('CNON_12_15')
CNON_12_30 = Id('CNON_12_30')
CNON_12_45 = Id('CNON_12_45')
C_NI_12_00 = Id('C_NI_12_00')
C_NI_12_15 = Id('C_NI_12_15')
C_NI_12_30 = Id('C_NI_12_30')
C_NI_12_45 = Id('C_NI_12_45')

events = [CNON_12_45 << {"2_MG_L"}, CNON_12_45 << {"4_MG_L"}, CNON_12_45 << {"6_MG_L"}, CNON_12_45 << {"10_MG_L"}]
for i in range(len(events)):
	start_time=time.time()
	query_prob=model.prob(events[i])
	end_time = time.time()
	print("--- %s seconds ---" % (end_time - start_time))

	print(query_prob)

