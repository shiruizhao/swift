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
import re

isclose = lambda a, b : abs(a-b) < 1e-10


import matplotlib.pyplot as plt

import subprocess
import numpy as np

#benchmark 
#CNON_12_45
exact_results = np.array([0.0042, 0.9048, 0.0911, 0.0])
exact_runtime = 2.58
def KL(a, b):
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

x = range(100, 50000, 100)
KL_results = np.zeros((3, len(x)))
runtime_results = np.zeros((3, len(x)))
appro_results = np.zeros(4)
filelist=['../lw_out_rng/water_xor', '../lw_out_rng/water_default_prng', '../lw_out_rng/water_lfsr']
for cmd_index in range(len(filelist)):
    kl_index = 0
    for i in x:
        cmd = [filelist[cmd_index], str(i), str(i/2)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        resultlist=result.stdout.decode('utf-8').split("\n")
        print(resultlist)
        sampletime = float(resultlist[-2].split(":")[1].split("s")[0])
        print(appro_results)
        appro_results[0] = float(resultlist[1].split(" -> ")[1])
        if re.search("->", resultlist[2]):
            appro_results[1] = float(resultlist[2].split(" -> ")[1])
        else:
            appro_results[1] = 0.0
        if len(resultlist) >3:
            if re.search("->", resultlist[3]):
                appro_results[2] = float(resultlist[3].split(" -> ")[1])
            else:
                appro_results[2] = 0.0
        else:
            appro_results[2] = 0.0
        print("appro result:")
        print(appro_results)
        print("KL:{KL}".format(KL=KL(exact_results, appro_results)))
        KL_results[cmd_index][kl_index] = (KL(exact_results, appro_results))
        runtime_results[cmd_index][kl_index] = (sampletime)
        kl_index += 1

fig,ax=plt.subplots()
ax.plot(runtime_results[0], KL_results[0], marker="o", label="Xoroshiro128+")
plt.xlabel("runtime[s]")
plt.ylabel('KL divergence')
ax.plot(runtime_results[1], KL_results[1], marker="o", label="MT19337")
ax.plot(runtime_results[2], KL_results[2], marker="o", label="LFSR")
#ax.plot( exact_runtime, 0.0 , marker="*", label = "ref. exact infer.")
ax.legend(loc='right')
plt.savefig("water_KL.runtime.jpg", dpi=100, bbox_inches='tight')
