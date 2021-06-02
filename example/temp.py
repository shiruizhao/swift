import numpy as np
import matplotlib.pyplot as plt

def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


values1 = [1.346112,1.337432,1.246655]
values2 = [1.033836,1.082015,1.117323]

print (KL(values1, values2))
print (KL(values2, values2))

fig,ax=plt.subplots()
ax.plot(10,0, marker="*")
plt.ylabel('KL divergence')
plt.xlabel("runtime[s]")
plt.savefig("alarm_KL.runtime.jpg")

