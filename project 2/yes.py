import numpy as np


region_shape = (5, 5)

c1 = np.outer(
    np.exp(-((np.linspace(-15, 15, num=region_shape[0]) / 2.0) ** 2) / 2),
    np.exp(-((np.linspace(-15, 15, num=region_shape[1]) / 2.0) ** 2) / 2),
)


h = np.arange(0, region_shape[0]) - (region_shape[0] - 1) / 2
w = np.arange(0, region_shape[1]) - (region_shape[1] - 1) / 2
c2 = np.outer(
    np.exp(-(h ** 2) / 2),
    np.exp(-(w ** 2) / 2),
)
c2 /= c2.max()

print(c1)
print(c2)
