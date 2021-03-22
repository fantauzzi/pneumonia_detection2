from matplotlib import pyplot as plt
from kerastuner.engine.hyperparameters import Float

''' Display evidence that log sampling in Keras Tuner actually samples with the same probability from each order of 
magnitude range'''
min_exp = -8
max_exp = 2
n_samples = 100000

hp = Float(name='test', min_value=10 ** min_exp, max_value=10 ** max_exp, sampling='log')

values = [0] * n_samples
for i in range(n_samples):
    values[i] = hp.random_sample()

''' If OK, bars of the histogram, one per order of magnitude range, should have about the same height'''
bins = [10 ** x for x in range(min_exp, max_exp + 1)]
plt.xscale('log')
plt.hist(values, bins=bins)
plt.show()
