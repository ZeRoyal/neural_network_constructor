import numpy as np

def load(name='data.txt'):
	t = []
	d = []
	with open(name) as f:
		for line in f:
			a = map(int, line.split())
			t.append(a[0])
			d.append(np.reshape(a[1:], (8, 8)))
	return t, d
