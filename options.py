import numpy as np
import sys



def randSc(shape, num_neurons_in, num_neurons_out):
    scale = np.sqrt(6. / (num_neurons_in + num_neurons_out))
    return np.random.uniform(low=-scale, high=scale, size=shape)

def vec_res(j,n):
    e = np.zeros((n, 1))
    e[j] = 1.0
    return e

def save(a, name='weights.txt'):
	with open(name, 'w') as f:
            f.write(np.string(a))



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def der_sigmoid(x, y=None):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def der_softmax(x, y=None):
    s = softmax(x)
    if y is not None:
        k = s[np.where(y == 1)]
        a = - k * s
        a[np.where(y == 1)] = k * (1 - k)
        return a
    return s * (1 - s)


def relu(x):
    if x < 0:
        return 0
    else:
        return x

def der_relu(x, y=None):
    if x < 0:
        return 0
    else:
        return 1




def qu(u, y):
    return u-y



def crossentropy(u, y):
    u = u.flatten() / np.sum(u)
    i = np.where(y.flatten() == 1)
    return np.log(u)[i]





class SGD():
    def __init__(self, lr):
        self.lr = lr

    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            gw = sum_der_w[layer]/batch_len
            layer.w += -(self.lr*gw)
            
            gb = sum_der_b[layer]/batch_len
            layer.b += -(self.lr*gb)




class ProgBar:
	def __init__(self, work_size, bar_size=50):
		self.bsize = bar_size
		self.wsize = work_size

	def show(self, prog=0, style_n = 0):
                style= ['=', 'O', '#', '>', '/', '|'][style_n]
		fac = float(self.bsize) / self.wsize
		bars = int(prog * fac)
		dots = self.bsize - bars
		sys.stdout.write('\r['+style*bars+'.'*dots+']'+'\n'*(dots==0))
		sys.stdout.flush()




