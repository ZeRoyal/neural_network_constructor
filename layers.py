import numpy as np
import options as o


def lambda1(x):
        return x

class InputLayer():
    def __init__(self, height, width):
        
        self.depth = 1
        self.height = height
        self.width = width
        self.n_out = self.depth * self.height * self.width
        self.der_activator = lambda1



class FCLayer():
    def __init__(self, height, init_val, activator):
        
        self.depth = 1
        self.height = height
        self.width = 1
        self.n_out = self.depth * self.height * self.width
        self.init_val = init_val
        self.activator = activator
        self.der_activator = getattr(o, "der_%s" % activator.__name__)

    def connecting(self, prev_layer):
        self.w = self.init_val((self.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)
        self.b = np.zeros((self.n_out, 1))



    def feedforward(self, prev_layer):


        prev_a=np.ravel(prev_layer.a)
        prev_a = np.reshape(prev_a, (len(prev_a), 1))
        self.z = np.dot(self.w , prev_a) + self.b
        self.a = self.activator(self.z)
        


    def backpropagate(self, prev_layer, delta):
        
        prev_a = np.reshape(prev_layer.a,(np.size(prev_layer.a), 1))

        
        der_w = np.dot(delta, prev_a.T)

        der_b = np.copy(delta)

        prev_delta = np.reshape(np.dot(self.w.T ,delta),(np.shape(prev_layer.z))) * prev_layer.der_activator(prev_layer.z)

        return der_w, der_b, prev_delta


class ConvLayer():
    def __init__(self, depth, kernel, init_val, activator):

        self.depth = depth
        self.kernel = kernel
        self.init_val = init_val
        self.activator = activator
        self.der_activator = getattr(o, "der_%s" % activator.__name__)

    def connecting(self, prev_layer):
        self.trans_length = 1
        self.height = ((prev_layer.height - self.kernel) // self.trans_length) + 1
        self.width  = ((prev_layer.width  - self.kernel) // self.trans_length) + 1
        self.n_out = self.depth * self.height * self.width

        self.w = 7*self.init_val((self.depth, prev_layer.depth, self.kernel, self.kernel),
            prev_layer.n_out, self.n_out)
        self.b = np.zeros((self.depth, 1))


        
    def feedforward(self, prev_layer):


        prev_a = np.reshape(prev_layer.a,(1,np.shape(prev_layer.a)[0],np.shape(prev_layer.a)[1]))
        
        filters_ch_out = self.w.shape[0]
        filters_ch_in = self.w.shape[1]
        filters_h = self.w.shape[2]
        filters_w = self.w.shape[3]

        image_ch = np.shape(prev_a)[0]
        image_h = np.shape(prev_a)[1]
        image_w = np.shape(prev_a)[2]

        trans = 1
        new_h = ((image_h - filters_h) // trans) + 1
        new_w = ((image_w - filters_w) // trans) + 1


        self.z = np.zeros((filters_ch_out, new_h, new_w))
        for f_c in range(filters_ch_out):
            for i_c in range(image_ch):
                filter1 = self.w[f_c, i_c]
                for i, m in enumerate(range(0, image_h - filters_h + 1, self.trans_length)):
                    for j, n in enumerate(range(0, image_w - filters_w + 1, self.trans_length)):
                        
                        prev_a_floor = np.array(prev_a)[i_c, m:m+filters_h, n:n+filters_w]
                        self.z[f_c, i, j] += np.correlate(prev_a_floor.ravel(), filter1.ravel())

        for i in range(self.depth):
            self.z[i] += self.b[i]

        self.a = np.vectorize(self.activator)(self.z)





    def backpropagate(self, prev_layer, delta):

        prev_a = np.reshape(prev_layer.a,(1,np.shape(prev_layer.a)[0],np.shape(prev_layer.a)[1]))


        der_w = np.zeros_like(self.w)
        for r in range(self.depth):
            for p_l in range(prev_layer.depth):
                for k1 in range(self.kernel):
                    for k2 in range(self.kernel):
                        prev_a_floor = np.array(prev_a)[p_l, k2:k2 + self.height - self.kernel +
                                                        1:self.trans_length, k1:k1 + self.width - self.kernel + 1:self.trans_length]
                        delta_floor  =  np.array(delta)[r, k2:k2 + self.height - self.kernel +
                                                        1:self.trans_length, k1:k1 + self.width-self.kernel + 1:self.trans_length]
                        der_w[r, p_l, k1, k2] = np.sum(prev_a_floor * delta_floor)
                        
        
        der_b = np.empty((self.depth, 1)) 
        for i in range(self.depth):
            der_b[i] = np.sum(delta[i])

        prev_delta = np.zeros_like(prev_a)

        for r in range(self.depth):
            for p_l in range(prev_layer.depth):
                kernel1 = self.w[r, p_l]
                for i, m in enumerate(range(0, prev_layer.height - self.kernel + 1, self.trans_length)):
                    for j, n in enumerate(range(0, prev_layer.width - self.kernel + 1, self.trans_length)):
                        prev_delta[p_l, m:m + self.kernel, n:n + self.kernel] += kernel1 * delta[r, i, j]
        prev_delta  *= prev_layer.der_activator(prev_layer.z)


        return der_w, der_b, prev_delta



class MaxLayer():
    def __init__(self, pool):
        
        self.pool = pool
        self.der_activator = lambda1

    def connecting(self, prev_layer):

        self.depth = prev_layer.depth
        self.height = ((prev_layer.height - self.pool) // self.pool) + 1
        self.width  = ((prev_layer.width  - self.pool) // self.pool) + 1
        self.n_out = self.depth * self.height * self.width

        self.w = np.empty((0))
        self.b = np.empty((0))

    def feedforward(self, prev_layer):

        prev_a = prev_layer.a

        prev_layer_fmap_size = prev_layer.height


        self.z = np.zeros((self.depth, self.height, self.width))
        for r, t in zip(range(self.depth), range(prev_layer.depth)):

            for i, m in enumerate(range(0, prev_layer.height, self.pool)):
                for j, n in enumerate(range(0, prev_layer.width, self.pool)):
                    prev_a_floor = prev_a[t, m:m+self.pool, n:n+self.pool]
                    self.z[r, i, j] = np.max(prev_a_floor)

        self.a = self.z

    def backpropagate(self, prev_layer, delta):

        prev_a = prev_layer.a

        der_w = []

        der_b = []

        prev_delta = np.zeros_like(prev_a)
        for r, t in zip(range(self.depth), range(prev_layer.depth)):

            for i, m in enumerate(range(0, prev_layer.height, self.pool)):
                for j, n in enumerate(range(0, prev_layer.width, self.pool)):
                    prev_a_floor = prev_a[t, m:m+self.pool, n:n+self.pool]

                    max_unit_index = np.unravel_index(np.argmax(prev_a_floor), np.shape(prev_a_floor))
                    prev_delta_window = np.zeros_like(prev_a_floor)
                    prev_delta_window[max_unit_index] = delta[t, i, j]
                    prev_delta[r, m:m+self.pool, n:n+self.pool] = prev_delta_window

        return der_w, der_b, prev_delta



class LSTMLayer():
    def __init__(self, height, init_val, activator1, activator2, activator3):
        
        self.depth = 1
        self.height = height
        self.width = 1
        self.n_out = self.depth * self.height * self.width
        self.init_val = init_val
        self.activator1 = activator1
        self.activator2 = activator2
        self.activator3 = activator3
        self.der_activator1 = getattr(o, "der_%s" % activator1.__name__)
        self.der_aactivator2 = getattr(o, "der_%s" % activator2.__name__)
        self.der_activator3 = getattr(o, "der_%s" % activator3.__name__)



    def connecting(self, prev_layer):

        self.wf = self.init_val((self.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)
        self.bf = np.zeros((self.n_out, 1))

        self.wi = self.init_val((self.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)
        self.bi = np.zeros((self.n_out, 1))

        self.wc = self.init_val((self.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)
        self.bc = np.zeros((self.n_out, 1))

        self.wo = self.init_val((self.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)
        self.bo = np.zeros((self.n_out, 1))

        self.wf2 = self.init_val((prev_layer.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)


        self.wi2 = self.init_val((prev_layer.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)


        self.wc2 = self.init_val((prev_layer.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)


        self.wo2 = self.init_val((prev_layer.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)


        
    def feedforward(self, prev_layer, cell_state, hid_state):

        self.cell_old = cell_state
        self.a_old = hid_state
        
        prev_a=np.ravel(prev_layer.a)
        prev_a = np.reshape(prev_a, (len(prev_a), 1))

        self.zf = np.dot(self.wf , prev_a) + np.dot(self.wf2 , self.a_old) + self.bf
        self.af = self.activator1(self.zf)

        self.zi = np.dot(self.wi , prev_a) + np.dot(self.wf2 , self.a_old) + self.bi
        self.ai = self.activator1(self.zi)        

        self.zc = np.dot(self.w , prev_a) + np.dot(self.wf2 , self.a_old) + self.bc
        self.ac = self.activator2(self.zc)

        self.zo = np.dot(self.w , prev_a) + np.dot(self.wf2 , self.a_old) + self.bo
        self.ao = self.activator1(self.zo)

        self.cell =  self.af * self.cell_old + self.ai * self.ac
        self.a = self.ao * self.activator2(self.cell)

        cash = [self.zf, self.zi, self.zc, self.zo, self.cell, self.a]



##### ......... #####
    def backpropagate(self, prev_layer, delta, delta_h_next, delta_cell_next, cash):
        
        assert delta.shape == self.z.shape == self.a.shape



        prev_a = np.reshape(prev_layer.a,(np.size(prev_layer.a), 1))

        
        der_wf = np.dot(delta, prev_a.T)
        der_bf = np.copy(delta)

        der_wi = np.dot(delta, prev_a.T)
        der_bi = np.copy(delta)        

        der_wc = np.dot(delta, prev_a.T)
        der_bc = np.copy(delta)

        der_wo = np.dot(delta, prev_a.T)
        der_bo = np.copy(delta)


        der_cell = np.dot(delta, prev_a.T)
        der_a = np.copy(delta)

        prev_delta = np.reshape(np.dot(self.w.T ,delta),(np.shape(prev_layer.z))) * prev_layer.der_act_func(prev_layer.z)

        return der_w, der_b, prev_delta


