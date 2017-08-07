import numpy as np
import options as o
import layers as l
import os




class NeuralNetwork():
    def __init__(self, layers, loss_func):

        self.input_layer = layers[0]
        self.output_layer = layers[-1]
        self.layers = [(prev_layer, layer) for prev_layer, layer in zip(layers[:-1], layers[1:])]

        self.sequence = layers
        self.loss_func = loss_func

        for prev_layer, layer in self.layers:
            layer.connecting(prev_layer)


    def extract(self, n):

        self.layer_extr = self.sequence [n]
        return self.layer_extr.w, self.layer_extr.b

    
    def feedforward(self, x):
        self.input_layer.z = x
        self.input_layer.a = x

        for prev_layer, layer in self.layers:
            layer.feedforward(prev_layer)

    def backpropagate(self, batch, optimizer):
        sum_der_w = {layer: np.zeros_like(layer.w) for _, layer in self.layers}
        sum_der_b = {layer: np.zeros_like(layer.b) for _, layer in self.layers}

        for x, y in batch:
            self.feedforward(x)


            loss = self.loss_func(self.output_layer.a, y)
            if self.loss_func == o.crossentropy:

                delta = o.qu(self.output_layer.a, y)
            else:
                delta = loss * self.output_layer.der_activator(self.output_layer.z, y)
                     
            
            
            for prev_layer, layer in reversed(self.layers):
                der_w, der_b, prev_delta = layer.backpropagate(prev_layer, delta)
                sum_der_w[layer] += der_w
                sum_der_b[layer] += der_b
                delta = prev_delta

        optimizer.apply(self.layers, sum_der_w, sum_der_b, len(batch))

  







def train(net, optimizer, num_epochs, batch_size, trn_set, num_class, vld_set=None):

    inputs = trn_set
    
    for i in range(num_epochs):
        np.random.shuffle(inputs)

        batches = [inputs[j:j+batch_size] for j in range(0, len(inputs), batch_size)]
        inputs_done=0.
        r=0.
        print(">>> Training Epoch %02d. [Batch size = %s. Training set length = %s] <<<" % (i+1, batch_size, len(inputs)))        
        bar=o.ProgBar(len(batches))
        for batch in batches:
            net.backpropagate(batch, optimizer)
            inputs_done += len(batch)
            r=r+1.
            bar.show(r, 4)
            '''print("Epoch %02d %0.1f%% " % (i+1,  float(inputs_done)/len(inputs)*100))'''
            if vld_set and (r % np.round(0.2*len(batches)) == 0.):
                accuracy = test(net, vld_set)
                print("Epoch %02d [%0.1f%%]. Validation accuracy: %0.2f%%" % (i+1, r/len(batches)*100, accuracy/len(vld_set)*100))



def test(net, tst_set):
    assert isinstance(net, NeuralNetwork)


    tests = tst_set

    r=0
    bar=o.ProgBar(len(tests))
    accuracy = 0.
    for x, y in tests:
        net.feedforward(x)
        r=r+1
        bar.show(r, 3)        
        if (np.argmax(net.output_layer.a) == y):
            accuracy += 1.

        '''print(np.argmax(net.output_layer.a),y)'''
        
    return accuracy




def save_net(net, num_spec_layers = [1, 3], name_n = 'default', active = None):
    if active:

        for i in num_spec_layers:
            weight_out, bias_out = net.extract(i)
            if np.ndim(weight_out) == 4:
                
                if not os.path.exists(os.path.join(os.getcwd(), 'saves/save(' + name_n + ')/cnn_filters')):
                    os.makedirs(os.path.join(os.getcwd(), 'saves/save(' + name_n + ')/cnn_filters'))
                    
                np.savetxt(os.path.join(os.getcwd(), 'saves/save(' + name_n + ')/cnn_filters','filter' + '_' +  str(i) + '.txt'),\
                           np.ravel(weight_out), delimiter=', ',newline='\n',fmt="%s")

                np.savetxt(os.path.join(os.getcwd(), 'saves/save(' + name_n + ')/cnn_filters','biases' + '_' +  str(i) + '.txt'),\
                           bias_out, delimiter=', ',newline='\n',fmt="%s")
            else:

                if not os.path.exists(os.path.join(os.getcwd(), 'saves/save(' + name_n + ')/fcl_weights')):
                    os.makedirs(os.path.join(os.getcwd(), 'saves/save(' + name_n + ')/fcl_weights'))
                    
                np.savetxt(os.path.join(os.getcwd(), 'saves/save(' + name_n + ')/fcl_weights','weights_'+ str(i) + '.txt'), \
                           weight_out, delimiter=', ', newline=', ')
                np.savetxt(os.path.join(os.getcwd(), 'saves/save(' + name_n + ')/fcl_weights','biases_'+ str(i) + '.txt'), \
                           bias_out, delimiter=', ', newline=', ')                




def load_weights(name='save(default2)'):
        dir_save = os.path.join(os.getcwd(), 'saves/' + name)
        dir_cnn = os.path.join(dir_save, 'cnn_filters')
        dir_fcl = os.path.join(dir_save, 'fcl_weights')

        cnn_w = []
        cnn_list = os.listdir(dir_cnn)        

        fcl_w = []
        fcl_list = os.listdir(dir_fcl)

        for cnn in cnn_list:
	    cnn_w.append(open(dir_cnn + '/' + cnn, 'r').read())

        for fcl in fcl_list:
	    fcl_w.append(open(dir_fcl + '/' + fcl, 'r').read())

	return cnn_w, fcl_w


def Loaded_nn(name='save(cnn(3_5_2)(0_1_2_8))'):

        batch_of_weights = load_weights(name)

        ccn_pack, fcl_pack = batch_of_weights
        cnn_b, cnn_w = ccn_pack
        fcl_b, fcl_w = fcl_pack

        
        cnn_w = map(float,np.delete(cnn_w.split('\n'),-1))
        cnn_b = map(float,np.delete(cnn_b.split('\n'),-1))

        
        fcl_w = map(float,np.delete(fcl_w.split(', '),-1))
        fcl_b = map(float,np.delete(fcl_b.split(', '),-1))
        '''
        fcl_w = map(float,np.delete(fcl_w.split('\n'),-1))
        fcl_b = map(float,np.delete(fcl_b.split('\n'),-1))
        '''


        weights = [cnn_w, cnn_b, fcl_w, fcl_b]
        
        con_layer = l.ConvLayer(3, kernel=5, init_val=o.randSc, activator=o.sigmoid)
        fc_layer = l.FCLayer(height=10, init_val=o.randSc, activator=o.softmax)

        net = NeuralNetwork([
            l.InputLayer(height=28, width=28),
            con_layer,
            l.MaxLayer(pool=2),
            fc_layer
        ], o.crossentropy)

        con_layer.w = np.reshape(weights[0], (np.shape(con_layer.w)))
        con_layer.b = np.reshape(weights[1], (np.shape(con_layer.b)))
        fc_layer.w = np.reshape(weights[2], (np.shape(fc_layer.w)))
        fc_layer.b = np.reshape(weights[3], (np.shape(fc_layer.b)))

        return net
