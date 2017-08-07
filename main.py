import numpy as np
import options as o
import layers as l
import net as n
import format as f
import mnist_loader as ml
import os
import random
from sklearn import datasets
import pickle
import token_mkr as t


#Y, X = f.load()



def shuffle(X, Y):
    XY = zip(X, Y)
    random.shuffle(XY)
    return map(lambda x: x[0], XY),  map(lambda x: x[1], XY)

def FCL1():
    net = n.NeuralNetwork([
        l.InputLayer(height=8, width=8),
        l.FCLayer(10, init_val=o.randSc, activator=o.sigmoid)
    ], o.qu)
    optimizer = o.SGD(0.1)
    num_epochs = 2000
    batch_size = 100
    num_class = 10
    return net, optimizer, num_epochs, batch_size, num_class


def CNN():
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.ConvLayer(10, kernel=3, init_val=o.randSc, activator=o.sigmoid),
        l.MaxLayer(pool=2),
        l.FCLayer(height=10, init_val=o.randSc, activator=o.softmax)
        
    ], o.crossentropy)
    optimizer = o.SGD(0.1)
    num_epochs = 2
    batch_size = 50
    num_class=2
    return net, optimizer, num_epochs, batch_size, num_class


if __name__ == "__main__":




    print("Loading1")
    #size = len(t.pn)
    #trn_set, tst_set = t.pn[:size/2], t.pn[size/2:]
    trn_set, tst_set = ml.load_data_wrapper()
    

    '''trn_x, trn_y = trn_set '''
    '''tst_x, tst_y = tst_set'''



    
    trn_set= [(np.reshape(x,(np.sqrt(len(x)),np.sqrt(len(x)))), y) for x, y in trn_set][:500]

    tst_set= [(np.reshape(x,(np.sqrt(len(x)),np.sqrt(len(x)))), y) for x, y in tst_set][:500]
    
    '''
    tst_set= np.ones_like(tst_set)-tst_set
    '''


    vld_set=tst_set[:1000]

    
    
    print("Loading2")
    net, optimizer, num_epochs, batch_size, num_class = CNN()


    print("Training network...")
    n.train(net, optimizer, num_epochs, batch_size, trn_set, num_class, vld_set=None)

    print('Saving network (pkl)...')
    pickle.dump(net, open("save1.pkl", "wb"))

    
    #print("Saving network...")
    #n.save_net(net, num_spec_layers = [1, 3], name_n = 'cnn(3_5_2)(0_1_2_8)', active = True )
    
    print("Testing network...")
    accuracy = n.test(net, tst_set)
    print("Test accuracy: %0.2f%%" % (accuracy*100/(len(tst_set))))
    
    '''
    net_load = n.Loaded_nn(name='save(math1)')
    print("Testing Loaded Network...")
    acc_load = n.test(net_load, tst_set)
    print("Test accuracy of loaded nn: %0.2f%%" % (acc_load*100./(len(tst_set))))
    
    net_tr=pickle.load(open("save.pkl", "rb"))
    print("Testing network...")
    accuracy = n.test(net_tr, tst_set)
    print("Test accuracy: %0.2f%%" % (accuracy*100/(len(tst_set))))
    '''
