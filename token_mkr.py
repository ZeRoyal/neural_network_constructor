import os, re, time, random

def load_trained2(name='gl50d.txt'):
        k = []
        v = []
        with open(name, 'r') as f:
                for line in f:
                        a=line.split(' ',1)
                        k.append(a[0])
                        v.append(map(float,a[1].split()))
        return  k, v

def load_trained(name='gl50d.txt'):
        d = {}
        for line in open(name, 'r'):
                tokens = line.split()
                d[tokens[0]] = map(float, tokens[1:])
        return d

dict1= load_trained('../word2vec/gl50d.txt')
zero = [0.]*len(dict1.values()[0])

def token(word):        
        if dict1.has_key(word):
                return dict1.get(word)  
        else:
                return zero

def sup_load(path):
        reg = re.compile('\w+')
        texts = []
        files = os.listdir(path)
        for f in files:
                texts.append(reg.findall(open(path+'/'+f, 'r').read().lower()))
        return texts

def ref(tokenized_list, lenght, zero= zero):

        for t in tokenized_list:
                if len(t) < lenght:
                        for i in range(lenght - len(t)):
                                t.append(zero)
                else:
                        del t[lenght:]
        
        return tokenized_list

def shuffle(pos, neg):
        p = map(lambda x: (x, 1), pos)
        n = map(lambda x: (x, 0), neg)
        pn = p + n
        random.shuffle(pn)
        return pn

pos_batch = sup_load("../word2vec/data1/pos")[:1000]
neg_batch = sup_load("../word2vec/data1/neg")[:1000]

p = [[token(s) for s in xs] for xs in pos_batch]
p = ref(p, 100)

n = [[token(s) for s in xs] for xs in neg_batch]
n = ref(n, 100)

pn = shuffle(p, n)












