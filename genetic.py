import numpy as np
from network import Network
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Genetic:
    
    def __init__(self,pop_size,nlayers,max_nfilters,max_sfilters):
        self.pop_size = pop_size
        self.nlayers = nlayers
        self.max_nfilters = max_nfilters
        self.max_sfilters = max_sfilters
        self.max_acc = 0
        self.best_arch = np.zeros((1,6))
        self.gen_acc = []
    
    def generate_population(self):
        np.random.seed(0)
        pop_nlayers = np.random.randint(1,self.max_nfilters,(self.pop_size,self.nlayers))
        pop_sfilters = np.random.randint(1,self.max_sfilters,(self.pop_size,self.nlayers))
        pop_total = np.concatenate((pop_nlayers,pop_sfilters),axis=1)
        return pop_total
    
    def select_parents(self,pop,nparents,fitness):
        parents = np.zeros((nparents,pop.shape[1]))
        for i in range(nparents):
            best = np.argmax(fitness)
            parents[i] = pop[best]
            fitness[best] = -99999
        return parents
    
    def crossover(self,parents):
        nchild = self.pop_size - parents.shape[0]
        nparents = parents.shape[0]
        child = np.zeros((nchild,parents.shape[1]))
        for i in range(nchild):
            first = i % nparents
            second = (i+1) % nparents
            child[i,:2] = parents[first][:2]
            child[i,2] = parents[second][2]
            child[i,3:5] = parents[first][3:5]
            child[i,5] = parents[second][5]
        return child

    def mutation(self,child):
        for i in range(child.shape[0]):
            val = np.random.randint(1,6)
            ind = np.random.randint(1,4) - 1
            if child[i][ind] + val > 100:
                child[i][ind] -= val
            else:
                child[i][ind] += val
            val = np.random.randint(1,4)
            ind = np.random.randint(4,7) - 1
            if child[i][ind] + val > 20:
                child[i][ind] -= val
            else:
                child[i][ind] += val
        return child
    
    def fitness(self,pop,train_loader,test_loader,epochs, total):
        pop_acc = []
        for i in range(pop.shape[0]):
            nfilters = pop[i][0:2]
            sfilters = pop[i][2:]
            model = Network(nfilters,sfilters)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            for epoch in range(epochs):
                print('EPOCH {}:'.format(epoch + 1))
                for images, labels in tqdm(train_loader):
                    optimizer.zero_grad()
                    output = model(images)
                    #print(output.shape,labels.shape)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in tqdm(test_loader):
                    output = model(images)
                    predictions = torch.argmax(output, dim=1)
                    correct += torch.sum((predictions == labels).float())
            acc = correct/total
            #acc = H.history['accuracy']
            pop_acc.append(acc*100)
        if max(pop_acc) > self.max_acc:
            self.max_acc = max(pop_acc)
            self.best_arch = pop[np.argmax(pop_acc)]
        self.gen_acc.append(max(pop_acc))
        return pop_acc
    
    def smooth_curve(self,factor,gen):
        smoothed_points = []
        for point in self.gen_acc:
            if smoothed_points:
                prev = smoothed_points[-1]
                smoothed_points.append(prev*factor + point * (1-factor))
            else:
                smoothed_points.append(point)
        plt.plot(range(gen+1),smoothed_points,'g',label='Smoothed training acc')
        plt.xticks(np.arange(gen+1))
        plt.legend()
        plt.title('Fitness Accuracy vs Generations')
        plt.xlabel('Generations')
        plt.ylabel('Fitness (%)')
        plt.show()
