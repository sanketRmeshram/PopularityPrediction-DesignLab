from torch.nn import Sigmoid
from torch.nn import Softmax
import torch 
from torch import nn
import torch.nn.functional as F
import util
# pytorch mlp for binary classification
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
'''
* performs element wise multiplication
@ performs matrix multication

linier(input_size,output_size)

model = nn.Linear(3, 2)
print(model.weight) size (2,3)
print(model.bias)  (1,2)
'''


class NFP(Module):
    def __init__(self, R, group_level_output):
        super().__init__()
        self.R = R # Radius
        # number of nodes in output group level perceptron
        self.group_level_output = group_level_output
        self.group_level_input = 14
        self.m = 10  # length of fingerprint
        self.t = 6   #  length of role distribution
        
        self.f = torch.full((1, self.m), 0).float()
        self.H = [ torch.full((self.t,self.m ),.2,requires_grad=True).float() for _ in range(self.R+1)]
        self.W = [torch.tensor(.2, requires_grad=True).float() for _ in range(self.R+1)]
        self.group_level_layer_weights = nn.Linear(self.group_level_input, self.group_level_output)
        self.merged_layer_weights = nn.Linear(self.m + self.group_level_output,3)
        self.Sigmoid = Sigmoid()
        self.Softmax = Softmax()



    def forward(self,X):
        
        x_member,graph,x_group = X

        # member level features NN
        r = {i:None  for i in graph} 
        for i in graph :
            r[i] = x_member[i]
            r[i] = torch.reshape(r[i],(1,self.t)).float()
#         print("r[0][0] : " , r[0][0])

        for L in range(0,self.R+1): 
#             print(L)
            for a in graph:
#                 print("     :",a)
                v1 = r[a] +sum([r[i] for i in graph[i]])
                
#                 print(v1,self.H[L])
#                 print("v1 :" ,v1.shape)
#                 print("H[L] : ",self.H[L].shape)
#                 print(v1 @ self.H[L])
#                 print(torch.matmul(v1 , self.H[L]))
                v2 = self.Sigmoid(v1 @ self.H[L])
#                 print("r[a]  : " ,r[a].shape)
                FL = self.Softmax(v2 * self.W[L])
#                 print("FL : " ,FL.shape)
                self.f = self.f+FL
        # return f
        # member level features NN

        # group level features NN
        group_perceptron_output = self.Sigmoid(self.group_level_layer_weights(x_group).double())
        group_perceptron_output = torch.reshape(group_perceptron_output,(1,self.group_level_output)).float()
#         print("group_perceptron_output : ",group_perceptron_output.shape)
#         print("self.f : " , self.f.shape)
        # group level features NN

        # merged features NN
        merged_input = torch.cat((self.f, group_perceptron_output),1)

        merged_output = self.Softmax(self.merged_layer_weights(merged_input))
        # merged features NN

        return merged_output
    
def split_train_test(data) :
    ind = int(.8 * len(data))
    train = data[:ind]
    test = data[ind:]
    return train,test

if __name__=="__main__":
    now = util.get_group_and_window()[:10]
    data = [ [(util.get_member_role_vectors(grp,window),util.get_graph(grp,window),util.get_group_level_featues(grp,window)),util.get_output(grp,window)] for grp,window in now]
    
    train,test = split_train_test(data)
    print("split done")
    model = NFP(6,7)
    loss_fn = nn.CrossEntropyLoss()
    for x,y in train :
        pred = model(x)
#         pred = torch.flatten(pred)
        print( "pred : ",pred,pred.shape)
        print("y : ",y,y.shape)
        target = []
        if y[0]==1:
            target.append(0)
        elif y[0]==1:
            target.append(1)
        else :
            target.append(2)
        target = torch.tensor(target).long()
        print("target : ",target)
#         print("pred , target : ",pred.view( -1),target.view( -1))
        loss = loss_fn(pred,target)
        print("loss  : ",loss)
        loss.backward(retain_graph=True)
        
        

    
    
    


