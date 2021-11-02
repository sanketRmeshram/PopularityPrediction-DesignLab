from torch.nn import Sigmoid
from torch.nn import Softmax
import torch 
from torch import nn


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
        super(NFP,self).__init()
        self.R = R # Radius
        # number of nodes in output group level perceptron
        self.group_level_output = group_level_output
        self.group_level_input = 14
        self.m = 10  # length of fingerprint
        self.t = 6   #  length of role distribution
        
        self.f = torch.tensor((1, self.m), 0)
        self.H = [ nn.linear(self.t,self.m ) for _ in range(self.R+1)]
        self.W = [torch.tensor(.2, requires_grad=True) for _ in range(self.R+1)]
        self.group_level_layer_weights = nn.linear(self.group_level_input, self.group_level_output)
        self.merged_layer_weights = nn.linear(self.m + self.group_level_output,3)



    def forward(self,X):
        
        x_member,graph,x_group = X

        # member level features NN
        r = [None for i in range(len(graph)) ]
        for i in range(len(graph)) :
            r[i] = torch.tensor(x_member[i])
            r[i] = torch.reshape(r[i],(1,self.t))
        print("r[0][0] : " , r[0][0])

        for L in range(0,self.R+1): 
            for a in range(len(graph)):
                v1 = r[a] +sum([r[i] for i in graph[i]])
                r[a] = Sigmoid(v1 @ H[L])
                FL = Softmax(r[a] * W[L])
                self.f = self.f+FL
        # return f
        # member level features NN

        # group level features NN
        group_perceptron_output = x_group @ self.group_level_layer_weights
        # group level features NN

        # merged features NN
        merged_input = torch.cat((f, group_perceptron_output),1)

        merged_output = merged_input @ self.merged_layer_weights
        # merged features NN

        return merged_output
    
    


