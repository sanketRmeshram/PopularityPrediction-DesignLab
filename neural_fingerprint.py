from torch.nn import Sigmoid
from torch.nn import Softmax

'''
* performs element wise multiplication
@ performs matrix multication

linier(input_size,output_size)

model = nn.Linear(3, 2)
print(model.weight) size (2,3)
print(model.bias)  (1,2)
'''

class NFP():
    def __init__(self,R) :
        super(NFP,self).__init()
        self.R = R
        self.m = 10
        
    def forward(self,X,graph):
        
        for L in range(0,R+1): 
            fL = torch.tensor((1,m),0)
            for a in range(len(graph)):
                v1 = r[a] +sum([r[i] for i in graph[i]])
                v2 = Sigmoid(v1.t() @ H[L])
                FL = FL + Softmax()
                
            f = f+FL
        return f
        