class Perceptron:
    def __init__(self,x1,x2,t,w1,w2,bias,a,theta=0):

        self.w1 = w1
        self.w2 = w2
        self.x1 = x1
        self.x2 = x2
        self.t = t
        self.b = bias
        self.alpha = a
        self.theta = theta

    def binary(self, z):
        if(z>=self.theta):
            return 1
        else:
            return 0
    
    def bipolar(self, z):
        if(z>=self.theta):
            return 1
        else:
            return -1
        
    def netInput(self,i):
        yin = 0
        yin = self.b + (self.x1[i]*self.w1 + self.x2[i]*self.w2 )
        return yin

    def updateWeights(self,i):

        self.w1 = self.w1 + ( self.alpha*self.x1[i]*self.t[i] )
        self.w2 = self.w2 + ( self.alpha*self.x2[i]*self.t[i] )
        self.b = self.b + ( self.alpha*self.t[i])
        return "Weight Updated as "+str(self.w1)+" , "+str(self.w2)+" , bias - "+str(self.b)
    
    def predict(self,test1,test2,fn):
        yin = self.b + ( test1*self.w1 + test2*self.w2)
        if(fn == "binary"):
            yout = self.binary(yin)
        else:
            yout = self.bipolar(yin)
        
        print("For given test data : "+str(test1)+" , "+str(test2)+" Predicted Result = "+str(yout))
        print("Weights are "+str(self.w1)+" , "+str(self.w2))

    def train(self):
        
        for epoch in range(1000):
            cnt = 0 #init counter for this epoch to verify convergence
            
            for i in range(len(self.x1)):
                z = self.netInput(i)#calc net input
                a = self.bipolar(z)#activated net input
                
                if(a!=self.t[i]):#if predicted out and actual out is not same, update weights
                    self.updateWeights(i)#update weights
                else:#if no weight update increment counter
                    cnt+=1
                    
            if(cnt==len(self.x1)):#no weight update for entire samples in this epoch, it means model is converged
                break
                
        print(self.w1, self.w2, self.b, epoch+1)
        
        print("------------ Training Completed -----------------")

x1 = [-1,1,-1,1]
x2 = [-1,-1,1,1]
t = [-1,-1,-1,1]

p = Perceptron(x1,x2,t,0,0,0,1,0)
p.train()
p.predict(0,1,"binary")