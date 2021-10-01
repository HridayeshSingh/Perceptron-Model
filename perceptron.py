import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
THRESHOLD=0.2

class PERCEPTRON:
    def __init__(self, features, predictor):
        self.features=features
        self.predictor=predictor
    
    def Scaling(self, features):
        scaled_features=StandardScaler().fit_transform(features)
        #return 
        print(np.c_[scaled_features, np.ones([scaled_features.shape[0],1])])
    
    def Activation(self, hypothesis):
        activ=(1/(1+np.exp(-(hypothesis))))
        #print(activ)
        return 1 if activ>=THRESHOLD else 0
        
    def Hypothesis (self, scaled_features,weights):
        hypo=np.dot(scaled_features,weights)
        output=self.Activation(hypo)
        return output
      
perc=PERCEPTRON(data_thin,predictor_target)
final=perc.Fit(5000,0.24)
