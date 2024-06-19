import numpy as np
import json
import os
from sklearn.linear_model import LinearRegression
from random import randint
from sklearn import preprocessing
import matplotlib.pyplot as plt
f=open("./Json.json","r")
JSON=json.load(f)
workbook = open("./Drug.json","r")
ALLDrugs = json.load(workbook)
IDs={}
# thispagestyle{empty}
TrainX=[]
TrainY=[]
for X,Y in ALLDrugs.items():
    IDs[X]=[np.zeros(3) for i in range(8)]
    for Year in range(2010,2018):
        if(str(Year) in Y):
            IDs[X][int(Year)-2010] = np.array([(Y[str(Year)]["H"]),(Y[str(Year)]["F"]),(Y[str(Year)]["O"])])

DIS={}
for dx in IDs:
    for dy in IDs:
        if(dx!=dy):
            DIS[str(dx)+str(dy)]=np.power(np.array(JSON[str(dx)])-np.array(JSON[str(dy)]),2).sum()
            for X,Y in IDs.items():
                for years in range(2,8):
                    SUM=0.0
                    for dx in IDs:
                        if(dx!=X):
                            SUM+=(IDs[dx][years-1])/DIS[str(dx)+str(X)]
                            TrainX.append(np.array([Y[years-1],Y[years-2],SUM]))
                            TrainY.append(Y[years])
                            TrainX=np.array(TrainX).reshape(-1,9)
                            TrainY=np.array(TrainY).reshape(-1,3)

model = LinearRegression()
model.fit(TrainX, TrainY)
AANS={}
# PC=0.7 # for X in IDs: # IDs[X][6]=IDs[X][6]*PC # IDs[X][7]=IDs[X][7]*PC
#
for tX in IDs:
    TrainX=[]
    TrainY=[]
    tY=IDs[tX]
    STX=tY

TrainY=np.array(tY).reshape(-1,3)

for years in range(8,10):
    SUM=0.0
    for dx in IDs:
        if(dx!=tX):
            SUM+=(STX[years-1])/DIS[str(dx)+str(tX)]
            TrainX=np.array([STX[years-1],STX[years-2],SUM]).reshape(-1,9)
            TX=model.predict(TrainX).flatten()
            STX.append(TX)
            STX=np.array(STX)
            tmp=STX[-2:]
            AANS[tX]=np.where(tmp<0,0,tmp).tolist()