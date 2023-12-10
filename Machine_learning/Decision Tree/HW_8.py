import pandas as pd
from sklearn.model_selection import KFold,cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:/Users/naduk\Downloads/UsedCars.csv")
#print(data.head())
# describing the data
#print(data.describe())
data.dropna()
y_dummies=pd.get_dummies(data,prefix=['trim','color','fuel','region','soundSystem','wheelType','IsOneOwner'],drop_first=True)
#print(y_dummies)
y=y_dummies['price']
x=y_dummies.drop(columns=['price'])
print(x)
kf=KFold(n_splits=5,shuffle=True,random_state=42)
mean_square=[]
depth=range(2,101)
for i in depth:
    model= DecisionTreeRegressor(max_depth=i)
    squared_error=cross_val_score(model,x,y,cv=kf,scoring='neg_mean_squared_error')
    mse=-np.mean(squared_error)
    mean_square.append(mse)
plt.scatter(depth, mean_square, marker='o')
plt.xlabel('depth')
plt.ylabel('Mean Square error')
plt.title('depth vs. mean square error')
plt.grid(True)
plt.show()

best_mse=float('inf')
best_depth=0
for mse,max_depth in zip(mean_square,depth):
    if mse<best_mse:
        best_mse=mse
        best_depth=max_depth
print(f'the split at which mean square error mse:{best_mse}, depth: {best_depth}')