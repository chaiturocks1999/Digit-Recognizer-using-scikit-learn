#A simple python code for digit recognizer using scikit learn library

# Importing the libraries
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

df = pd.read_csv("train.csv")
x= df.iloc[:,1:].values
y = df.iloc[:,0].values

# Splitting the dataset into the Training set and Test set for checking accuracy
from sklearn.model_selection import train_test_split
x_train , x_test ,y_train , y_test = train_test_split(x,y,random_state=0,test_size=0.1)


# Fitting the Decision Tree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
rg = DecisionTreeRegressor()
rg.fit(x_train,y_train)

# Predicting a new result
ypred = rg.predict(x_test)
ypred = np.array(ypred)    
y_test=np.array(y_test)
cnt=0                                #this count variable will count no of correct predictions 
for i in range(4200):
    if(ypred[i]==y_test[i]):
        cnt = cnt +1
    else:
        continue
acc = 100.0*cnt/4200                                        # there are 4200 records in the y_test
print("the accuracy is " + str(acc))                        #this is the accuracy of our model 

test = pd.read_csv("test.csv").as_matrix()                    #now lets import the test dataset asked in the question
ans = rg.predict(test)                                        #predict the results using our model

np.savetxt(fname="sub.txt",delimiter=',',fmt="%d", X=ans)     #now save your results in a .txt file
ans.shape

# Visualising the results
d  = test[1067]                          # I took a sample record 
d.shape = (28,28)                       # I converted it into a 28*28 matrix         
print("the model predicted : " + str(rg.predict([test[1067]]).astype(int)))  #printing the digit our model predicted 
plt.imshow(255-d,cmap ='gray')  # I have drawn an image to visualise 