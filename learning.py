#!/usr/bin/env python
# coding: utf-8

# In[351]:


import numpy as np
import pandas as pd
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[352]:


def load_data():
    data = pd.read_excel(r'shit.xlsx')
#     print(data.head())
#     data = data.sample(frac=1).reset_index(drop=True)
    X = data.drop(['Result'], axis=1)
    y = data['Result']
#     print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5, shuffle=True)
    
    return X_train, X_test, y_train, y_test
    


# In[353]:


def knn_build():
    knn = KNeighborsClassifier(n_neighbors=5)
    
    return knn


# In[354]:


def knn_train(model, x_train, y_train):
    model.fit(x_train, y_train)
    
    return model


# In[355]:


def knn_evaluate(model, x_test, y_test):
    score = model.score(x_test, y_test)
    
    return score


# In[356]:


def knn_predict(model, x_pred):
    y_pred = model.predict(x_pred)
    
    return y_pred


# In[357]:

'''
def mlp_build():
    mlp = Sequential()
    mlp.add(Dense(32, input_dim=2, kernel_initializer='random_uniform', activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(16, kernel_initializer='uniform', activation='relu'))
    mlp.add(Dropout(0.2))
    mlp.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     mlp.add(Dense(1, kernel_initializer='uniform', activation='softmax'))
    
#     mlp.summary()
    
    mlp.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    return mlp
    


# In[358]:


def mlp_train(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=50, verbose=0, validation_split=0.2)
    
    return model


# In[359]:


def mlp_evaluate(model, x_test, y_test): 
    score = model.evaluate(x_test, y_test, batch_size=32)
    
    return score


# In[360]:


def mlp_predict(model, x_pred):
    y_pred = model.predict(x_pred)
    
    return y_pred

'''
# In[361]:


def learn():
    #######  Load Data  ########
    X_train, X_test, y_train, y_test = load_data()
    
    
    #######  MLP Binary classifier   ########
#     mlp = mlp_build()
#     mlp = mlp_train(mlp, X_train, y_train)
#     score = mlp_evaluate(mlp, X_test, y_test)
#     print("Accuracy ", score)
    
    
    #######  KNN classifier   ########
    knn = knn_build()
    knn = knn_train(knn, X_train, y_train)
    # score = knn_evaluate(knn, X_test, y_test)
    # print("Accuracy ", score)
    
    '''
    #######  TESTING   ########
    d = 37.0
    temp = {'Distance': [d, d, d, d, d], 'Delay': [1, 2, 3, 4, 5]}
    x1 = pd.DataFrame(data=temp)
    y_pred = knn_predict(knn, x1)
    print(x1.dtypes)
    print("Prediction successful!")
    print(y_pred)
    '''

    return knn

# In[362]:


# if __name__ == "__main__":
#     main()
    


# In[ ]:




