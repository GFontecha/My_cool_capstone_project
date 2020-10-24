#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


def LRfit(vals_LRfit):
    X_train=vals_LRfit[0]
    y_train=vals_LRfit[1]
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    return LR


# In[ ]:


def LRpre(vals_LRpre):
    LR=vals_LRpre[0]
    X_test=vals_LRpre[1]
    yhatLR = LR.predict(X_test)
    return yhatLR
    

