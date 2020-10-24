#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import svm


# In[ ]:


def SVMfit(vals_SVMfit):
    X_train=vals_SVMfit[0]
    y_train=vals_SVMfit[1]
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    return clf


# In[ ]:


def SVMpre(vals_SVMpre):
    clf=vals_SVMpre[0]
    X_test=vals_SVMpre[1]
    yhatSVM = clf.predict(X_test)
    return yhatSVM


# In[ ]:


def SVMfitPoly(vals_SVMfit):
    X_train=vals_SVMfit[0]
    y_train=vals_SVMfit[1]
    clf = svm.SVC(kernel='poly')
    clf.fit(X_train, y_train)
    return clf

