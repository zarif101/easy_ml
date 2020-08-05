import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import sys

def logistic_regression_algo(X_train,X_test,y_train,y_test):
    ##print(y_train)
    model = LogisticRegression(max_iter=20000)
    ##print('made')
    ##print(y_train)
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    ##print('rerer')
    report = classification_report(y_test,preds)
    matrix = confusion_matrix(y_test,preds)
    st.write('Confusion matrix: ',matrix)
    st.write('Classification report: ',report)
    return model

def random_forest_algo(X_train,X_test,y_train,y_test):
    ##print('forest time baby')
    rfc = RandomForestClassifier(n_estimators=100)
    sys.stdout(y_train)
    rfc.fit(X_train,y_train)
    preds = rfc.predict(X_test)
    type = st.selectbox('Do you want to use your own custom n_estimators/trees values or use the one that we think is best?',['Default','Custom'])
    if type == 'Default':
        report = classification_report(y_test,preds)
        matrix = confusion_matrix(y_test,preds)
        st.write('Confusion matrix: ',matrix)
        st.write('Classification report: ',report)
        return rfc
    elif type == 'Custom':
        trees = st.text_input('Enter as an integer the number of neighbors/estimators you want')
        trees = int(trees)
        rfc = RandomForestClassifier(n_estimators=trees)
        rfc.fit(X_train,y_train)
        preds = rfc.predict(X_test)
        report = classification_report(y_test,preds)
        matrix = confusion_matrix(y_test,preds)
        st.write('Confusion matrix: ',matrix)
        st.write('Classification report: ',report)
        return rfc



def svm_algo(X_train,X_test,y_train,y_test):
    ##print('asdsq')
    model = SVC()
    param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
    ##print('keqwll')
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
    grid.fit(X_train,y_train)
    model = grid.best_estimator_
    ##print('gooo')
    type = st.selectbox('Do you want to input your own C and gamma values, or use the ones that we think are best?',['Default','Custom'])
    if type == 'Default':
        preds = model.predict(X_test)
        report = classification_report(y_test,preds)
        matrix = confusion_matrix(y_test,preds)
        st.write('Confusion matrix: ',matrix)
        st.write('Classification report: ',report)
        return model
    elif type == 'Custom':
        params = st.text_input('Enter the C value and then gamma value separated by commas, with no spaces')
        params = list(params)
        ##print(params)
        c = int(params[0])
        gamma = int(params[2])
        ##print(c)
        ##print(gamma)
        model = SVC(C=c,gamma=gamma)
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        report = classification_report(y_test,preds)
        matrix = confusion_matrix(y_test,preds)
        st.write('Confusion matrix: ',matrix)
        st.write('Classification report: ',report)
        return model



def knn_algo(X_train,X_test,y_train,y_test):
    error_rate = []
    neighbors = []
    ##print('asdasdeq')
    for i in range(1,40):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        ##print('predicted')
        error_rate.append(np.mean(preds != y_test))
        neighbors.append(i)
    junk = []
    for i in range(1,40):
        junk.append(i)
        ##print('junked')
    type = st.selectbox('Do you want to input your own neighbors values, or use the ones that we think are best?',['Default','Custom'])
    if type == 'Default':
        default_model = KNeighborsClassifier(n_neighbors=neighbors[error_rate.index(min(error_rate))])
        default_model.fit(X_train,y_train)
        default_preds = default_model.predict(X_test)
        default_preds = model.predict(X_test)
        default_classifcation_report = classification_report(y_test,default_preds)
        default_confusion_matrix = confusion_matrix(y_test,default_preds)
        ##print(neighbors[error_rate.index(min(error_rate))])
        st.write('Classification report with {0} neighbors(lowest we could find): '.format(neighbors[error_rate.index(min(error_rate))]),default_classifcation_report)
        st.write('Confusion matrix with {0} neighbors(lowest we could find): '.format(neighbors[error_rate.index(min(error_rate))]),default_confusion_matrix)

        ax = sns.lineplot(x=junk,y=error_rate)
        ax.set(xlabel='Number of Neighbors',ylabel='Error_rate')
        st.pyplot()
        return default_model

    elif type == 'Custom':
        neighbor = st.text_input('Enter the number of neighbors you want, as an integer. Keep in mind error_rate is on the test set, and above graph may change')
        ##print(int(neighbor))
        if neighbor is not None:
            neighbor = int(neighbor)
            model = KNeighborsClassifier(n_neighbors=neighbor)
            model.fit(X_train,y_train)
            preds = model.predict(X_test)
            report = classification_report(y_test,preds)
            matrix = confusion_matrix(y_test,preds)
            st.write('Confusion matrix: ',matrix)
            st.write('Classification report: ',report)
            return model
        else:
            pass

def linear_regression_algo(X_train,X_test,y_train,y_test):
    model = LinearRegression()
    model.fit(X_train,y_train)
    preds = model.predict(X_test)
    MAE = metrics.mean_absolute_error(y_test, preds)
    MSE = metrics.mean_squared_error(y_test,preds)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test,preds))
    st.write('Mean Absolute Error',MAE)
    st.write('Mean Square Error',MSE)
    st.write('Root Mean Square Error',RMSE)
    return model
