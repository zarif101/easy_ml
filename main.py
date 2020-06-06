import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from algorithms import logistic_regression_algo,random_forest_algo,svm_algo,knn_algo,linear_regression_algo

st.title('Welcome')
st.write('Select your file type and then upload!')

def main():
    try:
        thedata,data_type = upload_data()
        if data_type == 'Columns/rows':
            X_col,y_col = get_correct_columns(thedata,type)
            if len(X_col) > 1:
                X_train,X_test,y_train,y_test = get_data_arrays(thedata,X_col,y_col)
                #try_algos(X_train,X_test,y_train,y_test)
                algo_type = st.sidebar.selectbox('Are you doing classification or regression?', ['Classification','Regression'])
                if algo_type == 'Classification':
                    algo_list = ['Logistic Regresssion','Random Forest','Support Vector Machine','K Nearest Neighbors']
                    algo = st.selectbox('Pick an algorithm to try', algo_list)
                    if algo == algo_list[0]:
                        logistic_regression_algo(X_train,X_test,y_train,y_test)
                    elif algo == algo_list[1]:
                        random_forest_algo(X_train,X_test,y_train,y_test)
                    elif algo == algo_list[2]:
                        svm_algo(X_train,X_test,y_train,y_test)
                    elif algo == algo_list[3]:
                        knn_algo(X_train,X_test,y_train,y_test)
                elif algo_type == 'Regression':
                    algo_list = ['Linear Regression']
                    algo = st.selectbox('Pick and algorithm to try',algo_list)
                    if algo == algo_list[0]:
                        linear_regression_algo(X_train,X_test,y_train,y_test)
            else:
                pass
    except:
        pass

def upload_data():
    file_type = st.sidebar.selectbox('What type of data will you upload?', ['Image','Columns/rows'])

    if file_type=='Columns/rows':
        data = st.file_uploader('Upload file(s) here',type='csv')
        if data is None:
            return
        data = pd.read_csv(data)
        st.write(data)
    elif file_type == 'Image':
        data = st.file_uploader('Upload image(s) here',type=['png','jpg','jpeg'])
        if data is None:
            return
        st.image(data)
    if data is not None:
        return data,file_type


def get_correct_columns(thedata,type):
    df = thedata
    columns = list(df.columns)
    X_cols = st.multiselect('Which columns are the features? (Currently only supporting numerical features)',columns)
    y_cols = st.multiselect('Which column contains the target/labels? (Numerical OR categorical) ONLY SELECT ONE!',columns)
    # y_cols = []
    # y_cols.append(y_colss)
    return list(X_cols), y_cols

def get_data_arrays(thedata,X_col,y_col):
    print(X_col)
    X = thedata[X_col]
    y = thedata[y_col]
    if isinstance(y.values[1][0], str):
        dummy_y = pd.get_dummies(thedata[y_col])
        y_real = []
        print(dummy_y)
        for each in dummy_y.iloc[:,0]:
            if each == 0:
                y_real.append(1)
            elif each == 1:
                y_real.append(0)

        y = y_real
        X = np.array(X)
    else:
        y = y
        X = np.array(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    return X_train,X_test,y_train,y_test

# def logistic_regression_algo(X_train,X_test,y_train,y_test):
#     print(y_train)
#     model = LogisticRegression(max_iter=20000)
#     print('made')
#     print(y_train)
#     model.fit(X_train,y_train)
#     preds = model.predict(X_test)
#     print('rerer')
#     report = classification_report(y_test,preds)
#     matrix = confusion_matrix(y_test,preds)
#     st.write('Confusion matrix: ',matrix)
#     st.write('Classification report: ',report)

main()
