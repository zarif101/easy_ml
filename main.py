import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from algorithms import logistic_regression_algo,random_forest_algo,svm_algo,knn_algo,linear_regression_algo
from send_files import Email
import email_info
import pickle
st.title('Welcome')
st.write('Select your file type and then upload! (Currently only accepting Excel or CSV)')

def main():
    try:
        model = None
        thedata,data_type = upload_data()
        if data_type == 'Excel' or 'CSV':
            X_col,y_col = get_correct_columns(thedata,type)
            if len(X_col) > 1:
                X_train,X_test,y_train,y_test = get_data_arrays(thedata,X_col,y_col)
                #print(y_train)
                #try_algos(X_train,X_test,y_train,y_test)
                algo_type = st.sidebar.selectbox('Are you doing classification or regression?', ['Classification','Regression'])
                save_choice_box = st.empty()
                if algo_type == 'Classification':
                    number_unique = pd.DataFrame(y_train).nunique()[0]
                    #print(pd.DataFrame(y_train).nunique().unique())
                    if number_unique < 3:
                        algo_list = ['Logistic Regresssion','Random Forest','Support Vector Machine','K Nearest Neighbors']
                        algo = st.selectbox('Pick an algorithm to try', algo_list)
                        if algo == algo_list[0]:
                            model = logistic_regression_algo(X_train,X_test,y_train,y_test)
                            #save_choice = save_choice_box.selectbox('Do you want to save your model?',['No','Yes'])
                        elif algo == algo_list[1]:
                            model = random_forest_algo(X_train,X_test,y_train,y_test)
                            #save_choice = save_choice_box.selectbox('Do you want to save your model?',['No','Yes'])
                        elif algo == algo_list[2]:
                            model = svm_algo(X_train,X_test,y_train,y_test)
                            #save_choice = save_choice_box.selectbox('Do you want to save your model?',['No','Yes'])
                        elif algo == algo_list[3]:
                            model = knn_algo(X_train,X_test,y_train,y_test)
                            #save_choice = save_choice_box.selectbox('Do you want to save your model?',['No','Yes'])
                    else:
                        #print('GOGOGO')
                        algo_list = ['Random Forest','Support Vector Machine','K Nearest Neighbors']
                        algo = st.selectbox('Pick an algorithm to try', algo_list)
                        if algo == algo_list[0]:
                            model = random_forest_algo(X_train,X_test,y_train,y_test)
                            #save_choice = save_choice_box.selectbox('Do you want to save your model?',['No','Yes'])
                        elif algo == algo_list[1]:
                            model = svm_algo(X_train,X_test,y_train,y_test)
                            #save_choice = save_choice_box.selectbox('Do you want to save your model?',['No','Yes'])
                        elif algo == algo_list[2]:
                            model = knn_algo(X_train,X_test,y_train,y_test)
                            #save_choice = save_choice_box.selectbox('Do you want to save your model?',['No','Yes'])

                elif algo_type == 'Regression':
                    algo_list = ['Linear Regression']
                    algo = st.selectbox('Pick and algorithm to try',algo_list)
                    if algo == algo_list[0]:
                        model = linear_regression_algo(X_train,X_test,y_train,y_test)
                save_choice_box = st.empty()
                #save_choice = save_choice_box.selectbox('Do you want to save your model?',['No','Yes'])

                if model is not None:
                    save_choice_box = st.empty()
                    save_choice = st.selectbox('Do you want to save your model?',['No','Yes'])
                    if save_choice == 'Yes':
                        receiver_email = st.text_input('Type in your email').rstrip().lstrip()
                        if '@' in receiver_email:
                            if st.button('Email me my model!'):
                                sender_email = email_info.get_sender_email()
                                sender_password = email_info.get_sender_password()
                                subject = 'Your Model!'
                                message = 'Your model is attatched as a pickle file...'
                                model_path = 'user_models/'
                                model_name = 'send_model.pkl'
                                with open(model_path+model_name,'wb') as pick:

                                    pickle.dump(model,pick)


                                email = Email(sender_email,receiver_email,subject,message,model_name,model_path,sender_password)

                                email.send_email()

                                st.write('Your model has been emailed to you!')
                            else:
                                pass
            else:
                pass
    except:
        pass


def upload_data():
    file_type = st.sidebar.selectbox('What type of data will you upload?', ['Excel','CSV'])

    if file_type=='CSV':
        data = st.file_uploader('Upload file(s) here',type='csv')
        if data is None:
            return
        data = pd.read_csv(data)
        st.write(data)
    elif file_type == 'Excel':
        data = st.file_uploader('Upload file(s) here',type='xlsx')
        if data is None:
            return
        data = pd.read_excel(data)
    # elif file_type == 'Image':
    #     data = st.file_uploader('Upload image(s) here',type=['png','jpg','jpeg'])
    #     if data is None:
    #         return
    #     st.image(data)
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
    X = thedata[X_col]
    y = thedata[y_col]
    try:
        if isinstance(y.values[0][0], str):
            y_real = []
            y_unique = pd.unique(y['Species'])
            for val in y['Species'].values:
                for index,target in enumerate(y_unique):
                    if val == target:
                        y_real.append(index)
            y = y_real
            X = np.array(X)
            #y_real = []
            # for each in dummy_y.iloc[:,0]:
            #     if each == 0:
            #         y_real.append(1)
            #     elif each == 1:
            #         y_real.append(0)

        else:
            y = y
            X = np.array(X)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
        return X_train,X_test,y_train,y_test
    except Exception as e:
        pass
        #print(e)

# def logistic_regression_algo(X_train,X_test,y_train,y_test):
#     #print(y_train)
#     model = LogisticRegression(max_iter=20000)
#     #print('made')
#     #print(y_train)
#     model.fit(X_train,y_train)
#     preds = model.predict(X_test)
#     #print('rerer')
#     report = classification_report(y_test,preds)
#     matrix = confusion_matrix(y_test,preds)
#     st.write('Confusion matrix: ',matrix)
#     st.write('Classification report: ',report)


main()
