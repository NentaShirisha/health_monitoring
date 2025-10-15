from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import pymysql
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

global username
global X_train, X_test, y_train, y_test, X, Y, le, scaler
accuracy = []
precision = []
recall = [] 
fscore = []

#function to calculate all metrics
def calculateMetrics(algorithm, predict, y_test):
    a = (accuracy_score(y_test,predict)*100)
    p = (precision_score(y_test, predict,average='macro') * 100)
    r = (recall_score(y_test, predict,average='macro') * 100)
    f = (f1_score(y_test, predict,average='macro') * 100)
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm

dataset = pd.read_csv("Dataset/human_vital_signs_dataset_2024.csv")
labels = np.unique(dataset['Risk Category'].ravel())
le = LabelEncoder()
dataset['Risk Category'] = pd.Series(le.fit_transform(dataset['Risk Category'].astype(str)))#encode all str columns to numeric
le = LabelEncoder()
dataset['Gender'] = pd.Series(le.fit_transform(dataset['Gender'].astype(str)))#encode all str columns to numeric
Y = dataset['Risk Category'].ravel()
dataset.drop(['Risk Category'], axis = 1,inplace=True)
dataset.fillna(dataset.mean(), inplace = True)
scaler = StandardScaler()
X = dataset.values
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

svm_cls = svm.SVC()
svm_cls.fit(X_train, y_train)
predict = svm_cls.predict(X_test)
calculateMetrics("SVM", predict, y_test)

rf_cls = RandomForestClassifier()
rf_cls.fit(X_train, y_train)
predict = rf_cls.predict(X_test)
calculateMetrics("Random Forest", predict, y_test)
conf_matrix = confusion_matrix(predict, y_test)

mlp_cls = MLPClassifier()
mlp_cls.fit(X_train, y_train)
predict = mlp_cls.predict(X_test)
calculateMetrics("MLP Neural Network", predict, y_test)

xgb_cls = XGBClassifier()
xgb_cls.fit(X_train, y_train)
predict = xgb_cls.predict(X_test)
calculateMetrics("XGBoost", predict, y_test)

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})

def PredictAction(request):
    if request.method == 'POST':
        global rf_cls, scaler, le
        class_label = ['High Risk', 'Low Risk']
        myfile = request.FILES['t1'].read()
        if os.path.exists('HealthApp/static/test.csv'):
            os.remove('HealthApp/static/test.csv')
        with open('HealthApp/static/test.csv', "wb") as file:
            file.write(myfile)
        file.close()
        testData = pd.read_csv('HealthApp/static/test.csv')#reading test data
        data = testData.values
        testData['Gender'] = pd.Series(le.transform(testData['Gender'].astype(str)))#encode all str columns to numeric
        testData.fillna(dataset.mean(), inplace = True)
        testData = testData.values
        testData = scaler.transform(testData)
        predict = rf_cls.predict(testData)
        color = ['green', 'red']
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Test Data</th><th><font size="3" color="black">Predicted Health Status</th></tr>'
        for i in range(len(predict)):
            output += '<td><font size="3" color="black">'+str(data[i])+'</td><td><font size="3" color="'+color[predict[i]]+'">'+class_label[predict[i]]+'</font></td></tr>'
        output+= "</table></br></br></br></br>"
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def TrainModel(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, y_pred, X_test
        global accuracy, precision, recall, fscore, conf_matrix
        class_label = ['High Risk', 'Low Risk']
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Algorithm Name</th><th><font size="3" color="black">Accuracy</th>'
        output += '<th><font size="3" color="black">Precision</th><th><font size="3" color="black">Recall</th><th><font size="3" color="black">FSCORE</th></tr>'
        algorithms = ['SVM', 'Random Forest', 'MLP Neural Network', 'XGBoost']
        for i in range(len(algorithms)):
            output += '<td><font size="3" color="black">'+algorithms[i]+'</td><td><font size="3" color="black">'+str(accuracy[i])+'</td><td><font size="3" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(recall[i])+'</td><td><font size="3" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Random Forest Confusion Matrix Graph")
        axis[1].set_title("All Algorithms Performance Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = class_label, yticklabels = class_label, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(class_label)])    
        df = pd.DataFrame([['SVM','Accuracy',accuracy[0]],['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','FSCORE',fscore[0]],
                           ['Random Forest','Accuracy',accuracy[1]],['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','FSCORE',fscore[1]],
                           ['MLP Neural Network','Accuracy',accuracy[2]],['MLP Neural Network','Precision',precision[2]],['MLP Neural Network','Recall',recall[2]],['MLP Neural Network','FSCORE',fscore[2]],
                           ['XGBoost','Accuracy',accuracy[3]],['XGBoost','Precision',precision[3]],['XGBoost','Recall',recall[3]],['XGBoost','FSCORE',fscore[3]],
                          ],columns=['Parameters','Algorithms','Value'])
        df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=axis[1])  
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):    
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, X, Y
        class_label = ['High Risk', 'Low Risk']
        output = '<font size="3" color="black">Healthcare Monitoring Dataset Loaded</font><br/>'
        output += '<font size="3" color="blue">Total records found in Dataset = '+str(X.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">Different Class Labels found in Dataset = '+str(class_label)+'</font><br/><br/>'
        output += '<font size="3" color="black">Dataset Train & Test Split details</font><br/>'
        output += '<font size="3" color="blue">80% dataset records used to train Isoloation Forest = '+str(X_train.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">20% dataset records used to test Isoloation Forest = '+str(X_test.shape[0])+'</font><br/><br/>'
        dataset = pd.read_csv("Dataset/human_vital_signs_dataset_2024.csv",nrows=100)
        columns = dataset.columns
        dataset = dataset.values
        output+='<table border=1 align=center width=100%><tr>'
        for i in range(len(columns)):
            output+='<th><font size="3" color="black">'+columns[i]+'</th>'
        output += '</tr>'
        for i in range(len(dataset)):
            output += '<tr>'
            for j in range(len(dataset[i])):
                output += '<td><font size="3" color="black">'+str(dataset[i,j])+'</td>'
            output += '</tr>'    
        output+= "</table></br></br></br></br></br>"        
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)

        output = "none"

        # First check if username already exists
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='health', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT username FROM register WHERE username = %s", (username,))
            row = cur.fetchone()
            if row:
                output = username + " Username already exists"

        # If no existing username, proceed to insert
        if output == "none":
            try:
                db_connection = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='health', charset='utf8')
                db_cursor = db_connection.cursor()

                # Use parameterized query
                student_sql_query = """
                INSERT INTO register(username, password, contact, email, address)
                VALUES (%s, %s, %s, %s, %s)
                """
                db_cursor.execute(student_sql_query, (username, password, contact, email, address))
                db_connection.commit()

                if db_cursor.rowcount == 1:
                    output = "Signup process completed. Login to perform Live Health Monitoring System"
                else:
                    output = "Signup failed. Please try again."
            except Exception as e:
                output = f"An error occurred: {str(e)}"
            finally:
                db_connection.close()

        context = {'data': output}
        return render(request, 'Register.html', context)
   

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username, email_id
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'health',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password,email FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    email_id = row[2]
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})
