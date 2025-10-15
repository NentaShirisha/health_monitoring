from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.conf import settings
import os
import random
from random import randint, uniform
import io
import base64
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import seaborn as sns
import pymysql
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import matplotlib.pyplot as plt
import io, base64
import numpy as np

accuracy = []
precision = []
recall = [] 
fscore = []
class_label = ['High Risk', 'Low Risk']

# Load and preprocess dataset
dataset = pd.read_csv("Dataset/human_vital_signs_dataset_2024.csv")
dataset['Risk Category'] = LabelEncoder().fit_transform(dataset['Risk Category'].astype(str))
dataset['Gender'] = LabelEncoder().fit_transform(dataset['Gender'].astype(str))
Y = dataset['Risk Category'].values
dataset.drop(['Risk Category'], axis=1, inplace=True)
dataset.fillna(dataset.mean(), inplace=True)
scaler = StandardScaler()
X = scaler.fit_transform(dataset.values)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train models with improved parameters
rf_cls = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
rf_cls.fit(X_train, y_train)

svm_cls = svm.SVC(C=10, gamma=0.01, kernel='rbf', probability=True)
svm_cls.fit(X_train, y_train)

xgb_cls = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
xgb_cls.fit(X_train, y_train)

models = {
    "Random Forest": rf_cls,
    "SVM": svm_cls,
    "XGBoost": xgb_cls
}

# Smartwatch API fallback generator
def fetch_smartwatch_data():
    return {
        'heart_rate': randint(60, 100),
        'temperature': round(uniform(36.0, 38.0), 1),
        'spo2': randint(93, 100),
        'bp_sys': randint(110, 140),
        'bp_dia': randint(70, 90),
        'steps': randint(1000, 8000)
    }

def fetch_smartwatch_data():
    try:
        response = requests.get("https://api.mocksmartwatch.com/user/vitals")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return fetch_smartwatch_data()

# Views

def index(request):
    return render(request, 'index.html')

def UserLogin(request):
    return render(request, 'UserLogin.html')

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        users = request.POST.get('t1')
        password = request.POST.get('t2')
        status = "none"
        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='health', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT username, password, email FROM register")
            for row in cur.fetchall():
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            return render(request, "UserScreen.html", {
                'data': 'Welcome ' + username,
                'watch_data': fetch_smartwatch_data()
            })
        else:
            return render(request, 'UserLogin.html', {'data': 'Invalid username'})
    return render(request, 'UserLogin.html')

def UserScreen(request):
    return render(request, 'UserScreen.html')

def Register(request):
    return render(request, 'Register.html')

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1')
        password = request.POST.get('t2')
        contact = request.POST.get('t3')
        email = request.POST.get('t4')
        address = request.POST.get('t5')

        con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='health', charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("SELECT username FROM register WHERE username = %s", (username,))
            if cur.fetchone():
                return render(request, 'Register.html', {'data': f"{username} Username already exists"})

        try:
            con = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', database='health', charset='utf8')
            cur = con.cursor()
            cur.execute("INSERT INTO register(username, password, contact, email, address) VALUES (%s, %s, %s, %s, %s)",
                        (username, password, contact, email, address))
            con.commit()
            return render(request, 'Register.html', {'data': "Signup process completed. Login to perform Live Health Monitoring System"})
        except Exception as e:
            return render(request, 'Register.html', {'data': f"An error occurred: {str(e)}"})
        finally:
            con.close()

def LoadDataset(request):
    if request.method == 'GET':
        sample = pd.read_csv("Dataset/human_vital_signs_dataset_2024.csv", nrows=100)

        output = f"""
        <div class="container mt-4">
            <div class="alert alert-info" role="alert">
                <h5 class="mb-0">✅ Healthcare Monitoring Dataset Loaded</h5>
            </div>
            <p><span class='badge bg-primary'>Total Records: {X.shape[0]}</span>
            <span class='badge bg-success ms-2'>Classes: {', '.join(class_label)}</span>
            <span class='badge bg-warning text-dark ms-2'>Train Samples: {X_train.shape[0]}</span>
            <span class='badge bg-danger ms-2'>Test Samples: {X_test.shape[0]}</span></p>
            <div class="table-responsive">
                <table class="table table-bordered table-striped table-hover mt-3">
                    <thead class="table-dark">
        """

        for col in sample.columns:
            output += f"<th>{col}</th>"
        output += "</thead><tbody>"

        for row in sample.values:
            output += "<tr>" + "".join([f"<td>{val}</td>" for val in row]) + "</tr>"
        output += "</tbody></table></div></div>"

        return render(request, 'UserScreen.html', {
            'data': output,
            'watch_data': fetch_smartwatch_data()
        })
    
def generate_watch_data():
    return {
        'heart_rate': randint(60, 100),
        'temperature': round(uniform(36.0, 38.0), 1),
        'spo2': randint(93, 100),
        'bp_sys': randint(110, 140),
        'bp_dia': randint(70, 90),
        'steps': randint(1000, 8000)
    }

def generate_watch_data_view(request):
    return JsonResponse(generate_watch_data())

def TrainModel(request):
    if request.method == 'GET':
        output = """
        <div class="container mt-4">
            <div class="alert alert-success" role="alert">
                <h5 class="mb-0">✅ Model Training Results</h5>
            </div>
            <table class="table table-bordered text-center mt-3">
                <thead class="table-dark">
                    <tr>
                        <th>Algorithm</th>
                        <th>Accuracy (%)</th>
                        <th>Precision (%)</th>
                        <th>Recall (%)</th>
                        <th>F1 Score (%)</th>
                        <th>Confusion Matrix</th>
                        <th>Prediction Pie</th>
                    </tr>
                </thead>
                <tbody>
        """

        for name, model in models.items():
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds) * 100
            prec = precision_score(y_test, preds, average='macro') * 100
            rec = recall_score(y_test, preds, average='macro') * 100
            fs = f1_score(y_test, preds, average='macro') * 100
            cm = confusion_matrix(y_test, preds)

            # Confusion matrix
            fig_cm, axis_cm = plt.subplots(figsize=(3, 3))
            sns.heatmap(cm, annot=True, xticklabels=class_label, yticklabels=class_label, cmap="Blues", fmt="g", ax=axis_cm)
            axis_cm.set_title(f"{name} Confusion Matrix")
            buf_cm = io.BytesIO()
            plt.savefig(buf_cm, format='png', bbox_inches='tight')
            cm_img_b64 = base64.b64encode(buf_cm.getvalue()).decode()
            plt.close()

            # Pie chart
            fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
            class_counts = [np.sum(preds == i) for i in range(len(class_label))]
            ax_pie.pie(class_counts, labels=class_label, autopct='%1.1f%%', startangle=90)
            ax_pie.set_title(f"{name} Prediction Distribution")
            buf_pie = io.BytesIO()
            plt.savefig(buf_pie, format='png', bbox_inches='tight')
            pie_img_b64 = base64.b64encode(buf_pie.getvalue()).decode()
            plt.close()

            output += f"""
                <tr>
                    <td>{name}</td>
                    <td>{acc:.2f}</td>
                    <td>{prec:.2f}</td>
                    <td>{rec:.2f}</td>
                    <td>{fs:.2f}</td>
                    <td><img src='data:image/png;base64,{cm_img_b64}' height='120'></td>
                    <td><img src='data:image/png;base64,{pie_img_b64}' height='120'></td>
                </tr>
            """

        output += "</tbody></table></div>"

        return render(request, 'UserScreen.html', {
            'data': output,
            'watch_data': generate_watch_data()
        })


def TrainModel(request):
    if request.method == 'GET':
        output = """
        <div class="container mt-4">
            <div class="alert alert-success" role="alert">
                <h5 class="mb-0">✅ Model Training Results</h5>
            </div>
            <table class="table table-bordered text-center mt-3">
                <thead class="table-dark">
                    <tr>
                        <th>Algorithm</th>
                        <th>Accuracy (%)</th>
                        <th>Precision (%)</th>
                        <th>Recall (%)</th>
                        <th>F1 Score (%)</th>
                        <th>Confusion Matrix</th>
                        <th>Prediction Pie</th>
                    </tr>
                </thead>
                <tbody>
        """

        for name, model in models.items():
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds) * 100
            prec = precision_score(y_test, preds, average='macro') * 100
            rec = recall_score(y_test, preds, average='macro') * 100
            fs = f1_score(y_test, preds, average='macro') * 100
            cm = confusion_matrix(y_test, preds)

            # Confusion Matrix Image
            fig_cm, axis_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, xticklabels=class_label, yticklabels=class_label, cmap="Blues", fmt="g", ax=axis_cm)
            axis_cm.set_title(f"{name} Confusion Matrix")
            buf_cm = io.BytesIO()
            plt.savefig(buf_cm, format='png', bbox_inches='tight')
            cm_img_b64 = base64.b64encode(buf_cm.getvalue()).decode()
            plt.close()

            # Pie Chart Image
            fig_pie, ax_pie = plt.subplots()
            class_counts = [np.sum(preds == i) for i in range(len(class_label))]
            ax_pie.pie(class_counts, labels=class_label, autopct='%1.1f%%', startangle=90)
            ax_pie.set_title(f"{name} Prediction Distribution")
            buf_pie = io.BytesIO()
            plt.savefig(buf_pie, format='png', bbox_inches='tight')
            pie_img_b64 = base64.b64encode(buf_pie.getvalue()).decode()
            plt.close()

            output += f"""
                <tr>
                    <td>{name}</td>
                    <td>{acc:.2f}</td>
                    <td>{prec:.2f}</td>
                    <td>{rec:.2f}</td>
                    <td>{fs:.2f}</td>
                    <td><img src='data:image/png;base64,{cm_img_b64}' height='150'></td>
                    <td><img src='data:image/png;base64,{pie_img_b64}' height='150'></td>
                </tr>
            """

        output += "</tbody></table></div>"

        return render(request, 'UserScreen.html', {
            'data': output,
            'watch_data': generate_watch_data()
        })

def Predict(request):
    return render(request, 'Predict.html')

def BatchPredictAction(request):
    if request.method == 'POST':
        file = request.FILES['file']
        df = pd.read_csv(file)
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'].astype(str))
        df_scaled = scaler.transform(df.values)
        predictions = xgb_cls.predict(df_scaled)
        labels = [class_label[p] for p in predictions]

        output = "<h4>Batch Prediction Results</h4><table class='table table-bordered'><tr><th>Sample</th><th>Prediction</th></tr>"
        for i, label in enumerate(labels):
            color = 'text-success' if label == 'Low Risk' else 'text-danger'
            output += f"<tr><td>Patient {i+1}</td><td class='{color}'>{label}</td></tr>"
        output += "</table>"

        return render(request, 'UserScreen.html', {
            'data': output,
            'watch_data': fetch_smartwatch_data()
        })

def fetch_smartwatch_data():
    return {
        'heart_rate': randint(60, 100),
        'temperature': round(uniform(36.0, 38.0), 1),
        'spo2': randint(93, 100),
        'bp_sys': randint(110, 140),
        'bp_dia': randint(70, 90),
        'steps': randint(1000, 8000)
    }

def generate_watch_data():
    return {
        'heart_rate': randint(60, 100),
        'temperature': round(uniform(36.0, 38.0), 1),
        'spo2': randint(93, 100),
        'bp_sys': randint(110, 140),
        'bp_dia': randint(70, 90),
        'steps': randint(1000, 8000)
    }

def PredictAction(request):
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            gender = request.POST.get('gender')
            age = float(request.POST.get('age'))
            heart_rate = float(request.POST.get('heart_rate'))
            temperature = float(request.POST.get('temperature'))
            respiration_rate = float(request.POST.get('respiration_rate'))
            bp_sys = float(request.POST.get('bp_sys'))
            bp_dia = float(request.POST.get('bp_dia'))
            spo2 = float(request.POST.get('spo2'))

            # Vital values
            vitals = {
                'Age': age,
                'Heart Rate': heart_rate,
                'Temperature': temperature,
                'Respiration Rate': respiration_rate,
                'Systolic BP': bp_sys,
                'Diastolic BP': bp_dia,
                'SpO2': spo2,
            }

            # Safe thresholds
            thresholds = {
                'Age': (0, 120),
                'Heart Rate': (60, 100),
                'Temperature': (36.1, 37.2),
                'Respiration Rate': (12, 20),
                'Systolic BP': (90, 120),
                'Diastolic BP': (60, 80),
                'SpO2': (95, 100),
            }

            # Risk check
            risk_flags = [
                not (thresholds[k][0] <= v <= thresholds[k][1])
                for k, v in vitals.items()
            ]
            risk_count = sum(risk_flags)

            if risk_count >= 2:
                pred_label = 'High Risk'
                result_color = 'red'
            else:
                pred_label = 'Low Risk'
                result_color = 'green'

            result_text = f"{name} is at {pred_label}"

            # -------------- Graph --------------
            import matplotlib.pyplot as plt
            import numpy as np
            import io
            import base64

            fig, ax = plt.subplots(figsize=(11, 5))
            categories = list(vitals.keys())
            values = list(vitals.values())
            x = np.arange(len(categories))

            # Bar colors
            bar_colors = []
            for i, cat in enumerate(categories):
                val = values[i]
                low, high = thresholds[cat]
                bar_colors.append('green' if low <= val <= high else 'red')

            # Draw bars
            ax.bar(x, values, color=bar_colors, width=0.5, zorder=2)

            # Add threshold lines
            for i, cat in enumerate(categories):
                low, high = thresholds[cat]
                ax.hlines(y=low, xmin=i - 0.25, xmax=i + 0.25, colors='orange',
                          linestyles='dashed', linewidth=2, label='Lower Threshold' if i == 0 else "")
                ax.hlines(y=high, xmin=i - 0.25, xmax=i + 0.25, colors='blue',
                          linestyles='dashed', linewidth=2, label='Upper Threshold' if i == 0 else "")

            # Value labels
            for i, val in enumerate(values):
                ax.text(x[i], val + 1, f'{val:.1f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=15)
            ax.set_ylabel('Measured Value')
            ax.set_title(f'{name} - Vitals vs Thresholds')
            ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

            # Add legend only once
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')

            # Convert to base64
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            return render(request, 'predict.html', {
                'result': result_text,
                'result_color': result_color,
                'graph_html': graph_base64
            })

        except Exception as e:
            return render(request, 'predict.html', {
                'result': f"Error: {str(e)}",
                'result_color': 'red'
            })

    return render(request, 'predict.html')



def contact_view(request):
    return render(request, 'contact.html')