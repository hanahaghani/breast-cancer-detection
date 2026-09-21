import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,StackingClassifier,GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score,classification_report,confusion_matrix,auc,roc_curve
import seaborn as sns
from sklearn.svm import SVC
from visualization import plot_histograms,plot_pca_scatter,plot_model_predictions

data=pd.read_csv("breast-cancer-data.csv")
data=data.drop(columns=['Unnamed: 32','id']) 

x=data.drop(columns='diagnosis')
y=data['diagnosis']

le=LabelEncoder()
y_encoded=le.fit_transform(y)

plot_pca_scatter(x,y_encoded,save_path='image/pca-cancer.png')

x_train,x_test,y_train,y_test=train_test_split(x,y_encoded,test_size=0.2,random_state=42)

preprocess_pipline= Pipeline([("scaler",StandardScaler(with_mean=False))])

x_train_preperd=preprocess_pipline.fit_transform(x_train)
x_test_preperd=preprocess_pipline.transform(x_test)

model=LogisticRegression()
model.fit(x_train_preperd,y_train)
y_preb=model.predict(x_test_preperd)

y_prob=model.predict_proba(x_test_preperd)[:,1]
fpr,tpr,thresholds= roc_curve(y_test,y_prob)
roc_auc=auc(fpr,tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,label=f'ROC curve logistic model(AUC ={roc_auc:.2f})')
plt.plot([0,1],[0,1],linestyle='--',color='gray')
plt.xlabel('false positive rate')
plt.ylabel('roc curve for breast-canser detection')
plt.legend()
plt.savefig('image/roc-curve.png')
plt.show()

cm=confusion_matrix(y_test,y_preb)
plt.figure(figsize=(5,4))
sns.heatmap(cm ,annot=True , fmt='d',cmap='Blues',
            xticklabels=['Benign','Malignant'],
            yticklabels=['Benign','Malignant'])
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix logistic model')
plt.savefig('image/confusion_matrix.png')
plt.show()

y_preb_new=(y_prob>=0.4).astype(int)
cm=confusion_matrix(y_test,y_preb_new)
plt.figure(figsize=(5,4))
sns.heatmap(cm ,annot=True , fmt='d',cmap='Blues',
            xticklabels=['Benign','Malignant'],
            yticklabels=['Benign','Malignant'])
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix logistic model')
plt.savefig('image/confusion_matrix.png')
plt.show()

report=classification_report(y_test,y_preb,output_dict=True)

svc_model=SVC(random_state=42,probability=True)
svc_model.fit(x_train_preperd,y_train)
y_preb_svc=svc_model.predict(x_test_preperd)

y_prob_svc=svc_model.decision_function(x_test_preperd)
fpr,tpr,thresholds= roc_curve(y_test,y_prob_svc)
roc_auc=auc(fpr,tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,label=f'ROC curve svc model (AUC ={roc_auc:.2f})')
plt.plot([0,1],[0,1],linestyle='--',color='gray')
plt.xlabel('false positive rate')
plt.ylabel('roc curve for breast-canser detection')
plt.legend()
plt.savefig('image/roc-curve-svc.png')
plt.show()

report_svc=classification_report(y_test,y_preb_svc,output_dict=True)

cm_svc=confusion_matrix(y_test,y_preb_svc)
plt.figure(figsize=(5,4))
sns.heatmap(cm ,annot=True , fmt='d',cmap='Blues',
            xticklabels=['Benign','Malignant'],
            yticklabels=['Benign','Malignant'])
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix svc model')
plt.savefig('image/confusion_matrix_svc.png')
plt.show()

rf_model=RandomForestClassifier(random_state=42)
rf_model.fit(x_train_preperd,y_train)

y_preb_rf=rf_model.predict(x_test_preperd)

print(classification_report(y_test,y_preb_rf))

gb_model=GradientBoostingClassifier(random_state=42)
gb_model.fit(x_train_preperd,y_train)

y_preb_gb=gb_model.predict(x_test_preperd)

print(classification_report(y_test,y_preb_gb))

base_model=[('rf',rf_model),('logic',model),('svc',svc_model),('gb',gb_model)]

stack_model=StackingClassifier(estimators=base_model,final_estimator=svc_model,cv=10,stack_method='predict_proba',passthrough=False)
stack_model.fit(x_train_preperd,y_train)

y_preb_stack=stack_model.predict(x_test_preperd)

y_prob_stack=stack_model.predict_proba(x_test_preperd)[:,1]
fpr,tpr,thresholds= roc_curve(y_test,y_prob_stack)
roc_auc=auc(fpr,tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr,tpr,label=f'ROC curve stacking model (AUC ={roc_auc:.2f})')
plt.plot([0,1],[0,1],linestyle='--',color='gray')
plt.xlabel('false positive rate')
plt.ylabel('roc curve for breast-canser detection')
plt.legend()
plt.savefig('image/roc-curve-stacking.png')
plt.show()

cm_svc=confusion_matrix(y_test,y_preb_stack)
plt.figure(figsize=(5,4))
sns.heatmap(cm ,annot=True , fmt='d',cmap='Blues',
            xticklabels=['Benign','Malignant'],
            yticklabels=['Benign','Malignant'])
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix stacking model')
plt.savefig('image/confusion_matrix_stacking.png')
plt.show()

print(classification_report(y_test,y_preb_stack))

new_data=x.iloc[[1]]
sample=preprocess_pipline.transform(new_data)
y_new=svc_model.predict(sample)
print(y_new)

print(le.classes_)

if y_new==1:
    print("cancer")
else:
    print("healthy")