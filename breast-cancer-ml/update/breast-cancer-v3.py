import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score,classification_report,confusion_matrix,roc_curve,auc
from visualization import plot_histograms,plot_pca_scatter,plot_model_predictions
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import joblib

data=pd.read_csv("breast-cancer-data.csv")
data=data.drop(columns=['Unnamed: 32','id']) 

data.info()
print(data.head())
print(data.describe())

x=data.drop(columns='diagnosis')
y=data['diagnosis']

print(x.shape)
print(y.shape)

le=LabelEncoder()
y_encoded=le.fit_transform(y)

plot_pca_scatter(x,y_encoded,save_path='image/pca-cancer.png')

x_train,x_test,y_train,y_test=train_test_split(
    x,y_encoded,test_size=0.2,random_state=42
)

preprocess_pipline= Pipeline([
    ('imputer',SimpleImputer(strategy='median')),
    ("scaler",StandardScaler()),
    ('pca',PCA(n_components=15))
])

x_train_preperd=preprocess_pipline.fit_transform(x_train)
x_test_preperd=preprocess_pipline.transform(x_test)

model=LogisticRegression()
model.fit(x_train_preperd,y_train)
y_preb=model.predict(x_test_preperd)

print(classification_report(y_test,y_preb))
print(cross_val_score(model,x_train_preperd, y_train, cv=3, scoring="accuracy"))

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

print(model.score(x_train_preperd,y_train))
print(model.score(x_test_preperd,y_test))

svc_model=SVC(random_state=42,probability=True)
svc_model.fit(x_train_preperd,y_train)
y_preb_svc=svc_model.predict(x_test_preperd)

print(classification_report(y_test,y_preb_svc))
print(cross_val_score(svc_model,x_train_preperd, y_train, cv=3, scoring="accuracy"))

cm_svc=confusion_matrix(y_test,y_preb_svc)
plt.figure(figsize=(5,4))
sns.heatmap(cm_svc,annot=True , fmt='d',cmap='Blues',
            xticklabels=['Benign','Malignant'],
            yticklabels=['Benign','Malignant'])
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('confusion matrix svc model')
plt.savefig('image/confusion_matrix_svc.png')
plt.show()

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

print(svc_model.score(x_train_preperd,y_train))
print(svc_model.score(x_test_preperd,y_test))

tsne=TSNE(
    n_components=3,perplexity=30,random_state=42,init='pca',learning_rate='auto'         
)
x_train_tsne=tsne.fit_transform(x_train)
plt.figure(figsize=(10,8))
scatter=plt.scatter(
    x_train_tsne[:,0],x_train_tsne[:,1],c=y_train,cmap='bwr',s=5
)
plt.title('tsne projection')
plt.savefig('image/tnse-scatter.png')
plt.show()

new_data=x.iloc[[1]]
sample=preprocess_pipline.transform(new_data)
y_new=model.predict(sample)
print(y_new)

print(le.classes_)

if y_new==1:
    print("cancer")
else:
    print("healthy")


joblib.dump(model,'model/logstic-model.joblib')
joblib.dump(svc_model,"model/svc-cancer.joblib")

print('model saved')