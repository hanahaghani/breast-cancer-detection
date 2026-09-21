import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_histograms(df):
    df.hist(figsize=(6,6), bins=20)
    plt.show()

def plot_pca_scatter(X, y, save_path=None):
    # استاندارد سازی
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA با دو کامپوننت
    pca = PCA(n_components=15)
    X_pca = pca.fit_transform(X_scaled)

    # رسم scatter plot
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette={0:"blue",1:"red"}, alpha=0.7)
    plt.title("PCA Scatter Plot")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    # ذخیره اگر مسیر داده شده
    if save_path:
        plt.savefig(save_path, dpi=200)
    
    plt.show()
    

def plot_model_predictions(X, y_true, y_pred, savepath=None):
    """
    X: ویژگی‌ها (2 بعدی برای نمایش PCA یا دو ویژگی)
    y_true: برچسب‌های واقعی
    y_pred: برچسب‌های پیش‌بینی شده توسط مدل
    savepath: مسیر برای ذخیره تصویر (اختیاری)
    """
    # تبدیل همه به np.array و int برای اطمینان
    X = np.array(X)
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    # بررسی ابعاد
    assert X.shape[0] == len(y_true) == len(y_pred), "X, y_true و y_pred باید طول یکسان داشته باشند!"
    
    # تعریف TP, TN, FP, FN
    tp = (y_true == 1) & (y_pred == 1)
    tn = (y_true == 0) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    
    fig, ax = plt.subplots(figsize=(8,6))
    
    ax.scatter(X[tp, 0], X[tp, 1], c='green', label='True Positive', alpha=0.7)
    ax.scatter(X[tn, 0], X[tn, 1], c='blue', label='True Negative', alpha=0.7)
    ax.scatter(X[fp, 0], X[fp, 1], c='pink', label='False Positive', alpha=0.7)
    ax.scatter(X[fn, 0], X[fn, 1], c='red', label='False Negative', alpha=0.7)
    
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title('Model Predictions on Dataset')
    ax.legend()
    ax.grid(True)
    
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    
    plt.show()