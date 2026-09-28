#import library
import matplotlib.pyplot as plt
import seaborn as sns
from  sklearn.metrics  import confusion_matrix,classification_report

#training loss vs validaition loss
def plot_training_history(training_losses,valid_losses,save_path):

    plt.figure(figsize=(10,8))
    epochs=range(1,len(training_losses)+1)
    plt.plot(epochs,training_losses,label="training loss")
    plt.plot(epochs,valid_losses,label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title("training and validation loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

#confusion matrix heatmap
def plot_confusion_matrix(true_labels,prediction_labels,class_names,save_path):
    cm=confusion_matrix(true_labels,prediction_labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=class_names,yticklabels=class_names)
    plt.xlabel("predicted label")
    plt.ylabel("true label")
    plt.title("confusion matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#classification report

def plot_classification_report(labels,predictions,class_names,save_path):
    report = classification_report(labels,predictions,target_names=class_names,output_dict=True)

    classes = class_names

    precision = [report[class_name]["precision"] for class_name in classes]
    recall = [report[class_name]["recall"] for class_name in classes]
    f1_score = [report[class_name]["f1-score"] for class_name in classes]

    x = range(len(classes))

    plt.figure(figsize=(12, 6))

    plt.bar([i - 0.2 for i in x],precision,width=0.2,label="Precision")

    plt.bar(x,recall,width=0.2,label="Recall")

    plt.bar([i + 0.2 for i in x],f1_score,width=0.2,label="F1-score")

    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.title("Classification Report")
    plt.xticks(list(x), classes, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()