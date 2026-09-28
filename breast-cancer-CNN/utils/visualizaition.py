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