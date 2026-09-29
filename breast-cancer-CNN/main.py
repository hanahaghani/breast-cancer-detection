import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from collections import Counter
from collections import Counter

from models.model_architectur.model import CNN_model
from utils.visualizaition import plot_training_history,plot_confusion_matrix,plot_classification_report

#  Define Transform
basic_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

#  use image folder to load dataset

train_dataset = datasets.ImageFolder(
    "/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/data/train",
    transform=basic_transform
)

valid_dataset = datasets.ImageFolder(
    "/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/data/valid",
    transform=basic_transform
)

test_dataset = datasets.ImageFolder(
    "/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/data/test",
    transform=basic_transform
)

#  inspect the dataset
print("train:", Counter(train_dataset.targets))
print("valid:", Counter(valid_dataset.targets))
print("test :", Counter(test_dataset.targets))
print(train_dataset.class_to_idx)

img,lbl=train_dataset[10]
print("Image shape:", img.shape)
print("Image dtype:", img.dtype)
print("Min:", img.min())
print("Max:", img.max())
print("Label:", lbl)

plt.imshow(img.permute(1,2,0))
plt.title(f"Label: {[ 'healthy' if lbl == 0 else 'cancer' ]}")
plt.axis("off")
plt.savefig("/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/img/img.png")
plt.close()

#  normalize the dataset
# compute the mean & std of the dataset
channel_sum = torch.zeros(3)
channel_sum_squared = torch.zeros(3)
num_pixels = 0
for i in range(len(train_dataset)):
    image,label=train_dataset[i]
    channel_sum += image.sum(dim=(1,2))
    channel_sum_squared += (image ** 2).sum(dim=(1, 2))
    pixels = image.shape[1] * image.shape[2]

    num_pixels += pixels

mean= channel_sum/num_pixels
std=(channel_sum_squared/num_pixels - mean**2)**0.5

print("\n--- Training Data Statistics ---")
print("Mean:",mean)
print("Standard Deviation:",std)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

# apply the transform to the datasets
train_dataset.transform=train_transform
valid_dataset.transform=transform
test_dataset.transform=transform


class_counts = Counter(train_dataset.targets)

num_class_0 = class_counts[0]
num_class_1 = class_counts[1]

total = num_class_0 + num_class_1

weight_0 = total / (2 * num_class_0)
weight_1 = total / (2 * num_class_1)

class_weights = torch.tensor(
    [weight_0, weight_1],
    dtype=torch.float32
)

print("Class counts:", class_counts)
print("Class weights:", class_weights)

# create dataloaders
train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
valid_loader=DataLoader(valid_dataset,batch_size=32,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)

# building the model(CNN)
torch.manual_seed(42)

model=CNN_model()

n_epochs=20
criterion=nn.CrossEntropyLoss(weight=class_weights)
optimizer=torch.optim.Adam(
    model.parameters(),lr=0.001
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)
# training the model
train_losses=[]
valid_losses=[]

best_valid_loss=float('inf')
patience=3
counter=0

# training loop
for epoch in range(n_epochs):
    # training phase
    model.train()
    train_loss=0.0
    for images,labels in train_loader:
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()* images.size(0)
    train_loss /= len(train_loader.dataset)

    # validation phase
    all_label=[]
    all_prediction=[]

    model.eval()

    valid_running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in valid_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            valid_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_label.extend(labels.tolist())
            all_prediction.extend(predicted.tolist())

    valid_loss = valid_running_loss / len(valid_loader)
    valid_accuracy = correct / total
    # Save losses
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    #_______test____
    correct = 0
    total = 0
    test_running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_running_loss / len(test_loader)
    test_accuracy = correct / total
    
    scheduler.step(valid_loss)

    if valid_loss<best_valid_loss:
        best_valid_loss = valid_loss
        counter = 0
        torch.save(model.state_dict(), "/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/models/model_save/best_model.pth")
    else:
        counter+=1

    if counter>=patience:
        print("early stop!!!!!!!!!!!!!!")
        break

    print(
    f"Epoch [{epoch + 1}/{n_epochs}]|"
    f"Train Loss: {train_loss:.4f}|"
    f"Validation Loss: {valid_loss:.4f}|"
    f"Validation Accuracy: {valid_accuracy:.4f}|"
    f"Test Loss:{test_loss:.4f}|"
    f"Test Accuracy: {test_accuracy:.4f}"
    )


model.load_state_dict(torch.load("/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/models/model_save/best_model.pth"))
model.eval()

test_all_label, test_all_prediction = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_all_label.extend(labels.tolist())
        test_all_prediction.extend(predicted.tolist())

test_acc = sum(p == l for p, l in zip(test_all_prediction, test_all_label)) / len(test_all_label)
print("Test accuracy:", test_acc)


plot_training_history(train_losses,valid_losses,"/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/img/training_history.png")
# test plot
plot_confusion_matrix(all_label,all_prediction,valid_dataset.classes,"/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/img/valid_confusion-matrix.png")
plot_classification_report( all_label,all_prediction,valid_dataset.classes,"/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/img/valid_classification-report.png")

plot_confusion_matrix(test_all_label,test_all_prediction,valid_dataset.classes,"/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/img/test_confusion-matrix.png")
plot_classification_report( test_all_label,test_all_prediction,valid_dataset.classes,"/home/pc/Documents/machine learning/breast-cancer-detection/breast-cancer-CNN/img/test_classification-report.png")
