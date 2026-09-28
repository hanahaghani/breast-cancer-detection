import torch 
import torch.nn as nn

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        # define the layers of the CNN model
        # convolutional layer 1
        self.conv1=nn.Conv2d(
            in_channels=3,out_channels=16,kernel_size=3
        )
        # convolutional layer 2 
        self.conv2=nn.Conv2d(
            in_channels=16,out_channels=32,kernel_size=3
        )
        #convolutional layer 3
        self.conv3=nn.Conv2d(
            in_channels=32,out_channels=64,kernel_size=3
        )
        #activation function
        self.relu=nn.ReLU()

        #pooling layer
        self.pool=nn.MaxPool2d(
            kernel_size=2,stride=2
        )

        #classifier layer
        self.fc=nn.Linear(
            in_features=64 * 26 * 26,out_features=2
        )

        #dropout layer
        self.dropout=nn.Dropout(0.3)

    def forward(self,x):
        # pass the input through the layers of the model
        x=self.conv1(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.dropout(x)

        x=self.conv2(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.dropout(x)

        x=self.conv3(x)
        x=self.relu(x)
        x=self.pool(x)
        x=self.dropout(x)

        x=torch.flatten(x, start_dim=1)
        x=self.fc(x)

        return(x)