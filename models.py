import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # def __init__(self, *args, **kwargs) -> None:
    #     super().__init__(*args, **kwargs)

    def __init__(self) -> None: # the function returns none
        super(CNN), self.__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2) # kernel size and stride
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop = nn.Dropout2d() # p=0.5 as default probability
        self.fc1 = nn.Linear(320, 50) # 320 = 20 channels * 4 x 4 channel dim
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # Convolution -> Activation -> Dropout -> Pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dims except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # when using CrossEntropy as loss, output raw x
        # nn.CrossEntropyLoss()
        # when using negative log-likelihood, output softmax(x)
        # F.nll_loss()

        return x

