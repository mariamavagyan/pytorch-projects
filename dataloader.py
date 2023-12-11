# import torch
# import torch.nn as nn
import torch.utils.data as Data
import torchvision

import os

class MNISTDataLoader:

    DOWNLOAD_MNIST = False # why is this here? is it global?

    # If MNIST directory does not exist: not(os.path.exists('./mnist'))
    # or is empty: os.listdir('./mnist') -- this lists the contents of the dir
    if not(os.path.exists('./mnist')) or not os.listdir('./mnist'):
        DOWNLOAD_MNIST = True
    
    def __init__(self, batch_size = 32, num_workers = 2):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = torchvision.transforms.ToTensor()

    
    def _load_train(self):

        self.train_data = torchvision.datasets.MNIST(root='./mnist',
                                                     train = True,
                                                     transform=self.transform,
                                                     download=self.DOWNLOAD_MNIST)
        
        self.train_loader = Data.DataLoader(dataset = self.train_data,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=self.num_workers)
        return self.train_loader
        
    def _load_test(self):

        self.test_data = torchvision.datasets.MNIST(root='./mnist',
                                                    train = False,
                                                    transform=self.transform,
                                                    download=self.DOWNLOAD_MNIST)
        
        self.test_loader = Data.DataLoader(dataset = self.test_data,
                                           batch_size=self.batch_size,
                                           shuffle = True,
                                           num_workers=self.num_workers)