import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

BATCH_SIZE = 32

# preprocessing
# normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor()])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,drop_last=True)

import matplotlib.pyplot as plt

examples = enumerate(train_loader)
batch_idx, (images, labels) = next(examples)
# images is the whole batch, so 32 x 1 x 28 x28

new_img = torch.cat((images[0], images[1], images[2]), dim=0)
plt.imshow(new_img.permute(1,2,0))
plt.show()

new_new_img = new_img[1]
plt.imshow(new_new_img)
plt.show()

fig = plt.figure()

for i in range(BATCH_SIZE):
    plt.subplot(4,8,i+1)
    plt.tight_layout()
    plt.imshow(images[i][0], cmap='gray', interpolation=None)
    plt.title('Ground Truth: {}'.format(labels[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
