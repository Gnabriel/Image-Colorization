import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import random
import sklearn
from sklearn.model_selection import train_test_split
import os
from skimage import color
from kmeans_init import *
from sklearn import preprocessing


class ColorizationNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ColorizationNet, self).__init__()

        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))

    def normalize_l(self, in_l):
        return (in_l-self.l_cent)/self.l_norm

    def unnormalize_l(self, in_l):
        return in_l*self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab/self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab*self.ab_norm


class LFWDataset(Dataset):
    def __init__(self, X, Y, transform):       # TODO: lägga till transform?
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        l_image, ab_image = self.X[index], self.Y[index]
        return self.transform(l_image), self.transform(ab_image)

    def __len__(self):
        return len(self.X)


def resize_image(image, h=256, w=256, resample=3):
    return np.asarray(Image.fromarray(image).resize((h, w), resample=resample))


def preprocess_image(image):
    """
    Resize and split image into L and ab channels.
    :param image: RGB-image
    :return:
    """
    image = resize_image(image)
    lab_image = color.rgb2lab(image)
    l_image = lab_image[:, :, 0]
    ab_image = lab_image[:, :, 1:]
    return l_image, ab_image


def load_images(data_size):
    images = os.listdir("./data")
    l_images = np.empty((data_size, 256*256))
    ab_images = np.empty((data_size, 256*256, 2))
    for i in range(data_size):
            img_as_matrix = np.asarray(Image.open(f"./data/{images[i]}"))
            l_img, ab_img = preprocess_image(img_as_matrix)
            l_images[i] = np.reshape(l_img, newshape=np.size(l_img))    # Image as vector.
            ab_images[i, :, 0] = np.reshape(ab_img[:, :, 0], newshape=np.size(ab_img[:, :, 0]))
            ab_images[i, :, 1] = np.reshape(ab_img[:, :, 1], newshape=np.size(ab_img[:, :, 1]))
    # Convert from float64 to uint8
    l_images = (255 * l_images.astype(np.float64)).astype(np.uint8)
    ab_images = (255 * ab_images.astype(np.float64)).astype(np.uint8)

    print("Data loaded.")
    return l_images, ab_images


def train(model, trainloader, criterion, optimizer):
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0

    print('Finished Training')


def test(model, testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def main():
    print(torch.cuda.is_available())

    # CNN.
    model = ColorizationNet()

    # Parameters.
    data_size = 40
    batch_size = 1
    learning_rate = 0.001       # ADAM Standard
    Q = 300

    # Load & preprocess images.
    l_images, ab_images = load_images(data_size)

    # Discretize data.
    # TODO: Fixa så att discretizer lägger värden i samma bins som gabbe&razzmus.
    # l_images = l_images[:, 100:105]
    # ab_images = ab_images[:, 100:105, :]
    # print("--- L channel original ---")
    # print(l_images)
    # print("--- a channel original ---")
    # print(ab_images[:, :, 0])
    # print("--- b channel original ---")
    # print(ab_images[:, :, 1])
    discretizer = preprocessing.KBinsDiscretizer(n_bins=Q, encode='ordinal', strategy='uniform')      # TODO: Encode/strategy?
    l_images = discretizer.fit_transform(l_images)
    ab_images[:, :, 0] = discretizer.fit_transform(ab_images[:, :, 0])
    ab_images[:, :, 1] = discretizer.fit_transform(ab_images[:, :, 1])
    # print("--- L channel disc ---")
    # print(l_images)
    # print("--- a channel disc ---")
    # print(ab_images[:, :, 0])
    # print("--- b channel disc ---")
    # print(ab_images[:, :, 1])

    # Resize back to 256x256 images.
    l_images = np.reshape(l_images, newshape=(data_size, 256, 256))
    ab_images = np.reshape(ab_images, newshape=(data_size, 256, 256, 2))
    # Convert from float64 to uint8.
    l_images = (Q * l_images.astype(np.float64)).astype(np.uint8)       # TODO: Rätt med Q?
    ab_images = (Q * ab_images.astype(np.float64)).astype(np.uint8)

    # Split and shuffle training- and test sets.
    # train_X, test_X, train_Y, test_Y = train_test_split(l_images, ab_images, test_size=0.2, stratify=ab_images)
    train_X, test_X, train_Y, test_Y = train_test_split(l_images, ab_images, test_size=0.2)

    # Training data.
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    trainset = LFWDataset(train_X, train_Y, transform)
    # for data in trainset:
    #     print(np.shape(list(data)))
    #     for tensor in data:
    #         print(tensor.shape)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Test data
    testset = LFWDataset(test_X, test_Y, transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Loss function.
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    # criterion_mce = våran egen

    # Optimizer.
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.001)

    # k-means initialization.
    # train_X_tens = np.empty(train_X.shape[0], dtype='object')
    # for i in range(train_X.shape[0]):
    #     train_X_tens[i] = transform(train_X[i])
    #     # train_X_tens[i] = torch.from_numpy(train_X[i])

    kmeans_trainloader = torch.utils.data.DataLoader(trainset, batch_size=data_size, shuffle=True, num_workers=2)
    for i, data in enumerate(kmeans_trainloader):
        kmeans_inputs, kmeans_labels = data.copy()
    kmeans_init(model, kmeans_inputs, num_iter=3, use_whitening=False)

    # Train the network.
    train(model, trainloader, criterion_mse, optimizer)

    # Save the trained network.
    torch.save(model.state_dict(), './colorize_cnn.pth')

    # Test the network.
    # test(model, testloader)

    # Show random image.
    # random_image = random.randint(0, len(images))
    # plt.imshow(images[random_image])
    # plt.title(f"Training example #{random_image}")
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    main()
