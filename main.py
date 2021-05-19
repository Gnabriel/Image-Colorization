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
from sklearn.model_selection import train_test_split
import os
from skimage import color
from kmeans_init import *
from sklearn import preprocessing
from sklearn import neighbors
import pickle
from helper_functions import *


class ColorizationNet(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ColorizationNet, self).__init__()

        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]

        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]

        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]

        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]

        model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True), ]
        model8 += [nn.ReLU(True), ]

        model8 += [nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True), ]

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
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm


class LFWDataset(Dataset):
    def __init__(self, X, Y, transform):  # TODO: lägga till transform?
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
    if os.path.isfile("pickles/lfw.p"):
        l_images, ab_images = pickle.load(open("pickles/lfw.p", "rb"))
        return l_images, ab_images

    images = os.listdir("./data")
    l_images = np.empty((data_size, 256, 256))
    ab_images = np.empty((data_size, 256, 256, 2))
    for i in range(data_size):
        img_as_matrix = np.asarray(Image.open(f"./data/{images[i]}"))
        l_img, ab_img = preprocess_image(img_as_matrix)
        l_images[i] = l_img
        ab_images[i] = ab_img
        # l_images[i] = np.reshape(l_img, newshape=np.size(l_img))    # Image as vector.
        # ab_images[i, :, 0] = np.reshape(ab_img[:, :, 0], newshape=np.size(ab_img[:, :, 0]))
        # ab_images[i, :, 1] = np.reshape(ab_img[:, :, 1], newshape=np.size(ab_img[:, :, 1]))

        if i % 50 == 49:  # print every 20 mini-batches
            print("Loaded data nr.: " + str(i + 1))
    # Convert from float64 to int8
    l_images = l_images.astype(np.uint8)
    ab_images = ab_images.astype(np.int8)

    # Save
    # pickle.dump((l_images, ab_images), open("pickles/lfw.p", "wb"))

    print("Data loaded.")
    return l_images, ab_images


def discretize_images(ab_images):
    ab_images = np.floor_divide(ab_images, 10) * 10
    return ab_images


def discretized_ab_images_to_q(discretized_ab_images, ab_to_q_dict):  # TODO: Göra kopia av discretized_ab_images?
    def ab_to_q_index(a_color, b_color):
        color_key = f"({a_color}, {b_color})"
        return ab_to_q_dict[color_key]

    ab_images_to_q_index = np.vectorize(ab_to_q_index)
    discretized_q_images = ab_images_to_q_index(discretized_ab_images[:, :, :, 0], discretized_ab_images[:, :, :, 1])
    return discretized_q_images


def one_hot_encode_labels(q_images, Q_vector):
    q_images_flat = np.reshape(q_images, newshape=(q_images.shape[0], q_images.shape[1] * q_images.shape[2]))
    Q_matrix = np.tile(Q_vector, (q_images_flat.shape[1], 1))
    enc = preprocessing.OneHotEncoder(categories=Q_matrix)
    Y_one_hot = enc.fit_transform(q_images_flat)
    Y_one_hot = np.reshape(Y_one_hot, newshape=(100, 256, 256, 247))
    return Y_one_hot


def q_images_to_Z_old(q_images, Q):
    ab_to_q_dict = pickle.load(open("pickles/ab_to_q_index_dict.p", "rb"))
    q_to_ab_dict = pickle.load(open("pickles/q_index_to_ab_dict.p", "rb"))

    def nn_gaussian(q):
        q_vec = np.zeros(Q)
        possible_neighbor_dists = np.array([[0, 10], [10, 0], [0, -10], [-10, 0],  # Closest neighbours
                                   [10, 10], [10, -10], [-10, -10], [10, 10],  # Second closest
                                   [0, 20], [20, 0], [0, -20], [-20, 0],  # Third closest
                                   [10, 20], [20, 10], [20, -10], [10, -20], [-10, -20], [-20, -10], [-10, 20],
                                   [-20, 10],  # Fourth closest
                                   [20, 20], [20, -20], [-20, -20], [-20, 20]  # Fifth closest
                                   ])
        a, b = get_ab_colors_from_key(q_to_ab_dict[q])
        neighbours_found = 0
        for neighbor_dist in possible_neighbor_dists:
            if neighbours_found >= 5:
                break
            neighbour_color = [a, b] + np.array(neighbor_dist)
            neighbour_color_key = get_color_key(neighbour_color[0], neighbour_color[1])
            if neighbour_color_key in ab_to_q_dict:
                neighbours_found += 1
                neighbour_q = ab_to_q_dict[neighbour_color_key]
                q_vec[neighbour_q] = 1 / np.sqrt(neighbor_dist.dot(neighbor_dist))
        q_vec /= np.sum(q_vec)
        return q_vec

    nn_gaussian_array = np.vectorize(nn_gaussian)
    # Z_images = np.empty((q_images.shape[0], q_images.shape[1], q_images.shape[2], Q))
    Z_images = np.reshape(q_images, newshape=(q_images.shape[0], q_images.shape[1], q_images.shape[2], Q))
    Z_images = nn_gaussian_array(Z_images)
    return Z_images


def get_ab_domain(ab_to_q_dict):
    ab_domain_strings = list(ab_to_q_dict.keys())
    ab_domain = [list(get_ab_colors_from_key(ab_string)) for ab_string in ab_domain_strings]
    ab_domain = sorted(ab_domain, key=lambda x: (x[0], x[1]))
    return ab_domain


def q_images_to_Z(ab_images, Q, ab_domain, num_neighbors=5, sigma=5):
    data_size, w, h = ab_images.shape[0], ab_images.shape[1], ab_images.shape[2]
    nearest_neighbors = neighbors.NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree')
    nearest_neighbors = nearest_neighbors.fit(ab_domain)
    points = data_size * w * h
    ab_images_flat = np.reshape(ab_images, (points, 2))
    points_encoded_flat = np.empty((points, Q))
    points_indices = np.arange(0, points, dtype='int')[:, np.newaxis]

    distances, indices = nearest_neighbors.kneighbors(ab_images_flat)

    gaussian_kernel = np.exp(-distances**2 / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel, axis=1)[:, np.newaxis]

    points_encoded_flat[points_indices, indices] = gaussian_kernel
    points_encoded = np.reshape(points_encoded_flat, (data_size, w, h, Q))
    return points_encoded


def weighted_loss(outputs, labels):
    return torch.sum(outputs)


def train(model, trainloader, criterion, optimizer):

    #Temp
    ab_to_q_dict = pickle.load(open("pickles/ab_to_q_index_dict.p", "rb"))
    Q = 247
    ab_domain = get_ab_domain(ab_to_q_dict)



    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            print(labels.shape)
            labels = q_images_to_Z(labels, Q, ab_domain)

            # Load data to GPU if using Cuda.
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            # running_loss += loss.item()
            if i % 50 == 49:  # print every 20 mini-batches
                print("Epoch: " + str(epoch + 1) + ", Trained data: " + str(i * (len(inputs)) + 1))
                # print('[%d, %5d] loss: %.3f' %
                #       (epoch + 1, i + 1, running_loss / 20))
                # running_loss = 0.0

    print('Finished Training')


def test(model, testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # Load data to GPU if using Cuda.
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def main():
    # CNN.
    model = ColorizationNet()

    if torch.cuda.is_available():
        # model.to(torch.device("cuda:0"))
        model = model.cuda()
        torch.cuda.empty_cache()
    print("Model on GPU: " + str(next(model.parameters()).is_cuda))

    # Parameters.
    # data_size = 13233
    data_size = 100
    batch_size = 1
    learning_rate = 0.001  # ADAM Standard
    Q_vector = np.arange(247)
    Q = len(Q_vector)

    # Load & preprocess images.
    l_images, ab_images = load_images(data_size)
    ab_to_q_dict = pickle.load(open("pickles/ab_to_q_index_dict.p", "rb"))
    ab_domain = get_ab_domain(ab_to_q_dict)

    # Discretize data.
    ab_images = discretize_images(ab_images)
    print(ab_images.shape)
    # q_images = discretized_ab_images_to_q(ab_images, ab_to_q_dict)
    # ab_images = one_hot_encode_labels(ab_images, Q_vector=Q_vector)
    # Z_images = q_images_to_Z(ab_images, Q, ab_domain)
    # ab_images = Z_images

    # Convert from float64 to uint8.
    # l_images = l_images.astype(np.int8)
    # ab_images = ab_images.astype(np.int8)

    # Split and shuffle training- and test sets.
    train_X, test_X, train_Y, test_Y = train_test_split(l_images, ab_images, test_size=0.2)

    # Training data.
    # transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = LFWDataset(train_X, train_Y, transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Test data
    testset = LFWDataset(test_X, test_Y, transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Loss function.
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_wce = weighted_loss

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

    # kmeans_trainloader = torch.utils.data.DataLoader(trainset, batch_size=data_size, shuffle=True, num_workers=2)
    # for i, data in enumerate(kmeans_trainloader):
    #     kmeans_inputs, kmeans_labels = data.copy()
    #     if torch.cuda.is_available():
    #         kmeans_inputs, kmeans_labels = kmeans_inputs.cuda(), kmeans_labels.cuda()
    # kmeans_init(model, trainloader, num_iter=3, use_whitening=False)

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
