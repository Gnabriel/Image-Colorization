import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import pickle
from helper_functions import *


class ColorizationNet(nn.Module):
    """
    Credit:
    Richard Zhang, Phillip Isola, and Alexei A Efros.  Colorful image colorization.  In Computer Vision – ECCV 2016,
    Lecture Notes in Computer Science, pages 649–666, Cham, 2016. SpringerInternational Publishing.
    """
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ColorizationNet, self).__init__()

        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0
        self.Q = 247

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
        model8 += [nn.Conv2d(256, self.Q, kernel_size=1, stride=1, padding=0, bias=True), ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(self.Q, self.Q, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
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
        out_reg = self.upsample4(self.softmax(conv8_3))
        return out_reg

    def normalize_l(self, in_l):
        return (in_l - self.l_cent) / self.l_norm

    def unnormalize_l(self, in_l):
        return in_l * self.l_norm + self.l_cent

    def normalize_ab(self, in_ab):
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab):
        return in_ab * self.ab_norm


class LFWDataset(Dataset):
    def __init__(self, X, Y, transform, ab_domain):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.ab_domain = ab_domain
        self.nearest_neighbors = neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        self.nearest_neighbors = self.nearest_neighbors.fit(ab_domain)

    def __getitem__(self, index):
        l_image, ab_image = self.X[index], self.Y[index]
        ab_image = ab_image_to_Z(ab_image, len(self.ab_domain), self.nearest_neighbors, sigma=5)
        return self.transform(l_image), self.transform(ab_image).float()

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
    if os.path.isfile("pickles/lfw_{}.p".format(data_size)):
        print("Data loaded from pickle.")
        l_images, ab_images = pickle.load(open("pickles/lfw_{}.p".format(data_size), "rb"))
        return l_images, ab_images

    images = os.listdir("./data")
    l_images = np.empty((data_size, 256, 256))
    ab_images = np.empty((data_size, 256, 256, 2))
    for i in range(data_size):
        img_as_matrix = np.asarray(Image.open(f"./data/{images[i]}"))
        l_img, ab_img = preprocess_image(img_as_matrix)
        l_images[i] = l_img
        ab_images[i] = ab_img

        if i % 50 == 49:  # print every 50 mini-batches
            print("Loaded data nr.: " + str(i + 1))
    # Convert from float64 to int8
    l_images = l_images.astype(np.int8)
    ab_images = ab_images.astype(np.int8)

    # Save
    pickle.dump((l_images, ab_images), open("pickles/lfw_{}.p".format(data_size), "wb"))

    print("Data loaded.")
    return l_images, ab_images


def discretize_images(ab_images):
    ab_images = np.floor_divide(ab_images, 10) * 10
    return ab_images


def get_ab_domain(data_size, ab_to_q_dict_unsorted):
    if os.path.isfile("pickles/ab_domain_{}.p".format(data_size)):
        ab_domain = pickle.load(open("pickles/ab_domain_{}.p".format(data_size), "rb"))
        print("ab_domain loaded from pickle.")
        return ab_domain

    ab_domain_strings = list(ab_to_q_dict_unsorted.keys())
    ab_domain = [list(get_ab_colors_from_key(ab_string)) for ab_string in ab_domain_strings]
    ab_domain = sorted(ab_domain, key=lambda x: (x[0], x[1]))

    print("ab_domain computed.")
    pickle.dump(ab_domain, open("pickles/ab_domain_{}.p".format(data_size), "wb"))
    return ab_domain


def get_ab_to_q_dict(data_size, ab_domain):
    if os.path.isfile("pickles/ab_to_q_dict_{}.p".format(data_size)):
        ab_to_q_dict = pickle.load(open("pickles/ab_to_q_dict_{}.p".format(data_size), "rb"))
        print("ab_to_q_dict loaded from pickle.")
        return ab_to_q_dict
    q_values = np.arange(0, len(ab_domain))
    ab_to_q_dict = dict(zip(map(tuple, ab_domain), q_values))
    print("ab_to_q_dict computed.")
    pickle.dump(ab_to_q_dict, open("pickles/ab_to_q_dict_{}.p".format(data_size), "wb"))
    return ab_to_q_dict


def get_q_to_ab_dict(data_size, ab_to_q_dict):
    if os.path.isfile("pickles/q_to_ab_dict_{}.p".format(data_size)):
        q_to_ab_dict = pickle.load(open("pickles/q_to_ab_dict_{}.p".format(data_size), "rb"))
        print("q_to_ab_dict loaded from pickle.")
        return q_to_ab_dict
    ab_to_q_dict = ab_to_q_dict.copy()
    q_to_ab_dict = {v: k for k, v in ab_to_q_dict.items()}
    print("q_to_ab_dict computed.")
    pickle.dump(q_to_ab_dict, open("pickles/q_to_ab_dict_{}.p".format(data_size), "wb"))
    return q_to_ab_dict


def get_p(data_size, ab_images, ab_to_q_dict, Q):
    if os.path.isfile("pickles/p_{}.p".format(data_size)):
        p = pickle.load(open("pickles/p_{}.p".format(data_size), "rb"))
        print("p loaded from pickle.")
        return p
    data_size, w, h, _ = ab_images.shape
    p = np.zeros(Q)

    def add_on_index(ab):
        q = ab_to_q_dict[tuple(ab)]
        p[q] += 1

    np.apply_along_axis(add_on_index, 3, ab_images)
    p /= data_size * w * h
    print("p computed.")
    pickle.dump(p, open("pickles/p_{}.p".format(data_size), "wb"))
    return p


def get_train_test(data_size, l_images, ab_images, test_size=0.05):
    if os.path.isfile("pickles/train_test_{}.p".format(data_size)):
        train_X, test_X, train_Y, test_Y = pickle.load(open("pickles/train_test_{}.p".format(data_size), "rb"))
        print("train_test loaded from pickle.")
        return train_X, test_X, train_Y, test_Y
    train_X, test_X, train_Y, test_Y = train_test_split(l_images, ab_images, test_size=test_size, shuffle=True)
    print("train_test computed.")
    pickle.dump((train_X, test_X, train_Y, test_Y), open("pickles/train_test_{}.p".format(data_size), "wb"))
    return train_X, test_X, train_Y, test_Y


def ab_image_to_Z(ab_image, Q, nearest_neighbors, sigma=5):
    w, h = ab_image.shape[0], ab_image.shape[1]
    points = w * h
    ab_images_flat = np.reshape(ab_image, (points, 2))
    points_encoded_flat = np.empty((points, Q))
    points_indices = np.arange(0, points, dtype='int')[:, np.newaxis]

    distances, indices = nearest_neighbors.kneighbors(ab_images_flat)

    gaussian_kernel = np.exp(-distances**2 / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel, axis=1)[:, np.newaxis]

    points_encoded_flat[points_indices, indices] = gaussian_kernel
    points_encoded = np.reshape(points_encoded_flat, (w, h, Q))
    return points_encoded


def get_loss_weights(Z_tens, p_tilde_tens, Q, lam=0.5):
    w = ((1 - lam) * p_tilde_tens + lam / Q) ** -1
    w /= torch.sum(p_tilde_tens*w)
    q_star_matrix = torch.argmax(Z_tens, 1)
    weights = w[q_star_matrix]
    return weights.cuda()


def weighted_cross_entropy_loss(Z_hat_tens, Z_tens, batch_size, p_tilde_tens, Q):
    eps = torch.tensor(0.0001).cuda()
    weights = get_loss_weights(Z_tens, p_tilde_tens, Q)
    loss = -torch.sum(weights * torch.sum(Z_tens * torch.log(Z_hat_tens + eps), 1)) / batch_size
    return loss


def train(model, trainloader, data_size, num_epochs, batch_size, criterion, optimizer, p_tilde_tens, Q):
    current_epoch = 0
    losses = []
    # Loop over the dataset.
    for epoch in range(num_epochs):
        losses.append([])
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # Load data to GPU if using Cuda.
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            # Zero parameter gradients.
            optimizer.zero_grad()

            # Forward-pass.
            outputs = model(inputs)

            # Loss function.
            loss = criterion(outputs, labels, batch_size, p_tilde_tens, Q)          # Custom loss
            # loss = criterion(outputs, labels)                                     # MSE

            # Backward-pass.
            loss.backward()

            optimizer.step()

            # Save losses.
            losses[current_epoch].append(loss.item())

            # Debugging.
            if i % 50 == 49:
                print("Epoch: " + str(current_epoch + 1) + ", Trained data: " + str((i+1) * batch_size))
        current_epoch += 1

    # Save losses.
    if os.path.isfile('./losses_{}.pth'.format(data_size)):
        os.remove('./losses_{}.pth'.format(data_size))
    pickle.dump(losses, open("pickles/losses_{}.p".format(data_size), "wb"))

    print('Finished Training')


def main():
    # CNN.
    model = ColorizationNet()

    if torch.cuda.is_available():
        model = model.cuda()
        torch.cuda.empty_cache()
    print("Model on GPU: " + str(next(model.parameters()).is_cuda))

    # Parameters.
    # data_size = 13233
    data_size = 13233
    batch_size = 1
    num_epochs = 2
    learning_rate = 0.00003

    # Load & preprocess images.
    l_images, ab_images = load_images(data_size)
    ab_to_q_dict_unsorted = pickle.load(open("pickles/ab_to_q_index_dict_unsorted.p", "rb"))
    ab_domain = get_ab_domain(data_size, ab_to_q_dict_unsorted)
    ab_to_q_dict = get_ab_to_q_dict(data_size, ab_domain)
    Q = len(ab_domain)

    # Discretize data.
    ab_images = discretize_images(ab_images)

    # Split and shuffle training- and test sets.
    train_X, test_X, train_Y, test_Y = get_train_test(data_size, l_images, ab_images, test_size=0.05)

    # Data transformer.
    transform = transforms.Compose([transforms.ToTensor()])

    # Training data.
    trainset = LFWDataset(train_X, train_Y, transform, ab_domain)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

    # Test data
    testset = LFWDataset(test_X, test_Y, transform, ab_domain)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Loss function.
    # criterion = nn.MSELoss()                          # MSE
    criterion = weighted_cross_entropy_loss             # Custom loss

    p = get_p(data_size, ab_images, ab_to_q_dict, Q)
    p_tilde = gaussian_filter(p, sigma=5)
    p_tilde_tens = torch.tensor(p_tilde).cuda()

    # Optimizer.
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.001)

    # Train the network.
    train(model, trainloader, data_size, num_epochs, batch_size, criterion, optimizer, p_tilde_tens, Q)

    # Save the trained network.
    if os.path.isfile('./colorize_cnn_{}.pth'.format(data_size)):
        os.remove('./colorize_cnn_{}.pth'.format(data_size))
    torch.save(model.state_dict(), './colorize_cnn_{}.pth'.format(data_size))


if __name__ == '__main__':
    main()
