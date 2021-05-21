from scipy import ndimage

from main import *


def get_trained_CNN(data_size):
    model = ColorizationNet()
    model.load_state_dict(torch.load('./colorize_cnn_{}.pth'.format(data_size)))
    model.eval()
    print("Loaded CNN: colorize_cnn_{}".format(data_size))
    return model


def image_to_tens(image):
    return torch.Tensor(image)[None, None, :, :]


def encode_image(Z, ab_domain):
    height, width, _ = Z.shape
    final_ab = np.zeros((256, 256, 2))

    nr_colors = 100000
    Z__max_prob = np.argsort(-Z, axis=2)[:, :, :nr_colors]
    print(Z__max_prob.shape)

    for y in range(height):
        for x in range(width):
            non_zero_ab_domain_indices = np.nonzero(Z[y][x])[0]
            for prob_nr, ab_domain_idx in enumerate(non_zero_ab_domain_indices):
                #a_color, b_color = ab_domain[Z__max_prob[y][x][prob_nr]]
                a_color, b_color = ab_domain[ab_domain_idx]
                scalar = Z[y][x][ab_domain_idx]
                if scalar < 0 or a_color < 0 or b_color < 0:
                    print('Hallo')
                final_ab[y][x][0] += a_color * scalar
                final_ab[y][x][1] += b_color * scalar
                #final_ab[y][x][0] += a_color / nr_colors
                #final_ab[y][x][1] += b_color / nr_colors
                # if prob_nr >= nr_colors:
                #     break
    return final_ab


def H(Z, T, q_to_ab_dict):
    # plt.plot(Z[35, 60, :])
    # plt.show()

    def f_T(Z):
        Z = np.exp(np.log(Z) / T) / np.sum(np.exp(np.log(Z)) / T, axis=2)[:, :, np.newaxis]
        return Z
    Z = f_T(Z)


    # plt.plot(Z[35, 60, :])
    # plt.show()
    # kirra "argmean"
    # Z_mean = np.mean(Z, axis=2)

    ab = np.empty((256, 256, 2))

    # Det här är fel (blir som mode). Tar max bara för att testa =)
    # argmax = np.argmax(Z, axis=2)
    # ab = np.empty((256, 256, 2))
    # for i in range(Z.shape[0]):
    #     for j in range(Z.shape[1]):
    #         q = argmax[i, j]
    #         a, b = q_to_ab_dict[q]
    #         ab[i, j, 0], ab[i, j, 1] = a, b\

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            center = np.round(ndimage.measurements.center_of_mass(Z[i, j, :]))[0]
            a, b = q_to_ab_dict[center]
            ab[i, j, 0], ab[i, j, 1] = a, b
    return ab


def postprocess_output(l_original, Z_output_tens, ab_domain, q_to_ab_dict, T):
    Z_output_tens = torch.nn.functional.softmax(Z_output_tens, 1)
    Z_output = Z_output_tens.detach().numpy()
    Z_output = np.moveaxis(Z_output, 1, 3)[0]
    ab_output = H(Z_output, T, q_to_ab_dict)

    # ab_output = ab_output.astype(dtype=np.uint8)

    lab_output = np.empty((l_original.shape[0], l_original.shape[1], 3))
    lab_output[:, :, 0], lab_output[:, :, 1:] = l_original, ab_output
    rgb_output = color.lab2rgb(lab_output)
    return Z_output, lab_output, rgb_output


def plot_images(original_rgb, output_rgb, T):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    data = [('Original', original_rgb), ('Result (T={})'.format(T), output_rgb)]
    for ax, (title, img) in zip(axes, data):
        ax.set_title(title)
        ax.imshow(img)
        ax.axis('off')
    fig.tight_layout()
    plt.show()


def demo():
    # Parameters.
    data_size = 1000                                                                # TODO: Välj rätt size!
    T = 1.0

    # Load trained and saved CNN model.
    model = get_trained_CNN(data_size)

    # Load and pre-process demo image.
    original_image = np.asarray(Image.open(f"./data/Vladimir_Putin_0033.jpg"))
    l_original, ab_original = preprocess_image(original_image)
    l_original_tens = image_to_tens(l_original)

    # Load and get helpers.
    ab_to_q_dict_unsorted = pickle.load(open("pickles/ab_to_q_index_dict_unsorted.p", "rb"))
    ab_domain = get_ab_domain(data_size, ab_to_q_dict_unsorted)
    ab_to_q_dict = get_ab_to_q_dict(data_size, ab_domain)
    q_to_ab_dict = get_q_to_ab_dict(data_size, ab_to_q_dict)

    # Get prediction for demo image by CNN.
    Z_output_tens = model(l_original_tens).cpu()

    # Post-process prediction.
    Z_output, lab_output, rgb_output = postprocess_output(l_original, Z_output_tens, ab_domain, q_to_ab_dict, T)

    # Plot images and color-channels.
    plot_images(original_image, rgb_output, T)

    # Save the colored image.
    plt.imsave('out_img.png', rgb_output)


if __name__ == '__main__':
    demo()
