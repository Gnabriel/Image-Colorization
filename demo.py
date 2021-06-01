import matplotlib.pyplot as plt
from scipy import ndimage

from main import *


def get_trained_CNN(data_size):
    model = ColorizationNet()
    model.load_state_dict(torch.load('./colorize_cnn_{}_bra_skit.pth'.format(data_size)))
    # model.load_state_dict(torch.load('./colorize_cnn_{}_bra_skit.pth'.format(data_size)))
    model.eval()
    print("Loaded CNN: colorize_cnn_{}".format(data_size))
    return model


def image_to_tens(image):
    return torch.Tensor(image)[None, None, :, :]


def H(Z, T, ab_domain):
    # plot_q_probabilities(Z[35, 60, :], Z[128, 128, :])

    def f_T(Z):
        Z = np.exp(np.log(Z) / T) / np.sum(np.exp(np.log(Z)) / T, axis=2)[:, :, np.newaxis]
        return Z
    Z = f_T(Z)

    # Minmax_scale
    # Z_std = (Z - Z.min(axis=2)[:, :, np.newaxis]) / (Z.max(axis=2) - Z.min(axis=2))[:, :, np.newaxis]
    # Z = Z_std * (1 - 0) + 0

    Z = Z / np.sum(Z, axis=2)[:, :, np.newaxis]
    Z = Z * (3/np.exp(T))      # Higher saturation

    ab_domain = np.array(ab_domain)
    final_ab = np.sum(Z[:, :, :, np.newaxis] * ab_domain[np.newaxis, np.newaxis, :, :], axis=2)
    return final_ab


def H_old(Z, T, q_to_ab_dict):
    plot_q_probabilities(Z[35, 60, :], Z[128, 128, :])

    def f_T(Z):
        Z = np.exp(np.log(Z) / T) / np.sum(np.exp(np.log(Z)) / T, axis=2)[:, :, np.newaxis]
        return Z
    Z = f_T(Z)
    plot_q_probabilities(Z[35, 60, :], Z[128, 128, :])

    Z = Z / np.sum(Z, axis=2)[:, :, np.newaxis]

    ab = np.empty((256, 256, 2))

    # Det här är fel (blir som mode). Tar max bara för att testa =)
    # argmax = np.argmax(Z, axis=2)
    # for i in range(Z.shape[0]):
    #     for j in range(Z.shape[1]):
    #         q = argmax[i, j]
    #         a, b = q_to_ab_dict[q]
    #         ab[i, j, 0], ab[i, j, 1] = a, b

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            center = ndimage.measurements.center_of_mass(Z[i, j, :])
            center = np.round(center[0])
            a, b = q_to_ab_dict[center]
            ab[i, j, 0], ab[i, j, 1] = a, b
    return ab


def postprocess_output(l_original, Z_output_tens, ab_domain, q_to_ab_dict, T):
    # Z_output_tens = torch.nn.functional.softmax(Z_output_tens, 1)
    Z_output = Z_output_tens.detach().numpy()
    Z_output = np.moveaxis(Z_output, 1, 3)[0]
    # ab_output = H_old(Z_output, T, q_to_ab_dict, ab_domain)
    ab_output = H(Z_output, T, ab_domain)

    # ab_output = ab_output.astype(dtype=np.uint8)

    lab_output = np.empty((l_original.shape[0], l_original.shape[1], 3))
    lab_output[:, :, 0], lab_output[:, :, 1:] = l_original, ab_output

    rgb_output = lab_to_rgb(lab_output)
    return Z_output, lab_output, rgb_output


def lab_to_rgb(lab):
    rgb = color.lab2rgb(lab)        # Using scikit-image library.
    # srgb_profile = ImageCms.createProfile("sRGB")
    # lab_profile = ImageCms.createProfile("LAB")
    # lab2rgb_trans = ImageCms.buildTransformFromOpenProfiles(lab_profile, srgb_profile, "LAB", "RGB")
    # lab_pillow = Image.fromarray((lab).astype('uint8'), 'LAB')
    # rgb = ImageCms.applyTransform(lab_pillow, lab2rgb_trans)
    return rgb


def plot_q_probabilities(q1, q2):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    data = [('q1', q1), ('q2', q2)]
    for ax, (title, img) in zip(axes, data):
        ax.set_title(title)
        ax.plot(img)
        ax.axis('off')
    fig.tight_layout()
    plt.show()


def plot_ab_channels(a, b, path, save_image):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    data = [('A', a), ('B', b)]
    for ax, (title, img) in zip(axes, data):
        ax.set_title(title)
        ax.imshow(img, vmin=-110, vmax=110, cmap='Greys')
        ax.axis('off')
    fig.tight_layout()
    if save_image:
        plt.savefig(path)
    plt.show()


def plot_images(original_rgb, output_rgb, T, path, save_image):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    data = [('Original', original_rgb), ('Result (T={})'.format(T), output_rgb)]
    for ax, (title, img) in zip(axes, data):
        ax.set_title(title)
        ax.imshow(img)
        ax.axis('off')
    fig.tight_layout()
    if save_image:
        plt.savefig(path)
    plt.show()


def plot_losses(data_size, path, save_images):
    losses = pickle.load(open("pickles/losses_{}_mse.p".format(data_size), "rb"))
    losses = np.array(losses).flatten()
    plt.plot(losses)
    plt.ylabel("Loss")
    plt.xlabel("Update step")
    if save_images:
        plt.savefig(path)
    else:
        plt.show()


def experiment(data_size, model, num_images, T=0.6):
    _, test_X, _, test_Y = pickle.load(open("pickles/train_test_{}.p".format(data_size), "rb"))

    # Load and get helpers.
    ab_to_q_dict_unsorted = pickle.load(open("pickles/ab_to_q_index_dict_unsorted.p", "rb"))
    ab_domain = get_ab_domain(data_size, ab_to_q_dict_unsorted)
    ab_to_q_dict = get_ab_to_q_dict(data_size, ab_domain)
    q_to_ab_dict = get_q_to_ab_dict(data_size, ab_to_q_dict)

    for i in range(num_images):
        l_original, ab_original = test_X[i], test_Y[i]
        l_original_tens = image_to_tens(l_original)

        # Get prediction for demo image by CNN.
        Z_output_tens = model(l_original_tens).cpu()

        # Post-process prediction.
        Z_output, lab_output, rgb_output = postprocess_output(l_original, Z_output_tens, ab_domain, q_to_ab_dict, T)

        # Save the grayscale image.
        plt.imsave('experiment/original_cnn/grayscale_{}.png'.format(i+1), l_original, cmap='gray')

        # Save the colored output image.
        plt.imsave('experiment/original_cnn/out_{}.png'.format(i+1), rgb_output)

        # Save the original image.
        lab_original = np.empty((l_original.shape[0], l_original.shape[1], 3))
        lab_original[:, :, 0], lab_original[:, :, 1:] = l_original, ab_original
        rgb_original = lab_to_rgb(lab_original)
        plt.imsave('experiment/original_cnn/original_{}.png'.format(i+1), rgb_original)

        print("Image {} done and saved.".format(i+1))


def demo():
    # Parameters.
    data_size = 13233           # 13 233
    # T = 0.38
    # T_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.38]
    T_steps = [0.6]
    save_images = True

    # Load trained and saved CNN model.
    model = get_trained_CNN(data_size)

    # experiment(data_size, model, num_images=50)
    # exit("Experiment done")

    for T in T_steps:
        # Load and pre-process demo image.
        # original_image = np.asarray(Image.open(f"./data/Vladimir_Putin_0033.jpg"))
        # original_image = np.asarray(Image.open(f"./data/Abdoulaye_Wade_0002.jpg"))
        # original_image = np.asarray(Image.open(f"./data/Mary_Elizabeth_Mastrantonio_0001.jpg"))
        # original_image = np.asarray(Image.open(f"./data/Pascal_Affi_Nguessan_0001.jpg"))
        # original_image = np.asarray(Image.open(f"./data/Pascal_Quignard_0003.jpg"))
        # original_image = np.asarray(Image.open(f"./data/Ai_Sugiyama_0004.jpg"))
        # original_image = np.asarray(Image.open(f"./data/Saddam_Hussein_0009.jpg"))
        original_image = np.asarray(Image.open(f"./data_imgnet/gabbe_glad.png"))[:, :, :3]
        # original_image = np.asarray(Image.open(f"./data_imgnet/ILSVRC2012_val_00005026.jpeg"))  # lemurer
        # original_image = np.asarray(Image.open(f"./data_imgnet/ILSVRC2012_val_00005002.jpeg"))  # hamster
        # original_image = np.asarray(Image.open(f"./data_imgnet/ILSVRC2012_val_00005006.jpeg"))  # tiger
        # original_image = np.asarray(Image.open(f"./data_imgnet/ILSVRC2012_val_00005011.jpeg"))  # tiger
        # original_image = np.asarray(Image.open(f"./data_imgnet/gabe.jpg"))  # tony the tiger
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
        plot_ab_channels(lab_output[:, :, 1], lab_output[:, :, 2], "image_results/ab_channels_{}.png".format(T), save_images)
        plot_images(original_image, rgb_output, T, "image_results/original_vs_colored_{}.png".format(T), save_images)

        # Plot losses.
        # plot_losses(data_size, "image_results/losses.png", save_images)

        # Save the grayscale image.
        if save_images:
            plt.imsave('image_results/grayscale.png', l_original, cmap='gray')

        # Save the colored image.
        if save_images:
            plt.imsave('image_results/out_img_{}.png'.format(T), rgb_output)


if __name__ == '__main__':
    demo()
