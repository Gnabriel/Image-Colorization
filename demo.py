from main import *


def preprocess_demo_image(l_image):
    l_image_tensor = torch.Tensor(l_image)[None, None, :, :]
    return l_image_tensor


def postprocess_tens(l_tens, out_ab):
    out_lab_tens = torch.cat((l_tens, out_ab), dim=1)
    out_lab = out_lab_tens.data.cpu().numpy()[0, ...].transpose((1, 2, 0))
    out_rgb = color.lab2rgb(out_lab)
    return out_lab, out_rgb


def demo():
    model = ColorizationNet()
    model.load_state_dict(torch.load('./colorize_cnn.pth'))
    model.eval()

    image = np.asarray(Image.open(f"./data/lfw/Aaron_Eckhart/Aaron_Eckhart_0001.jpg"))
    l_image, ab_image = preprocess_image(image)
    l_tens = preprocess_demo_image(l_image)

    out_ab = model(l_tens).cpu()
    out_lab, out_rgb = postprocess_tens(l_tens, out_ab)
    print(out_ab[:, :, 0])
    print(out_ab[:, :, 1])
    # Original image rgb2lab2rgb
    a_tens = preprocess_demo_image(ab_image[:, :, 0])
    b_tens = preprocess_demo_image(ab_image[:, :, 1])
    ab_tens = torch.cat((a_tens, b_tens), dim=1)
    original_lab, original_rgb = postprocess_tens(l_tens, ab_tens)

    # Save the colored image.
    plt.imsave('out_img.png', out_rgb)

    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    plt.show()

    plt.imshow(l_image, cmap='gray')
    plt.title('Original (L channel)')
    plt.axis('off')
    plt.show()

    # plt.imshow(out_lab[:, :, 0], cmap='gray')
    # plt.title('Out channel 0')
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(out_lab[:, :, 1])
    # plt.title('Out channel 1')
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(out_lab[:, :, 2])
    # plt.title('Out channel 2')
    # plt.axis('off')
    # plt.show()

    plt.imshow(out_rgb)
    plt.title('Output (perfectly AI colored RGB)')
    plt.axis('off')
    plt.show()

    plt.imshow(original_rgb)
    plt.title('Original rgb2lab2rgb')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    demo()
