import pickle
from helper_functions import *
from scipy.ndimage import gaussian_filter


class ColorizerTool:
    def __init__(self, images_limit=13233, height=256, width=256):
        self.ab_color_to_possible_color_idx = {}  # {"(3, 2)": 1, "(1, 1)": 0, "(2, 0)": 2}
        self.possible_color_idx_to_ab_color = {}  # {1: "(3, 2)", 0: "(1, 1)", 2: "(2, 0)"}
        self.Q = 0
        self.Z = None
        self.Z_hat = None
        self.height = height
        self.width = width
        self.p = np.zeros((221, 221))
        self.images_limit = images_limit

    # def set_possible_colors(self, save_files=False):
    #     print()
    #     print("--- Set Possible Colors ---")
    #
    #     all_people = os.listdir("./data")[:self.images_limit]
    #
    #     for idx, image_on_person in enumerate(all_people):
    #         print(f"set image {idx+1}/{self.images_limit}")
    #         rgb_image = np.array(Image.open(f"./data/{image_on_person}"))
    #
    #         l_image, ab_image = preprocess_image(rgb_image)
    #         discretized_ab_image = discretize_image(ab_image)
    #
    #         for y in range(self.height):
    #             for x in range(self.width):
    #                 a_color = discretized_ab_image[y][x][0]
    #                 b_color = discretized_ab_image[y][x][1]
    #
    #                 self.p[a_color][b_color] += 1
    #                 color_key = f"({a_color}, {b_color})"
    #
    #                 if color_key not in self.ab_color_to_possible_color_idx:
    #                     new_possible_colors_idx = len(self.possible_colors)
    #                     self.possible_colors.append(color_key)
    #                     self.ab_color_to_possible_color_idx[color_key] = new_possible_colors_idx
    #                     self.possible_color_idx_to_ab_color[new_possible_colors_idx] = color_key
    #
    #     tot_ab_values = self.images_limit * self.height * self.width
    #     self.p[self.p > 0] /= tot_ab_values
    #     self.Q = len(self.possible_colors)
    #
    #     if save_files:
    #         pickle.dump(self.p, open("pickles/p_matrix.p", "wb"))
    #         pickle.dump(self.ab_color_to_possible_color_idx, open("pickles/ab_color_to_possible_color_idx.p", "wb"))

    def set_possible_colors(self, save_files=False):
        print()
        print("--- Set Possible Colors ---")

        all_people = os.listdir("./data")[:self.images_limit]

        color_index = 0
        for idx, image_on_person in enumerate(all_people):
            if idx % 50 == 49:
                print(f"set image {idx+1}/{self.images_limit}")
            rgb_image = np.array(Image.open(f"./data/{image_on_person}"))

            l_image, ab_image = preprocess_image(rgb_image)
            discretized_ab_image = discretize_image(ab_image)

            for y in range(self.height):
                for x in range(self.width):
                    a_color = discretized_ab_image[y][x][0]
                    b_color = discretized_ab_image[y][x][1]

                    self.p[a_color][b_color] += 1
                    color_key = f"({a_color}, {b_color})"

                    if self.ab_color_to_possible_color_idx.setdefault(color_key, color_index) is color_index:
                        color_index += 1

        self.p[self.p > 0] /= self.images_limit * self.height * self.width

        self.Q = len(self.ab_color_to_possible_color_idx.keys())

        self.possible_color_idx_to_ab_color = {v: k for k, v in self.ab_color_to_possible_color_idx.items()}

        if save_files:
            pickle.dump(self.p, open("pickles/p_matrix.p", "wb"))
            pickle.dump(self.ab_color_to_possible_color_idx, open("pickles/ab_to_q_index_dict.p", "wb"))
            pickle.dump(self.possible_color_idx_to_ab_color, open("pickles/q_index_to_ab_dict.p", "wb"))

    def set_Z_hat(self, save_files=False):
        self.Z_hat = np.zeros((self.width, self.height, self.Q))
        all_people = os.listdir("./data")[:self.images_limit]

        for idx, image_on_person in enumerate(all_people):
            rgb_image = np.array(Image.open(f"./data/{image_on_person}"))
            _, ab_image = preprocess_image(rgb_image)

            discretized_ab_image = discretize_image(ab_image)

            for y in range(self.height):
                for x in range(self.width):
                    a_color = discretized_ab_image[y][x][0]
                    b_color = discretized_ab_image[y][x][1]
                    color_key = f"({a_color}, {b_color})"
                    possible_color_idx = self.ab_color_to_possible_color_idx[color_key]
                    self.Z_hat[y][x][possible_color_idx] += 1

        self.Z_hat /= self.images_limit

        if save_files:
            pickle.dump(self.Z_hat, open("pickles/z_hat.p", "wb"))

    def get_Z_for_one_image(self, ab_image):
        Z = np.zeros((self.height, self.width, self.Q))
        discretized_ab_image = discretize_image(ab_image)

        for y in range(self.height):
            for x in range(self.width):
                a_color = discretized_ab_image[y][x][0]
                b_color = discretized_ab_image[y][x][1]
                color_key = get_color_key(a_color, b_color)

                true_color_idx = self.ab_color_to_possible_color_idx[color_key]
                Z[y][x][true_color_idx] = 1

        return Z

    def get_weighted_Z(self, Z, lamda=0.5, sigma=5, save_files=False):
        weighted_Z = np.zeros((self.height, self.width))

        p_tilde = gaussian_filter(self.p, sigma=sigma)

        term1 = (1-lamda)*p_tilde
        term2 = lamda / self.Q
        p_with_new_distribution = (term1 + term2) ** -1

        for y in range(self.height):
            for x in range(self.width):
                true_color_idx = np.argmax(Z[y][x])
                color_key = self.possible_color_idx_to_ab_color[true_color_idx]
                q_star_a, q_star_b = get_ab_colors_from_key(color_key)
                weighted_Z[y][x] = p_with_new_distribution[q_star_a][q_star_b]

        if save_files:
            pickle.dump(self.Z_hat, open("pickles/weighted_Z.p", "wb"))

        return weighted_Z

    def multinomial_cross_entropy_loss(self, Z):
        weighted_Z = self.get_weighted_Z(Z)

        q_sum = Z * np.log(self.Z_hat, where=(self.Z_hat != 0))
        q_sum = np.sum(q_sum, axis=2)
        total_sum = np.sum(weighted_Z * q_sum)

        loss = total_sum * -1

        return loss


def main():
    all_people = os.listdir("./data")[:1]
    rgb_image = np.array(Image.open(f"./data/{all_people[0]}"))

    number_of_images_to_use = 20  # max=13233

    colorizer_tool = ColorizerTool()
    colorizer_tool.set_possible_colors(save_files=True)
    colorizer_tool.set_Z_hat(save_files=True)
    colorizer_tool.get_weighted_Z(save_files=True)

    # l_images, ab_images = get_images(number_of_images_to_use)
    # Z1 = colorizer_tool.get_Z_for_one_image(ab_images[0])
    # Z1_loss = colorizer_tool.multinomial_cross_entropy_loss(Z1)

    # print(f"Z1_loss = {Z1_loss}")

    def ab_to_q_index(a_color, b_color):
        color_key = f"({a_color}, {b_color})"
        return colorizer_tool.ab_color_to_possible_color_idx[color_key]

    # discrete_color_to_q_index = np.vectorize(ab_to_q_index)

    # print(colorizer_tool.ab_color_to_possible_color_idx)


    # print( discrete_color_to_q_index(discretized_ab_image[:,:,0], discretized_ab_image[:,:,1]) )
    # plot_colors_in_use(colorizer_tool.possible_colors, number_of_images_to_use)

    # Z1 = colorizer_tool.get_Z_for_one_image(ab_images[0])
    # weighted_Z1 = colorizer_tool.get_weighted_Z(Z1)

    # print(weighted_Z1)

if __name__ == "__main__":
    main()