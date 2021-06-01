import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from helper_functions import *
from skimage import color

def plot_histogram(histogram_matrix, save_csv=False):
    hist_to_plot = histogram_matrix[:]

    print(hist_to_plot.shape)

    hist_to_plot[hist_to_plot == 0] = 1
    hist_to_plot = np.log(hist_to_plot)
    hist_to_plot[hist_to_plot == 0] = None

    xticks = yticks = [i for i in range(-110, 110, 10)]
    norm = matplotlib.colors.Normalize(0, 1)
    colors = [
        [norm(0), "#00009B"],
        [norm(1 / 8), "#0016FE"],
        [norm(2 / 8), "#008FFE"],
        [norm(3 / 8), "#1DFFD8"],
        [norm(4 / 8), "#81FF75"],
        [norm(5 / 8), "#E1FE15"],
        [norm(6 / 8), "#FF9A00"],
        [norm(7 / 8), "#FE2B01"],
        [norm(1), "#850000"]
    ]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    df = pd.DataFrame(hist_to_plot, columns=yticks, index=xticks)

    sns.set_style("whitegrid", {'grid.linestyle': '--'})
    ax = sns.heatmap(
        df,
        linewidth=0.0,
        cmap=cmap,
        xticklabels=55,
        yticklabels=55,
    )

    for _, spine in ax.spines.items():
        spine.set_visible(True)

    plt.yticks(rotation=0)
    plt.show()

    if save_csv:
        df.replace(to_replace=[np.nan], value=0, inplace=True)
        df.to_csv('ab_prob_dist.csv')


def plot_colors_in_use(ab_domain, num_images_used, L=50):
    xlabels = ylabels = [i for i in range(0, 221, 20)]
    xticks = yticks = [i for i in range(-110, 111, 20)]

    final_lab_image = np.zeros((221, 221, 3))

    for a_color, b_color in ab_domain:
        a_simple = (a_color + 110) // 10
        b_simple = (b_color + 110) // 10

        a_start = a_simple * 10
        a_end = (a_simple + 1) * 10
        b_start = b_simple * 10
        b_end = (b_simple + 1) * 10

        final_lab_image[a_start:a_end, b_start:b_end, 0] = L
        final_lab_image[a_start:a_end, b_start:b_end, 1] = a_color
        final_lab_image[a_start:a_end, b_start:b_end, 2] = b_color

    final_rgb_image = color.lab2rgb(final_lab_image)

    width, height, _ = final_rgb_image.shape

    for y in range(height):
        for x in range(width):
            r, g, b = final_rgb_image[y][x]
            if r == 0 and g == 0 and b == 0:
                final_rgb_image[y, x, :] = 1

    plt.imshow(final_rgb_image)
    plt.xticks(xlabels, xticks)
    plt.yticks(ylabels, yticks)
    plt.xlabel('$b$')
    plt.ylabel('$a$')
    plt.suptitle(f"   RGB(a, b | L = {L})")
    plt.rcParams["font.size"] = "8.0"
    plt.title(f"$Created$ $from$ ${num_images_used}$ $images$ | Q={247}")

    plt.savefig(f"./plot_results/colors_from_X_images/colors_from_{num_images_used}_images_with_L{L}.pdf")