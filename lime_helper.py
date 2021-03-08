from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

import numpy as np
import matplotlib.pyplot as plt

from skimage.color import label2rgb


# create the lime explainer and the segmenter use for an explanation
def get_explainer():
    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm(
        'quickshift', kernel_size=1, max_dist=200, ratio=0.2)
    return explainer, segmenter


# build the images with positive & positive/negative region for an explanation
# tied to the visualize_explanation function
def build_img_explainer(ax, explanation, label, title, **kwargs):
    temp, mask = explanation.get_image_and_mask(
        label, num_features=10, hide_rest=False, min_weight=0.01, **kwargs)
    ax.imshow(label2rgb(mask, temp, bg_label=0, bg_color="white"),
              interpolation='nearest')
    ax.set_title('{} {}'.format(title, label))


# vizualize an explanation
# take an explanation from lime and the label
def visualize_explanation(explanation, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    build_img_explainer(ax1, explanation, label,
                        'Positive Regions for',
                        positive_only=True)
    build_img_explainer(ax2, explanation, label,
                        'Positive/Negative Regions for',
                        positive_only=False)
    return fig


def plot_lime_pca(imgs, title=None):
    if title is None:
        title = [['Image', 'Image reduced with PCA'],
                 ['Masked Image', 'Masked image reduced with PCA']]
    fig, axs = plt.subplots(len(imgs[0]), len)
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i][j].imshow(imgs[i][j])
    return fig, axs


def make_plot_lime_projection(explanation, pipeline):
    image, mask_0, _, mask_1 = (*explanation.get_image_and_mask(0),
                                *explanation.get_image_and_mask(1))
    image = pipeline[0].transform([image]).reshape(128, 128)
    pca_img = pipeline[1].transform([image.reshape(-1)]).reshape(-1)
    mask = (mask_0 + mask_1).clip(0, 1)
    masked_img = image * mask
    pca_masked = pipeline[1].transform(masked_img.reshape(1, -1)).reshape(-1)
    plot_lime_pca(np.array([[image, 10*[pca_img]],
                            [masked_img, 10*[pca_masked]]], dtype=object))
