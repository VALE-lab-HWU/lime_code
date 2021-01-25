from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

import matplotlib.pyplot as plt

from skimage.color import label2rgb


def get_explainer():
    explainer = lime_image.LimeImageExplainer()
    segmenter = SegmentationAlgorithm(
        'quickshift', kernel_size=1, max_dist=200, ratio=0.2)
    return explainer, segmenter


def build_img_explainer(ax, explanation, label, title, **kwargs):
    temp, mask = explanation.get_image_and_mask(
        label, num_features=10, hide_rest=False, min_weight=0.01, **kwargs)
    ax.imshow(label2rgb(mask, temp, bg_label=0, bg_color="white"),
              interpolation='nearest')
    ax.set_title('{} {}'.format(title, label))


def visualize_explanation(explanation, label):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    build_img_explainer(ax1, explanation, label,
                        'Positive Regions for',
                        positive_only=True)
    build_img_explainer(ax2, explanation, label,
                        'Positive/Negative Regions for',
                        positive_only=False)
