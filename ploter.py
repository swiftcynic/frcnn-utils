from math import ceil
import PIL
import matplotlib.pyplot as plt

import torchvision.transforms as T
from .constants import PEST_CLASSES, DEVICE


# Function to add bounding boxes and labels to an image
def add_boxes_labels(image, boxes, labels):
    img = T.ToPILImage()(image).convert("RGB")

    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = map(int, box)
        PIL.ImageDraw.Draw(img).rectangle([(xmin, ymin), (xmax, ymax)], outline="black", width=3)

        font = PIL.ImageFont.load_default(size=24)
        ascent, descent = font.getmetrics()

        annotation_label = PEST_CLASSES[label-1]
        (width, baseline), (offset_x, offset_y) = font.font.getsize(annotation_label)
        text_height = ascent + descent
        text_width = width + 20

        if ymin - text_height - baseline >= 0:
            PIL.ImageDraw.Draw(img).rectangle([(xmin, ymin - text_height), (xmin + text_width, ymin)], fill="black")
            PIL.ImageDraw.Draw(img).text((xmin+10, ymin - text_height), annotation_label, fill="white", font=font)
        else:
            PIL.ImageDraw.Draw(img).rectangle([(xmin, ymin), (xmin + text_width, ymin + text_height)], fill="black")
            PIL.ImageDraw.Draw(img).text((xmin+10, ymin), annotation_label, fill="white", font=font)

    return img

# Function to display image with or without bounding boxes
def plot_images(images, targets, cmap=None, suptitle=None, title_y=1.0, fontsize=14):
    images_count = len(images)

    if images_count == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        if suptitle:
            plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=title_y)

        img = add_boxes_labels(images[0], targets[0]['boxes'], targets[0]['labels'])

        ax.imshow(img, cmap=cmap)
        ax.set_title(f'Sample 1', fontsize=fontsize)
        ax.axis('off')
        plt.tight_layout()
        return
    
    row_count = ceil(images_count / 3)
    col_count = min(images_count, 3)
    width = 5*col_count - (5-col_count if col_count < 5 else 0)
    height = 4*(row_count)

    if suptitle:
        fig, axs = plt.subplots(row_count, col_count, figsize=(width, height+5), squeeze=False)
        plt.suptitle(suptitle, fontsize=18, fontweight='bold')
    else:
        fig, axs = plt.subplots(row_count, col_count, figsize=(width, height), squeeze=False)

    for (i, image), ax in zip(enumerate(images), axs.flat[:images_count]):
        img = add_boxes_labels(image, targets[i]['boxes'], targets[i]['labels'])
        ax.imshow(img, cmap=cmap)
        ax.set_title(f'Sample {i+1}', fontsize=14)

    for ax in axs.flat: ax.axis('off')
    
    plt.tight_layout()