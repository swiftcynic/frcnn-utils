from math import ceil
import PIL
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from .constants import PEST_CLASSES, DEVICE


# Function to add bounding boxes and labels to an image
def add_boxes_labels(image, boxes, labels, scores=None, threshold=0.75):
    img = T.ToPILImage()(image).convert("RGB")
    
    scores = torch.zeros(
        len(boxes),
        device=DEVICE,
        dtype=torch.float32,
        requires_grad=False
    ) if not scores else scores

    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = map(int, box)
        fill_color = (0, 70, 0) # Default green color for boxes
        if score:
            if score < threshold:
                fill_color = (150, 0, 0)
            else:
                fill_color = (0, 70, 0)

        PIL.ImageDraw.Draw(img).rectangle([(xmin, ymin), (xmax, ymax)], outline=fill_color, width=3)

        font = PIL.ImageFont.load_default(size=14)
        ascent, descent = font.getmetrics()

        if score:

            annotation_label = f"{PEST_CLASSES[label-1]} {score:.2f}"
        else:
            annotation_label = f"{PEST_CLASSES[label-1]}"
        
        (width, baseline), (offset_x, offset_y) = font.font.getsize(annotation_label)
        text_height = ascent + descent
        text_width = width + 20

        # Determine text and background rectangle position
        if not score:
            if ymin - text_height - baseline >= 0:
                text_position = (xmin + 10, ymin - text_height)
                back_box_position = [(xmin, ymin - text_height), (xmin + text_width, ymin)]
            else:
                text_position = (xmin + 10, ymin)
                back_box_position = [(xmin, ymin), (xmin + text_width, ymin + text_height)]
        else:
            if xmax - text_width < 0:
                text_position = (xmax + 10, ymin)
                back_box_position = [(xmax, ymin), (xmax + text_width, ymin + text_height)]
            else:
                text_position = (xmax - text_width + 10, ymax - text_height)
                back_box_position = [(xmax - text_width, ymax - text_height), (xmax, ymax)]
        
        # Draw background rectangle first, then text immediately
        PIL.ImageDraw.Draw(img).rectangle(back_box_position, fill=fill_color)
        PIL.ImageDraw.Draw(img).text(text_position, annotation_label, fill="white", font=font)

    return img

# Function to display image with or without bounding boxes
def plot_images(images, targets, cmap=None, suptitle=None, title_y=1.0, fontsize=14, threshold=0.75):
    images_count = len(images)

    if images_count == 1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        if suptitle:
            plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=title_y)

        img = add_boxes_labels(
            images[0],
            targets[0]['boxes'],
            targets[0]['labels'],
            targets[0]['scores'] if 'scores' in targets[0] else None
        )

        ax.imshow(img, cmap=cmap)
        ax.set_title(f'Sample 1', fontsize=fontsize)
        ax.axis('off')
        plt.tight_layout()
        return
    
    row_count = ceil(images_count / 3)
    col_count = min(images_count, 3)
    width = 4*col_count
    height = 4*(row_count)

    fig, axs = plt.subplots(
        row_count,
        col_count,
        figsize=(width, height),
        squeeze=False,
        layout='constrained'
    )

    if suptitle:
        plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=title_y)

    for (i, image), ax in zip(enumerate(images), axs.flat[:images_count]):
        if len(targets[i]['labels']) == 0:
            img = T.ToPILImage()(image).convert("RGB")
            img = PIL.ImageOps.expand(img, border=5, fill='red')

        else:
            img = add_boxes_labels(
                image,
                targets[i]['boxes'],
                targets[i]['labels'],
                targets[i]['scores'] if 'scores' in targets[i] else None,
                threshold=threshold
            )
        ax.imshow(img, cmap=cmap)
        ax.set_title(f'Sample {i+1}', fontsize=14)

    for ax in axs.flat: ax.axis('off')

    plt.tight_layout()