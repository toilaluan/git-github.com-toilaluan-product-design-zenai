import os
import shutil
import numpy as np
from tqdm import tqdm
from controlnet_aux import CannyDetector, PidiNetDetector
from controlnet_aux.pidi import PidiNetDetector

pidinet = PidiNetDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

# pidi_detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
canny_detector = CannyDetector()

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y
    
def safer_memory(x):
    return np.ascontiguousarray(x.copy()).copy()


def resize_image_with_pad(input_image, resolution, skip_hwc3=False):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    interpolation = cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=interpolation)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode='edge')

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target])

    return safer_memory(img_padded), remove_pad

from typing import List
from PIL import Image, ImageDraw, ImageOps
import cv2
import numpy as np
import math

def prompting(main_object: str):
    prompt = f"{main_object}, textured, high quality, full detailed material, studio style, simple background, dslr, natural lighting, shot by camera, RAW image, photorealistic, sharp focus, 8k, uhd, file grain, masterpiece"
    negative_prompt = "deformed, animation, anime, cartoon, comic, cropped, out of frame, low res, draft, cgi, low quality render, thumbnail"
    return prompt, negative_prompt

def image_grid(images: List[Image.Image], col: int) -> Image.Image:
    row = math.ceil(len(images) / col)
    # Calculate the width and height of each cell in the grid
    cell_width = images[0].width
    cell_height = images[0].height

    # Create a new image to hold the grid
    grid_width = col * cell_width
    grid_height = row * cell_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    # Iterate through the images and paste them into the grid
    for i, image in enumerate(images):
        # Calculate the position (row, col) for the current image
        row_idx = i // col
        col_idx = i % col

        # Calculate the coordinates to paste the image into the grid
        paste_x = col_idx * cell_width
        paste_y = row_idx * cell_height

        # Paste the image into the grid
        grid_image.paste(image, (paste_x, paste_y))

    return grid_image

def resize_square(image: Image.Image, size: int, fill_color=(255, 255, 255)):
    """
    Resize and pad an image to make it square by resizing the longer side.

    Args:
        image (PIL.Image.Image): The input image.
        size (int): The desired size of the square image (width and height).
        fill_color (tuple, optional): The color to use for padding. Defaults to white (255, 255, 255).

    Returns:
        PIL.Image.Image: The square image.
    """
    # Get the dimensions of the input image
    width, height = image.size

    # Determine the longer side and calculate the scaling factor
    if width > height:
        scale_factor = size / width
    else:
        scale_factor = size / height

    # Calculate the new dimensions after resizing
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image using the calculated dimensions
    resized_image = image.resize((new_width, new_height), Image.BICUBIC)

    # Create a new image with the desired size and fill it with the specified color
    square_image = Image.new('RGB', (size, size), fill_color)

    # Calculate the position to paste the resized image in the center of the square
    x_offset = (size - new_width) // 2
    y_offset = (size - new_height) // 2

    # Paste the resized image onto the square canvas
    square_image.paste(resized_image, (x_offset, y_offset))

    return square_image

def preprocess_canny(image, preprocess_image_size, output_image_size):
    image, _ = resize_image_with_pad(np.array(image), preprocess_image_size)
    image = Image.fromarray(image)
    processed_image = canny_detector(image, detect_resolution=preprocess_image_size, image_resolution=output_image_size)
    processed_image = resize_square(processed_image, output_image_size, (0,0,0))
    return processed_image

def preprocess_sketch(image, preprocess_image_size, output_image_size):
    image, _ = resize_image_with_pad(np.array(image), preprocess_image_size)
    image = Image.fromarray(image)
    processed_image = pidinet(
       image, detect_resolution=preprocess_image_size, image_resolution=output_image_size, apply_filter=True
    )
    processed_image = resize_square(processed_image, output_image_size, (0,0,0))
    return processed_image

def preprocess_sketch_canny(image, preprocess_image_size, output_image_size):
    image, _ = resize_image_with_pad(np.array(image), preprocess_image_size)
    image = Image.fromarray(image)
    sketch_processed_image = pidi_detector(image, apply_filter=True, detect_resolution=preprocess_image_size, image_resolution=output_image_size)
    canny_processed_image = canny_detector(image, detect_resolution=preprocess_image_size, image_resolution=output_image_size, low_threshold=150)
    processed_image = np.array(sketch_processed_image) + np.array(canny_processed_image)
    processed_image = (processed_image > 0)*255
    processed_image = np.array(processed_image, dtype=np.uint8)
    processed_image = Image.fromarray(processed_image) 
    processed_image = resize_square(processed_image, output_image_size, (0,0,0))
    return processed_image