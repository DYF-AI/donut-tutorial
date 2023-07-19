import numpy as np
from PIL import Image

def load_image(image_path:str, size=None):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if size is not None:
        # resize image
        image = image.resize((size, size))
        image = np.asarray(image)
        image = image[:, :, ::-1]  # flip color channels from RGB to BGR
        image = image.transpose(2, 0, 1)  # move channels to first dimension
    return image, (w, h)
