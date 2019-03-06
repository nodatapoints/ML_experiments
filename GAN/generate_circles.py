import numpy as np
import idx2numpy
from PIL import Image, ImageDraw, ImageFilter
from random import random

size = width, height = 20, 20
max_radius = 10

n = 10000

output_file = 'circles.idx'


def generate_random_circle():
    image = Image.new('L', size)  # grayscale
    draw = ImageDraw.Draw(image)

    radius = 5 + max_radius * random()

    x, y = (width-radius)*random(), (height-radius)*random()
    draw.ellipse((x, y, x+radius, y+radius), outline=255)

    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    return np.array(image.getdata(), dtype=np.uint8).reshape(size)


data = np.array([generate_random_circle() for _ in range(n)])
idx2numpy.convert_to_file(output_file, data)
