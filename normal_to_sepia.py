import os
import shutil
from PIL import Image


def apply_sepia(image_path):
    _image = Image.open(image_path)
    width, height = _image.size
    pixels = _image.load()

    for py in range(height):
        for px in range(width):
            r, g, b = _image.getpixel((px, py))

            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            if tr > 255:
                tr = 255

            if tg > 255:
                tg = 255

            if tb > 255:
                tb = 255

            pixels[px, py] = (tr, tg, tb)

    return _image


if __name__ == '__main__':
    source_directory = 'normal'
    destination_directory = 'sepia'

    for filename in os.listdir(source_directory):
        source_file = os.path.join(source_directory, filename)
        if os.path.isfile(source_file) and any(
                filename.lower().endswith(image_extension) for image_extension in ['.jpg', '.jpeg', '.png', '.gif']):
            processed_image = apply_sepia(source_file)
            destination_file = os.path.join(destination_directory, os.path.basename(source_file))
            processed_image.save(destination_file)
