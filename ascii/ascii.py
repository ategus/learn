import cv2


def load_image(image_path):
    # Load an image in grayscale
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def resize_image(image, new_width=100):
    # Compute the ratio of the new width to the old width, and adjust height accordingly
    ratio = new_width / image.shape[1]
    new_height = int(image.shape[0] * ratio)
    return cv2.resize(image, (new_width, new_height))


def pixel_to_ascii(image):
    # Map grayscale values to ASCII characters
    ascii_chars = "@%#*+=-:. "
    new_image = []
    for row in image:
        new_image.append(''.join([ascii_chars[pixel//32] for pixel in row]))
    return "\n".join(new_image)


if __name__ == "__main__":
    image_path_ = input("Enter the path to the image file: ")
    image_ = load_image(image_path_)
    if image_ is None:
        print("Error loading image.")
    else:
        image_ = resize_image(image_)
        ascii_art = pixel_to_ascii(image_)
        print(ascii_art)

import cv2


def load_image(image_path):
    # Load an image in grayscale
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def resize_image(image, new_width=100):
    # Compute the ratio of the new width to the old width, and adjust height accordingly
    ratio = new_width / image.shape[1]
    new_height = int(image.shape[0] * ratio)
    return cv2.resize(image, (new_width, new_height))


def pixel_to_ascii(image):
    # Map grayscale values to ASCII characters
    ascii_chars = "@%#*+=-:. "
    new_image = []
    for row in image:
        new_image.append(''.join([ascii_chars[pixel//32] for pixel in row]))
    return "\n".join(new_image)


if __name__ == "__main__":
    image_path_ = input("Enter the path to the image file: ")
    image_ = load_image(image_path_)
    if image_ is None:
        print("Error loading image.")
    else:
        image_ = resize_image(image_)
        ascii_art = pixel_to_ascii(image_)
        print(ascii_art)

