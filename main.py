import cv2
import matplotlib.pyplot as plt


IMAGE_PATHS = {
    "Portrait": "images/portrait.jpg",
    "City": "images/city.jpg",
    "Text": "images/text.jpg"
}


def load_images(paths):
    images = {}

    for name, path in paths.items():
        image = cv2.imread(path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")

        images[name] = image

    return images


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def show_originals(images):
    plt.figure(figsize=(15, 5))

    for index, (name, image) in enumerate(images.items(), start=1):
        plt.subplot(1, 3, index)
        plt.imshow(bgr_to_rgb(image))
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_color_spaces(name, image):
    rgb = bgr_to_rgb(image)
    grayscale = to_grayscale(image)
    hsv = to_hsv(image)
    h, s, v = cv2.split(hsv)

    items = [
        ("RGB", rgb, None),
        ("Grayscale", grayscale, "gray"),
        ("HSV Hue", h, "gray"),
        ("HSV Saturation", s, "gray"),
        ("HSV Value", v, "gray")
    ]

    plt.figure(figsize=(18, 4))

    for index, (title, item, cmap) in enumerate(items, start=1):
        plt.subplot(1, 5, index)
        plt.imshow(item, cmap=cmap)
        plt.title(f"{name}: {title}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    images = load_images(IMAGE_PATHS)

    show_originals(images)

    for name, image in images.items():
        show_color_spaces(name, image)


if __name__ == "__main__":
    main()