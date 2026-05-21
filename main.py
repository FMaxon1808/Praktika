import cv2
import numpy as np
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


def show_grid(title, items, rows, cols, figsize=(16, 10)):
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16)

    for index, (item_title, image, cmap) in enumerate(items, start=1):
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=cmap)
        plt.title(item_title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def show_originals(images):
    items = []

    for name, image in images.items():
        items.append((name, bgr_to_rgb(image), None))

    show_grid("Original images", items, 1, 3, (15, 5))


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

    show_grid(f"Color spaces: {name}", items, 1, 5, (18, 4))


def show_smoothing(image):
    kernels = [5, 11, 21]
    items = []

    for kernel in kernels:
        gaussian = cv2.GaussianBlur(image, (kernel, kernel), 0)
        median = cv2.medianBlur(image, kernel)
        average = cv2.blur(image, (kernel, kernel))

        items.append((f"Gaussian {kernel}x{kernel}", bgr_to_rgb(gaussian), None))
        items.append((f"Median {kernel}x{kernel}", bgr_to_rgb(median), None))
        items.append((f"Average {kernel}x{kernel}", bgr_to_rgb(average), None))

    show_grid("Smoothing filters", items, 3, 3, (15, 12))


def show_sharpening(image):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharpened = cv2.filter2D(image, -1, kernel)

    items = [
        ("Before", bgr_to_rgb(image), None),
        ("After", bgr_to_rgb(sharpened), None)
    ]

    show_grid("Sharpening", items, 1, 2, (12, 5))


def show_morphology(image):
    gray = to_grayscale(image)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    items = [("Binary", binary, "gray")]

    for size in [3, 5, 9]:
        kernel = np.ones((size, size), np.uint8)

        erosion = cv2.erode(binary, kernel, iterations=1)
        dilation = cv2.dilate(binary, kernel, iterations=1)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        items.append((f"Erosion {size}x{size}", erosion, "gray"))
        items.append((f"Dilation {size}x{size}", dilation, "gray"))
        items.append((f"Opening {size}x{size}", opening, "gray"))
        items.append((f"Closing {size}x{size}", closing, "gray"))

    show_grid("Morphology", items, 4, 4, (18, 16))


def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.03):
    noisy = image.copy()
    total_pixels = image.shape[0] * image.shape[1]

    salt_count = int(total_pixels * amount / 2)
    pepper_count = int(total_pixels * amount / 2)

    salt_y = np.random.randint(0, image.shape[0], salt_count)
    salt_x = np.random.randint(0, image.shape[1], salt_count)

    pepper_y = np.random.randint(0, image.shape[0], pepper_count)
    pepper_x = np.random.randint(0, image.shape[1], pepper_count)

    noisy[salt_y, salt_x] = 255
    noisy[pepper_y, pepper_x] = 0

    return noisy


def show_noise_reduction(image, title, noisy):
    gaussian = cv2.GaussianBlur(noisy, (5, 5), 0)
    median = cv2.medianBlur(noisy, 5)
    average = cv2.blur(noisy, (5, 5))

    items = [
        ("Original", bgr_to_rgb(image), None),
        ("Noisy", bgr_to_rgb(noisy), None),
        ("GaussianBlur", bgr_to_rgb(gaussian), None),
        ("MedianBlur", bgr_to_rgb(median), None),
        ("AverageBlur", bgr_to_rgb(average), None)
    ]

    show_grid(title, items, 1, 5, (18, 4))


def main():
    images = load_images(IMAGE_PATHS)

    show_originals(images)

    for name, image in images.items():
        show_color_spaces(name, image)

    selected_image = images["Portrait"]
    text_image = images["Text"]

    show_smoothing(selected_image)
    show_sharpening(selected_image)
    show_morphology(text_image)

    gaussian_noisy = add_gaussian_noise(selected_image)
    salt_pepper_noisy = add_salt_pepper_noise(selected_image)

    show_noise_reduction(selected_image, "Gaussian noise reduction", gaussian_noisy)
    show_noise_reduction(selected_image, "Salt and pepper noise reduction", salt_pepper_noisy)


if __name__ == "__main__":
    main()