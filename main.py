import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass
class ImageItem:
    name: str
    path: str
    bgr: np.ndarray


class ImageRepository:
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def load(self):
        images = []

        for name, path in self.image_paths.items():
            image = cv2.imread(path)

            if image is None:
                raise FileNotFoundError(f"Image not found: {path}")

            images.append(ImageItem(name=name, path=path, bgr=image))

        return images


class ColorSpaceProcessor:
    def to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def to_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def to_hsv(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def get_hsv_channels(self, image):
        hsv = self.to_hsv(image)
        return cv2.split(hsv)


class FilterProcessor:
    def gaussian_blur(self, image, kernel_size):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def median_blur(self, image, kernel_size):
        return cv2.medianBlur(image, kernel_size)

    def average_blur(self, image, kernel_size):
        return cv2.blur(image, (kernel_size, kernel_size))

    def sharpen(self, image):
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])

        return cv2.filter2D(image, -1, kernel)

    def get_smoothing_results(self, image, kernel_sizes):
        results = []

        for size in kernel_sizes:
            results.append((f"Gaussian {size}x{size}", self.gaussian_blur(image, size)))
            results.append((f"Median {size}x{size}", self.median_blur(image, size)))
            results.append((f"Average {size}x{size}", self.average_blur(image, size)))

        return results


class NoiseProcessor:
    def add_gaussian_noise(self, image, mean=0, sigma=25):
        noise = np.random.normal(mean, sigma, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_salt_pepper_noise(self, image, amount=0.03):
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


class MorphologyProcessor:
    def __init__(self, color_processor):
        self.color_processor = color_processor

    def binarize(self, image):
        gray = self.color_processor.to_grayscale(image)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def erode(self, binary, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(binary, kernel, iterations=1)

    def dilate(self, binary, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(binary, kernel, iterations=1)

    def opening(self, binary, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    def closing(self, binary, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def get_morphology_results(self, image, kernel_sizes):
        binary = self.binarize(image)
        results = [("Binary", binary)]

        for size in kernel_sizes:
            results.append((f"Erosion {size}x{size}", self.erode(binary, size)))
            results.append((f"Dilation {size}x{size}", self.dilate(binary, size)))
            results.append((f"Opening {size}x{size}", self.opening(binary, size)))
            results.append((f"Closing {size}x{size}", self.closing(binary, size)))

        return results


class Visualizer:
    def __init__(self, color_processor, output_dir="outputs"):
        self.color_processor = color_processor
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def prepare_image(self, image):
        if len(image.shape) == 2:
            return image, "gray"

        return self.color_processor.to_rgb(image), None

    def show_and_save_grid(self, title, items, rows, cols, filename, figsize=(16, 10)):
        plt.figure(figsize=figsize)
        plt.suptitle(title, fontsize=16)

        for index, (item_title, image) in enumerate(items, start=1):
            prepared_image, cmap = self.prepare_image(image)

            plt.subplot(rows, cols, index)
            plt.imshow(prepared_image, cmap=cmap)
            plt.title(item_title)
            plt.axis("off")

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()

    def show_originals(self, images):
        items = [(item.name, item.bgr) for item in images]
        self.show_and_save_grid(
            title="Original images",
            items=items,
            rows=1,
            cols=len(items),
            filename="01_originals.png",
            figsize=(15, 5)
        )

    def show_color_spaces(self, image_item):
        rgb = self.color_processor.to_rgb(image_item.bgr)
        grayscale = self.color_processor.to_grayscale(image_item.bgr)
        h, s, v = self.color_processor.get_hsv_channels(image_item.bgr)

        items = [
            ("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)),
            ("Grayscale", grayscale),
            ("HSV Hue", h),
            ("HSV Saturation", s),
            ("HSV Value", v)
        ]

        self.show_and_save_grid(
            title=f"Color spaces: {image_item.name}",
            items=items,
            rows=1,
            cols=5,
            filename=f"02_color_spaces_{image_item.name.lower()}.png",
            figsize=(18, 4)
        )

    def show_smoothing(self, image_item, smoothing_results):
        self.show_and_save_grid(
            title=f"Smoothing filters: {image_item.name}",
            items=smoothing_results,
            rows=3,
            cols=3,
            filename="03_smoothing_filters.png",
            figsize=(15, 12)
        )

    def show_sharpening(self, image_item, sharpened):
        items = [
            ("Before", image_item.bgr),
            ("After", sharpened)
        ]

        self.show_and_save_grid(
            title=f"Sharpening: {image_item.name}",
            items=items,
            rows=1,
            cols=2,
            filename="04_sharpening.png",
            figsize=(12, 5)
        )

    def show_morphology(self, image_item, morphology_results):
        self.show_and_save_grid(
            title=f"Morphology: {image_item.name}",
            items=morphology_results,
            rows=4,
            cols=4,
            filename="05_morphology.png",
            figsize=(18, 16)
        )

    def show_noise_reduction(self, title, original, noisy, filtered_results, filename):
        items = [
            ("Original", original),
            ("Noisy", noisy)
        ]

        items.extend(filtered_results)

        self.show_and_save_grid(
            title=title,
            items=items,
            rows=1,
            cols=5,
            filename=filename,
            figsize=(18, 4)
        )

    def show_final_comparison(self, images, filter_processor, morphology_processor):
        rows = len(images)
        cols = 6

        plt.figure(figsize=(18, 5 * rows))
        plt.suptitle("Final preprocessing comparison", fontsize=16)

        for row_index, image_item in enumerate(images):
            original = image_item.bgr
            grayscale = self.color_processor.to_grayscale(original)
            gaussian = filter_processor.gaussian_blur(original, 11)
            median = filter_processor.median_blur(original, 11)
            sharpened = filter_processor.sharpen(original)
            binary = morphology_processor.binarize(original)
            morph = morphology_processor.opening(binary, 5)

            row_items = [
                ("Original", original),
                ("Grayscale", grayscale),
                ("GaussianBlur", gaussian),
                ("MedianBlur", median),
                ("Sharpening", sharpened),
                ("Morphology", morph)
            ]

            for col_index, (title, image) in enumerate(row_items):
                prepared_image, cmap = self.prepare_image(image)
                plot_index = row_index * cols + col_index + 1

                plt.subplot(rows, cols, plot_index)
                plt.imshow(prepared_image, cmap=cmap)
                plt.title(f"{image_item.name}\n{title}")
                plt.axis("off")

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "07_final_comparison.png")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.show()


class PreprocessingExperiment:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.repository = ImageRepository(image_paths)
        self.color_processor = ColorSpaceProcessor()
        self.filter_processor = FilterProcessor()
        self.noise_processor = NoiseProcessor()
        self.morphology_processor = MorphologyProcessor(self.color_processor)
        self.visualizer = Visualizer(self.color_processor)

    def run(self):
        images = self.repository.load()

        self.visualizer.show_originals(images)

        for image_item in images:
            self.visualizer.show_color_spaces(image_item)

        selected_image = images[0]
        text_image = images[2]

        smoothing_results = self.filter_processor.get_smoothing_results(
            selected_image.bgr,
            [5, 11, 21]
        )

        self.visualizer.show_smoothing(selected_image, smoothing_results)

        sharpened = self.filter_processor.sharpen(selected_image.bgr)
        self.visualizer.show_sharpening(selected_image, sharpened)

        morphology_results = self.morphology_processor.get_morphology_results(
            text_image.bgr,
            [3, 5, 9]
        )

        self.visualizer.show_morphology(text_image, morphology_results)

        gaussian_noisy = self.noise_processor.add_gaussian_noise(selected_image.bgr)
        salt_pepper_noisy = self.noise_processor.add_salt_pepper_noise(selected_image.bgr)

        gaussian_filtered = [
            ("GaussianBlur", self.filter_processor.gaussian_blur(gaussian_noisy, 5)),
            ("MedianBlur", self.filter_processor.median_blur(gaussian_noisy, 5)),
            ("AverageBlur", self.filter_processor.average_blur(gaussian_noisy, 5))
        ]

        salt_pepper_filtered = [
            ("GaussianBlur", self.filter_processor.gaussian_blur(salt_pepper_noisy, 5)),
            ("MedianBlur", self.filter_processor.median_blur(salt_pepper_noisy, 5)),
            ("AverageBlur", self.filter_processor.average_blur(salt_pepper_noisy, 5))
        ]

        self.visualizer.show_noise_reduction(
            title="Gaussian noise reduction",
            original=selected_image.bgr,
            noisy=gaussian_noisy,
            filtered_results=gaussian_filtered,
            filename="06_gaussian_noise_reduction.png"
        )

        self.visualizer.show_noise_reduction(
            title="Salt and pepper noise reduction",
            original=selected_image.bgr,
            noisy=salt_pepper_noisy,
            filtered_results=salt_pepper_filtered,
            filename="06_salt_pepper_noise_reduction.png"
        )

        self.visualizer.show_final_comparison(
            images=images,
            filter_processor=self.filter_processor,
            morphology_processor=self.morphology_processor
        )


def main():
    image_paths = {
        "Portrait": "images/portrait.jpg",
        "City": "images/city.jpg",
        "Text": "images/text.jpg"
    }

    experiment = PreprocessingExperiment(image_paths)
    experiment.run()


if __name__ == "__main__":
    main()