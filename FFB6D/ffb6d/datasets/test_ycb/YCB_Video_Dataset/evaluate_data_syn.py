import os
from PIL import Image
import numpy as np
from collections import Counter

def analyze_png_pixel_frequency(png_directory):
    """
    Analyzes PNG images in a directory, where each pixel is an integer,
    and prints the frequency of each integer in each image.

    Args:
        png_directory (str): The path to the directory containing the PNG images.
    """

    png_files = sorted([f for f in os.listdir(png_directory) if f.endswith("-label.png")])

    for png_file in png_files:
        png_path = os.path.join(png_directory, png_file)

        try:
            img = Image.open(png_path)
            img_array = np.array(img)

            # Flatten the array to get a 1D list of pixel values
            pixel_values = img_array.flatten()

            # Count the frequency of each pixel value
            pixel_counts = Counter(pixel_values)

            print(f"Pixel frequency for {png_file}:")
            for pixel_value, count in sorted(pixel_counts.items()):
                print(f"  Pixel {pixel_value}: {count}")
            print("-" * 20)  # Separator between images

        except FileNotFoundError:
            print(f"Error: File not found - {png_path}")
        except Exception as e:
            print(f"Error processing {png_file}: {e}")

if __name__ == "__main__":
    png_directory = "data_syn"  # Replace with your directory path
    analyze_png_pixel_frequency(png_directory)
