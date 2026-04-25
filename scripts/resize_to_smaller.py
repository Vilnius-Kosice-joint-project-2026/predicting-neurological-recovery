import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def process_single_image(
    img_path: Path, input_root: Path, output_root: Path, target_size: tuple[int, int]
) -> None:
    """Worker function to process a single image."""
    relative_path = img_path.relative_to(input_root)
    out_path = output_root / relative_path

    # Make sure the patient outcome folder exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Open, resize, and save the image
    with Image.open(img_path) as img:
        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
        resized_img.save(out_path)


def resize_dataset(
    input_root: Path, output_root: Path, target_size: tuple[int, int]
) -> None:
    """
    Iterates over all images in patient subfolders, resizes them using multiprocessing,
    and saves them to a new location preserving the folder hierarchy.

    Args:
        input_root: Root directory containing patient folders with images.
        output_root: Root directory where resized images will be saved.
        target_size: Desired image size as (width, height).
    """
    # Grab all PNGs regardless of directory depth
    image_paths = list(input_root.rglob("*.png"))
    
    if not image_paths:
        print(f"No images found in {input_root}.")
        return

    print(f"Found {len(image_paths)} images to resize in {input_root}")

    # Use 4-core multiprocessing
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(process_single_image, img_path, input_root, output_root, target_size)
            for img_path in image_paths
        ]
        
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Resizing images (4 cores)"):
            pass


if __name__ == "__main__":
    # Point these to your stitched dataset directories
    INPUT_DIR = Path("exports") / "mel_stitched"
    OUTPUT_DIR = Path("exports") / "mel_stitched_224x448"

    IMG_HEIGHT = 224
    IMG_WIDTH = 448

    print(f"Starting resizing process...")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Target size: {IMG_WIDTH}x{IMG_HEIGHT} (Width x Height)\n")

    resize_dataset(
        input_root=INPUT_DIR,
        output_root=OUTPUT_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
    )

    print("\nResizing complete.")
