from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def create_grid_stitched_image(image_paths: list[Path], output_path: Path) -> None:
    """Create a 3x6 stitched PNG from 18 single-channel images.

    Args:
        image_paths: Ordered list of 18 grayscale PNG paths for one segment.
        output_path: Output PNG path for the stitched grid image.

    Raises:
        ValueError: If the input count is not 18 or image shapes are inconsistent.
        OSError: If any image cannot be read or output cannot be written.
    """
    if len(image_paths) != 18:
        raise ValueError(
            f"Expected exactly 18 channel images, got {len(image_paths)}."
        )

    grayscale_images: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None

    for image_path in image_paths:
        channel_array = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
        if expected_shape is None:
            expected_shape = channel_array.shape
        elif channel_array.shape != expected_shape:
            raise ValueError(
                "All channel images must have the same shape for stitching."
            )
        grayscale_images.append(channel_array)

    stitched_rows: list[np.ndarray] = []
    for row_index in range(3):
        row_start = row_index * 6
        row_end = row_start + 6
        stitched_rows.append(np.hstack(grayscale_images[row_start:row_end]))

    stitched_grid = np.vstack(stitched_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(stitched_grid, mode="L").convert("RGB").save(output_path)


def build_and_stitch_all_images(
    training_images_root: Path, stitched_export_root: Path
) -> pd.DataFrame:
    """Scan all patient folders, find segments, stitch images and return metadata.

    Args:
        training_images_root: The root directory containing individual image folders.
        stitched_export_root: The root directory to save stitched images.

    Returns:
        A dataframe containing the paths to the stitched images.
    """
    records = []

    patient_folders = sorted(p for p in training_images_root.iterdir() if p.is_dir())
    print(f"Found {len(patient_folders)} patient folders in {training_images_root}")

    for patient_folder in tqdm(patient_folders, desc="Stitching patients"):
        patient_id = patient_folder.name

        segment_to_paths: dict[str, list[Path]] = {}
        for image_path in sorted(patient_folder.glob("*.png")):
            filename_parts = image_path.stem.split("_EEG__", maxsplit=1)
            if len(filename_parts) != 2:
                continue
            segment_token = filename_parts[0]
            segment_to_paths.setdefault(segment_token, []).append(image_path)

        for segment_token, segment_image_paths in sorted(segment_to_paths.items()):
            if len(segment_image_paths) != 18:
                print(
                    f"Skipping {patient_id} segment {segment_token}: found {len(segment_image_paths)} images instead of 18."
                )
                continue

            stitched_output_path = (
                stitched_export_root / patient_id / f"grid_stitched_{segment_token}.png"
            )

            # Create and save the stitched image
            create_grid_stitched_image(segment_image_paths, stitched_output_path)

            # Store relative paths for the parquet artifact
            relative_path = str(
                stitched_output_path.relative_to(training_images_root.parent.parent)
            )

            records.append(
                {
                    "Patient": patient_id,
                    "segment_token": segment_token,
                    "grid_relative_path": relative_path,
                }
            )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    training_images_root = Path("exports") / "mel_360x360" / "train"
    stitched_export_root = Path("exports") / "mel_stitched"

    print("Stitching images...")
    stitched_df = build_and_stitch_all_images(
        training_images_root, stitched_export_root
    )

    print(f"Total stitched images created: {len(stitched_df)}")

    artifact_root = Path("artifacts")
    artifact_root.mkdir(parents=True, exist_ok=True)

    print("Preparing metadata artifact...")
    parquet_path = artifact_root / "all_stitched_data.parquet"
    
    if not stitched_df.empty:
        stitched_df.to_parquet(parquet_path, index=False)
        print(f"Saved metadata artifact to {parquet_path.resolve()}")
    else:
        print("No images were stitched. Dataframe is empty.")

    print("Local image stitching complete.")

