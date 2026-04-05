# EEG recordings were consistently captured using 19 electrodes, adhering to the international 10-20 system. In the preprocessing phase, our initial step was to standardize the order of the 19 channels to the sequence. Subsequently, the data from these 19 channels was transformed into 18 bipolar channels, namely: 'Fp1-F7', 'F7-T3', 'T3-T5', 'T5- O1', 'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'Fz- Cz', and 'Cz-Pz'. We opted for bipolar referencing due to its prevalent utility in clinical settings, and the fact that many prior quantitative EEG analyses and models related to cardiac arrest have employed bipolar channels [2]. Additionally, each bipolar channel underwent processing with a 2nd order Butterworth band-pass filter, with cut-off frequencies established at 0.5 Hz and 50 Hz. This was followed by resampling to a uniform frequency of 128 Hz.During our analysis, we identified extended flat zero segments in many bipolar signals. To address this, we eliminated signal segments exhibiting flat zeros over a 10- second window. For our study, we converted one-hour segments from each bipolar channel into mel-spectrograms utilizing the 'librosa' library. We adopted a hop length of 10 seconds, ensuring that the duration of each hour is represented by 360 units. Our analysis concentrated on frequencies ranging from 0 to 45 Hz, resulting in mel-spectrograms with dimensions of 360x360. Fig. 2 provides illustrative examples, showcasing the average mel-spectrograms across 18 channels for five distinct CPC levels at varying times. For training purposes, we utilized the final 12 hours of data for each patient, translating to 12 mel-spectrogram images.

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import mne
import numpy as np
import scipy.io
from matplotlib.axes import Axes
from PIL import Image

# Static exploration settings
DATA_ROOT = Path("icare_data/training")
INCLUDED_SIGNAL_TYPES = ("EEG",)
MAX_EXAMPLES_PER_TYPE = 9999
PREFERRED_PATIENTS: set[str] = set()  # Example: {"0676", "0690", "0651"}
PREFERRED_SEGMENT_ID: Optional[str] = None  # Example: "002"

PREPROCESS_MODE = "bipolar"  # Options: "monopolar", "bipolar"
TARGET_SAMPLING_RATE_HZ = 128.0

ENABLE_BANDPASS_FILTER = True
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 50.0
FILTER_ORDER = 2

ENABLE_FLAT_ZERO_ELIMINATION = True
FLAT_ZERO_WINDOW_SECONDS = 10.0
FLAT_ZERO_ABS_TOLERANCE_UV = 0.5
FLAT_ZERO_MIN_CONSECUTIVE_WINDOWS = 1
FLAT_ZERO_REQUIRE_ALL_CHANNELS_FLAT = True

ENABLE_MEL_SPECTROGRAM = True
MEL_N_MELS = 360
MEL_N_FFT = 1024
MEL_HOP_SECONDS = 10.0
MEL_F_MIN_HZ = 0.0
MEL_F_MAX_HZ = 45.0
MEL_EXPECTED_TIME_STEPS = 360
MEL_PAD_VALUE_DB = -80.0

DEFAULT_START_S = 0.0
DEFAULT_DURATION_S = 300.0
DEFAULT_SPACING_UV = 150.0

# Output root for vision training
EXPORT_ROOT = Path("exports/mel_360x360")
EXPORT_SPLIT = "train"  # e.g., train/val/test
# If True: normalize each image independently to [0,255]
# If False: use fixed dB range for all images (better consistency across dataset)
PER_IMAGE_NORMALIZE = False
DB_MIN = -80.0
DB_MAX = 0.0

CANONICAL_19_ORDER: tuple[str, ...] = (
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
)


BIPOLAR_18_PAIRS: tuple[tuple[str, str], ...] = (
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
    ("Fz", "Cz"),
    ("Cz", "Pz"),
)


@dataclass(frozen=True)
class SegmentRecord:
    """Represents one paired WFDB segment (.hea + .mat)."""

    patient_id: str
    segment_id: str
    hour_token: str
    signal_type: str
    hea_path: Path
    mat_path: Path


def normalize_channel_label(channel_name: str) -> str:
    """Normalize EEG channel names into canonical 10-20 labels.

    Args:
        channel_name: Raw channel label from WFDB header.

    Returns:
        Canonicalized channel label.
    """
    compact_name = channel_name.strip().replace(" ", "")
    normalized = compact_name.upper()

    alias_candidates = {
        "FP1": "Fp1",
        "FP2": "Fp2",
        "F3": "F3",
        "F4": "F4",
        "C3": "C3",
        "C4": "C4",
        "P3": "P3",
        "P4": "P4",
        "O1": "O1",
        "O2": "O2",
        "F7": "F7",
        "F8": "F8",
        "T3": "T3",
        "T4": "T4",
        "T5": "T5",
        "T6": "T6",
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
    }

    if normalized in alias_candidates:
        return alias_candidates[normalized]

    return compact_name


def parse_hea_header(hea_path: Path) -> dict[str, object]:
    """Parse channel metadata and timing details from a WFDB .hea file.

    Args:
        hea_path: Path to the WFDB header file.

    Returns:
        Parsed metadata including channel names, gains, baselines, sampling rate,
        and total sample count.

    Raises:
        FileNotFoundError: If the header file is missing.
        ValueError: If the header content is invalid.
    """
    if not hea_path.exists():
        raise FileNotFoundError(f"Header not found: {hea_path}")

    lines = hea_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Empty header file: {hea_path}")

    first_tokens = lines[0].split()
    if len(first_tokens) < 4:
        raise ValueError(f"Invalid first header line in {hea_path}")

    sampling_rate_hz = float(first_tokens[2])
    total_samples = int(first_tokens[3])

    raw_channel_names: list[str] = []
    channel_names: list[str] = []
    gains: list[float] = []
    baselines: list[int] = []

    for line in lines[1:]:
        if not line.strip() or line.startswith("#"):
            break
        parts = line.split()
        if len(parts) < 3:
            continue

        gain_baseline_match = re.match(r"([\d\.eE+\-]+)\(([+\-]?\d+)\)", parts[2])
        if gain_baseline_match:
            gains.append(float(gain_baseline_match.group(1)))
            baselines.append(int(gain_baseline_match.group(2)))
        else:
            gains.append(1.0)
            baselines.append(0)

        raw_channel_name = parts[-1]
        raw_channel_names.append(raw_channel_name)
        channel_names.append(normalize_channel_label(raw_channel_name))

    if not channel_names:
        raise ValueError(f"No channels parsed from header: {hea_path}")

    duplicate_names = sorted(
        {name for name in channel_names if channel_names.count(name) > 1}
    )

    gains_array = np.array(gains, dtype=float)
    gains_array[gains_array == 0.0] = 1.0

    return {
        "sampling_rate_hz": sampling_rate_hz,
        "total_samples": total_samples,
        "channel_names": channel_names,
        "raw_channel_names": raw_channel_names,
        "gains": gains_array,
        "baselines": np.array(baselines, dtype=float),
        "duplicate_channel_names": duplicate_names,
    }


def load_icare_segment(
    hea_path: Path, mat_path: Path
) -> tuple[np.ndarray, dict[str, object]]:
    """Load one I-CARE segment and convert raw ADC values to microvolts.

    Args:
        hea_path: Path to segment header file.
        mat_path: Path to segment MAT signal file.

    Returns:
        Tuple of calibrated signal matrix (channels x samples) and metadata.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If data shape cannot be aligned to header channels.
    """
    if not mat_path.exists():
        raise FileNotFoundError(f"Signal file not found: {mat_path}")

    metadata = parse_hea_header(hea_path)
    channel_names = metadata["channel_names"]
    gains = metadata["gains"]
    baselines = metadata["baselines"]

    mat_data = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    candidate_arrays = {
        key: value
        for key, value in mat_data.items()
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.size > 1
    }

    if not candidate_arrays:
        raise ValueError(f"No 2D signal matrix found in MAT file: {mat_path}")

    raw_key = max(candidate_arrays, key=lambda key: candidate_arrays[key].size)
    raw_signal = candidate_arrays[raw_key].astype(float)

    n_channels = len(channel_names)
    if raw_signal.shape[0] == n_channels:
        aligned_signal = raw_signal
    elif raw_signal.shape[1] == n_channels:
        aligned_signal = raw_signal.T
    else:
        raise ValueError(
            "Channel count mismatch between header and MAT data: "
            f"header={n_channels}, mat_shape={raw_signal.shape}"
        )

    calibrated_signal_uv = (aligned_signal - baselines[:, None]) / gains[:, None]
    metadata["mat_key"] = raw_key
    metadata["raw_shape"] = raw_signal.shape
    metadata["calibrated_shape"] = calibrated_signal_uv.shape

    return calibrated_signal_uv, metadata


def discover_icare_segments(root_dir: Path) -> list[SegmentRecord]:
    """Find valid WFDB segment pairs under the training directory.

    Args:
        root_dir: Root directory that contains patient subfolders.

    Returns:
        Sorted list of segment records that have both .hea and .mat files.
    """
    records: list[SegmentRecord] = []
    pattern = re.compile(
        r"^(?P<patient>\d{4})_(?P<segment>\d{3})_(?P<hour>\d{3})_(?P<signal>[A-Z0-9]+)$"
    )

    for hea_path in sorted(root_dir.glob("*/*.hea")):
        stem_match = pattern.match(hea_path.stem)
        if stem_match is None:
            continue

        mat_path = hea_path.with_suffix(".mat")
        if not mat_path.exists():
            continue

        records.append(
            SegmentRecord(
                patient_id=stem_match.group("patient"),
                segment_id=stem_match.group("segment"),
                hour_token=stem_match.group("hour"),
                signal_type=stem_match.group("signal"),
                hea_path=hea_path,
                mat_path=mat_path,
            )
        )

    records.sort(
        key=lambda record: (
            record.signal_type,
            record.patient_id,
            record.segment_id,
            record.hour_token,
        )
    )
    return records


def select_static_examples(
    records: list[SegmentRecord],
    signal_types: tuple[str, ...],
    max_examples_per_type: int,
    preferred_patients: Optional[set[str]] = None,
    preferred_segment_id: Optional[str] = None,
) -> list[SegmentRecord]:
    """Select deterministic examples per signal type for static visualization."""
    selected: list[SegmentRecord] = []
    preferred_patients = preferred_patients or set()

    for signal_type in signal_types:
        by_type = [record for record in records if record.signal_type == signal_type]
        if preferred_segment_id:
            by_type = [
                record
                for record in by_type
                if record.segment_id == preferred_segment_id
            ]

        if preferred_patients:
            prioritized = [
                record for record in by_type if record.patient_id in preferred_patients
            ]
            fallback = [
                record
                for record in by_type
                if record.patient_id not in preferred_patients
            ]
            by_type = prioritized + fallback

        selected.extend(by_type[:max_examples_per_type])

    return selected


def standardize_to_19_channels(
    signal_uv: np.ndarray, channel_names: list[str]
) -> tuple[np.ndarray, dict[str, object]]:
    """Reorder/calibrate monopolar EEG to canonical 19-channel order."""
    canonical_index = {name: idx for idx, name in enumerate(CANONICAL_19_ORDER)}
    normalized_names = [normalize_channel_label(name) for name in channel_names]

    standardized = np.full((len(CANONICAL_19_ORDER), signal_uv.shape[1]), np.nan, dtype=float)
    unknown_channels: list[str] = []
    duplicate_canonical: list[str] = []

    seen: set[str] = set()
    for source_idx, channel_name in enumerate(normalized_names):
        if channel_name not in canonical_index:
            unknown_channels.append(channel_name)
            continue

        target_idx = canonical_index[channel_name]
        if channel_name in seen:
            duplicate_canonical.append(channel_name)
            continue

        standardized[target_idx, :] = signal_uv[source_idx, :]
        seen.add(channel_name)

    missing_channels = [
        channel_name
        for channel_name in CANONICAL_19_ORDER
        if channel_name not in seen
    ]

    diagnostics = {
        "input_channel_names": channel_names,
        "normalized_channel_names": normalized_names,
        "canonical_order": list(CANONICAL_19_ORDER),
        "missing_canonical_channels": missing_channels,
        "unknown_channels": sorted(set(unknown_channels)),
        "duplicate_canonical_channels": sorted(set(duplicate_canonical)),
    }
    return standardized, diagnostics


def to_bipolar_18_mne(
    standardized_monopolar_uv: np.ndarray,
    sampling_rate_hz: float,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Derive fixed 18 bipolar channels via MNE Raw.set_bipolar_reference."""
    if sampling_rate_hz <= 0:
        raise ValueError(f"Sampling rate must be positive, got {sampling_rate_hz}.")

    bipolar_names = [f"{left}-{right}" for left, right in BIPOLAR_18_PAIRS]
    info = mne.create_info(
        ch_names=list(CANONICAL_19_ORDER),
        sfreq=float(sampling_rate_hz),
        ch_types=["eeg"] * len(CANONICAL_19_ORDER),
    )
    raw = mne.io.RawArray(standardized_monopolar_uv.astype(float), info, verbose="ERROR")

    missing_mask = np.array(
        [
            bool(np.isnan(standardized_monopolar_uv[CANONICAL_19_ORDER.index(left)]).all() or np.isnan(standardized_monopolar_uv[CANONICAL_19_ORDER.index(right)]).all())
            for left, right in BIPOLAR_18_PAIRS
        ],
        dtype=bool,
    )
    missing_pair_names = [
        f"{left}-{right}"
        for (left, right), is_missing in zip(BIPOLAR_18_PAIRS, missing_mask)
        if is_missing
    ]

    bipolar_raw = mne.set_bipolar_reference(
        raw,
        anode=[left for left, _ in BIPOLAR_18_PAIRS],
        cathode=[right for _, right in BIPOLAR_18_PAIRS],
        ch_name=bipolar_names,
        copy=True,
        drop_refs=False,
        verbose="ERROR",
    )
    bipolar_uv = bipolar_raw.get_data(picks=bipolar_names)

    # Keep legacy missing-pair behavior: fully missing pairs remain NaN.
    if missing_mask.any():
        bipolar_uv[missing_mask, :] = np.nan

    return bipolar_uv, bipolar_names, missing_mask, missing_pair_names


def prepare_bipolar_segment(
    signal_uv: np.ndarray, metadata: dict[str, object]
) -> tuple[np.ndarray, dict[str, object]]:
    """Prepare one segment for bipolar analysis with MNE-based derivation."""
    standardized, diagnostics = standardize_to_19_channels(
        signal_uv=signal_uv, channel_names=list(metadata["channel_names"])
    )
    bipolar_uv, bipolar_names, missing_pair_mask, missing_pairs = to_bipolar_18_mne(
        standardized_monopolar_uv=standardized,
        sampling_rate_hz=float(metadata["sampling_rate_hz"]),
    )

    metadata = dict(metadata)
    metadata["preprocess_mode"] = "bipolar"
    metadata["standardized_monopolar_shape"] = standardized.shape
    metadata["bipolar_shape"] = bipolar_uv.shape
    metadata["channel_names"] = bipolar_names
    metadata["bipolar_pair_names"] = bipolar_names
    metadata["missing_bipolar_pair_mask"] = missing_pair_mask
    metadata["missing_bipolar_pairs"] = missing_pairs
    metadata["channel_diagnostics"] = diagnostics
    metadata["bipolar_derivation"] = "mne.set_bipolar_reference"

    return bipolar_uv, metadata


def summarize_segment(record: SegmentRecord, metadata: dict[str, object]) -> None:
    """Print compact segment metadata for quick exploratory review."""
    total_samples = int(metadata["total_samples"])
    sampling_rate_hz = float(metadata["sampling_rate_hz"])
    duration_s = total_samples / sampling_rate_hz if sampling_rate_hz > 0 else np.nan
    n_channels = len(metadata["channel_names"])

    print(
        " | ".join(
            [
                f"segment={record.patient_id}_{record.segment_id}_{record.hour_token}_{record.signal_type}",
                f"fs={sampling_rate_hz:.2f}Hz",
                f"samples={total_samples}",
                f"duration={duration_s:.1f}s",
                f"channels={n_channels}",
            ]
        )
    )

    if metadata.get("preprocess_mode") == "bipolar":
        missing_pairs = metadata.get("missing_bipolar_pairs", [])
        diagnostics = metadata.get("channel_diagnostics", {})
        missing_canonical = diagnostics.get("missing_canonical_channels", [])
        unknown_channels = diagnostics.get("unknown_channels", [])
        duplicate_channels = diagnostics.get("duplicate_canonical_channels", [])

        print(
            "readiness="
            + ("READY" if not missing_pairs else "PARTIAL")
            + f" | missing_pairs={len(missing_pairs)}"
        )
        if missing_canonical:
            print("  missing_canonical_channels=" + ", ".join(missing_canonical))
        if unknown_channels:
            print("  unknown_channels=" + ", ".join(unknown_channels))
        if duplicate_channels:
            print("  duplicate_canonical_channels=" + ", ".join(duplicate_channels))


def plot_stacked_channels(
    signal_uv: np.ndarray,
    metadata: dict[str, object],
    title: str,
    start_s: float = 0.0,
    duration_s: float = 60.0,
    spacing_uv: Optional[float] = None,
    channel_subset: Optional[list[str]] = None,
    clip_quantile: float = 0.995,
    max_points: int = 15000,
    axis: Optional[Axes] = None,
    show_figure: bool = True,
) -> None:
    """Render stacked traces for a selected time window."""
    sampling_rate_hz = float(metadata["sampling_rate_hz"])
    channel_names = list(metadata["channel_names"])

    if channel_subset:
        channel_index_map = {name: idx for idx, name in enumerate(channel_names)}
        selected_indices = [
            channel_index_map[name]
            for name in channel_subset
            if name in channel_index_map
        ]
        if selected_indices:
            signal_uv = signal_uv[selected_indices, :]
            channel_names = [channel_names[idx] for idx in selected_indices]

    start_idx = max(0, int(start_s * sampling_rate_hz))
    end_idx = min(signal_uv.shape[1], int((start_s + duration_s) * sampling_rate_hz))
    if end_idx <= start_idx:
        raise ValueError("Selected plotting window does not contain samples.")

    segment = signal_uv[:, start_idx:end_idx].copy()
    segment -= np.nanmedian(segment, axis=1, keepdims=True)

    step = max(1, int(np.ceil(segment.shape[1] / max_points)))
    segment = segment[:, ::step]
    time_axis = np.arange(start_idx, end_idx, step) / sampling_rate_hz

    clipped = segment.copy()
    if 0.5 < clip_quantile < 1.0:
        abs_limits = np.nanquantile(np.abs(clipped), clip_quantile, axis=1)
        abs_limits[~np.isfinite(abs_limits)] = 1.0
        abs_limits[abs_limits == 0.0] = 1.0
        clipped = np.clip(clipped, -abs_limits[:, None], abs_limits[:, None])

    if spacing_uv is None:
        robust_amp = np.nanquantile(np.abs(clipped), 0.95)
        if not np.isfinite(robust_amp) or robust_amp == 0.0:
            robust_amp = 20.0
        spacing_uv = max(20.0, 2.5 * float(robust_amp))

    n_channels = clipped.shape[0]
    if axis is None:
        _, axis = plt.subplots(figsize=(16, max(3.0, n_channels * 0.55 + 1.0)))

    for channel_idx, channel_name in enumerate(channel_names):
        offset = (n_channels - 1 - channel_idx) * spacing_uv
        axis.plot(time_axis, clipped[channel_idx] + offset, color="black", linewidth=0.6)

    y_ticks = [(n_channels - 1 - idx) * spacing_uv for idx in range(n_channels)]
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(channel_names, fontsize=8)
    axis.set_xlabel("Time (s)")
    axis.set_title(title)
    axis.set_xlim(time_axis[0], time_axis[-1])
    axis.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)

    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    if show_figure:
        plt.tight_layout()
        plt.show()

def _fill_nan_1d(signal_1d: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace NaN/inf values in a 1D signal using linear interpolation.

    Args:
        signal_1d: One-dimensional signal array.

    Returns:
        Tuple with cleaned signal and count of replaced samples.

    Raises:
        ValueError: If no finite values exist.
    """
    if signal_1d.ndim != 1:
        raise ValueError(f"Expected 1D input, got shape {signal_1d.shape}.")

    finite_mask = np.isfinite(signal_1d)
    replaced_count = int((~finite_mask).sum())
    if replaced_count == 0:
        return signal_1d.astype(float, copy=True), 0
    if not finite_mask.any():
        raise ValueError("Signal contains no finite values.")

    signal_clean = signal_1d.astype(float, copy=True)
    sample_index = np.arange(signal_clean.size, dtype=float)
    signal_clean[~finite_mask] = np.interp(
        sample_index[~finite_mask],
        sample_index[finite_mask],
        signal_clean[finite_mask],
    )
    return signal_clean, replaced_count


def _enforce_spectrogram_shape(
    spectrogram_2d: np.ndarray,
    target_shape: tuple[int, int],
    pad_value: float = -80.0,
) -> np.ndarray:
    """Deterministically crop/pad a 2D spectrogram to a fixed shape.

    Args:
        spectrogram_2d: Input spectrogram with shape (freq_bins, time_steps).
        target_shape: Desired output shape (freq_bins, time_steps).
        pad_value: Value used for padding.

    Returns:
        Spectrogram with exact target shape.

    Raises:
        ValueError: If input is not 2D or target shape is invalid.
    """
    if spectrogram_2d.ndim != 2:
        raise ValueError(
            f"Spectrogram must be 2D, got shape {spectrogram_2d.shape}."
        )
    target_freq_bins, target_time_steps = target_shape
    if target_freq_bins <= 0 or target_time_steps <= 0:
        raise ValueError(f"Invalid target shape: {target_shape}.")

    output = np.full(target_shape, pad_value, dtype=np.float32)
    copy_freq = min(target_freq_bins, spectrogram_2d.shape[0])
    copy_time = min(target_time_steps, spectrogram_2d.shape[1])
    output[:copy_freq, :copy_time] = spectrogram_2d[:copy_freq, :copy_time].astype(
        np.float32
    )
    return output


def create_bipolar_mel_spectrograms(
    signal_uv: np.ndarray,
    metadata: dict[str, object],
    expected_channel_names: list[str],
    n_mels: int = 360,
    n_fft: int = 1024,
    hop_seconds: float = 10.0,
    f_min_hz: float = 0.0,
    f_max_hz: float = 45.0,
    expected_sampling_rate_hz: float = 128.0,
    expected_time_steps: int = 360,
    pad_value: float = -80.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    """Create mel-spectrogram features from postprocessed bipolar EEG.

    Args:
        signal_uv: Channel-major bipolar EEG, shape (18, samples).
        metadata: Segment metadata that includes channel names and sampling rate.
        expected_channel_names: Required bipolar channel order.
        n_mels: Number of mel bins.
        n_fft: FFT window size.
        hop_seconds: Hop length in seconds.
        f_min_hz: Minimum analyzed frequency.
        f_max_hz: Maximum analyzed frequency.
        expected_sampling_rate_hz: Required input sampling rate.
        expected_time_steps: Target number of output time steps.
        pad_value: Padding value for deterministic time-dimension completion.

    Returns:
        Tuple with:
        - Mapping of channel_name -> spectrogram array (360, 360).
        - Stacked tensor with shape (18, 360, 360), axis order [channel, mel, time].
        - Spectrogram diagnostics metadata.

    Raises:
        ValueError: For invalid input shape, channel order, or sampling rate.
    """
    if signal_uv.ndim != 2:
        raise ValueError(
            f"Expected channel-major 2D signal, got shape {signal_uv.shape}."
        )
    if signal_uv.shape[0] != 18:
        raise ValueError(f"Expected 18 bipolar channels, got {signal_uv.shape[0]}.")

    sampling_rate_hz = float(metadata.get("sampling_rate_hz", np.nan))
    if not np.isclose(sampling_rate_hz, expected_sampling_rate_hz):
        raise ValueError(
            f"Expected sampling rate {expected_sampling_rate_hz} Hz, got {sampling_rate_hz} Hz."
        )

    channel_names = list(metadata.get("channel_names", []))
    if channel_names != expected_channel_names:
        raise ValueError(
            "Bipolar channel order mismatch for spectrogram extraction. "
            f"Expected {expected_channel_names} but got {channel_names}."
        )

    if n_mels <= 0 or expected_time_steps <= 0:
        raise ValueError(
            f"Invalid spectrogram dimensions n_mels={n_mels}, expected_time_steps={expected_time_steps}."
        )

    hop_length_samples = int(round(hop_seconds * sampling_rate_hz))
    if hop_length_samples <= 0:
        raise ValueError(
            f"Invalid hop_seconds={hop_seconds} for sampling_rate_hz={sampling_rate_hz}."
        )
    if not (0.0 <= f_min_hz < f_max_hz <= (sampling_rate_hz / 2.0)):
        raise ValueError(
            "Invalid frequency range: require "
            f"0 <= f_min_hz < f_max_hz <= Nyquist ({sampling_rate_hz / 2.0} Hz)."
        )

    by_channel: dict[str, np.ndarray] = {}
    stacked_list: list[np.ndarray] = []
    nan_replaced_per_channel: dict[str, int] = {}
    per_channel_shapes: dict[str, tuple[int, int]] = {}

    for channel_index, channel_name in enumerate(channel_names):
        channel_signal, replaced_count = _fill_nan_1d(signal_uv[channel_index, :])
        nan_replaced_per_channel[channel_name] = replaced_count

        mel_power = librosa.feature.melspectrogram(
            y=channel_signal.astype(np.float32),
            sr=int(sampling_rate_hz),
            n_fft=n_fft,
            hop_length=hop_length_samples,
            center=False,
            power=2.0,
            n_mels=n_mels,
            fmin=f_min_hz,
            fmax=f_max_hz,
        )
        mel_db = librosa.power_to_db(mel_power, ref=np.max)
        mel_fixed = _enforce_spectrogram_shape(
            spectrogram_2d=mel_db,
            target_shape=(n_mels, expected_time_steps),
            pad_value=pad_value,
        )
        if mel_fixed.shape != (n_mels, expected_time_steps):
            raise ValueError(
                f"Unexpected shape for {channel_name}: {mel_fixed.shape}."
            )

        by_channel[channel_name] = mel_fixed
        stacked_list.append(mel_fixed)
        per_channel_shapes[channel_name] = tuple(int(v) for v in mel_fixed.shape)

    stacked_tensor = np.stack(stacked_list, axis=0).astype(np.float32)
    if stacked_tensor.shape != (18, n_mels, expected_time_steps):
        raise ValueError(
            "Stacked spectrogram tensor shape mismatch: "
            f"expected (18, {n_mels}, {expected_time_steps}), got {stacked_tensor.shape}."
        )

    diagnostics = {
        "spectrogram_type": "librosa_mel_db",
        "spectrogram_channel_axis_order": "channel_mel_time",
        "spectrogram_channel_order": channel_names,
        "spectrogram_sampling_rate_hz": sampling_rate_hz,
        "spectrogram_n_mels": int(n_mels),
        "spectrogram_n_fft": int(n_fft),
        "spectrogram_hop_seconds": float(hop_seconds),
        "spectrogram_hop_length_samples": int(hop_length_samples),
        "spectrogram_f_min_hz": float(f_min_hz),
        "spectrogram_f_max_hz": float(f_max_hz),
        "spectrogram_time_steps_target": int(expected_time_steps),
        "spectrogram_pad_crop_policy": "right-pad with pad_value and right-crop to fixed time steps",
        "spectrogram_pad_value": float(pad_value),
        "spectrogram_per_channel_shape": per_channel_shapes,
        "spectrogram_stacked_shape": tuple(int(v) for v in stacked_tensor.shape),
        "spectrogram_nan_replaced_per_channel": nan_replaced_per_channel,
    }
    return by_channel, stacked_tensor, diagnostics


def _mne_info_for_signal(
    signal_uv: np.ndarray,
    metadata: dict[str, object],
) -> mne.Info:
    """Build an MNE Info object for a channel-major EEG array."""
    channel_names = list(metadata["channel_names"] if "channel_names" in metadata else [])
    if len(channel_names) != signal_uv.shape[0]:
        channel_names = [f"EEG {index:02d}" for index in range(signal_uv.shape[0])]

    return mne.create_info(
        ch_names=channel_names,
        sfreq=float(metadata["sampling_rate_hz"]),
        ch_types=["eeg"] * signal_uv.shape[0],
    )


def apply_bandpass_butterworth(
    signal_uv: np.ndarray,
    sampling_rate_hz: float,
    low_cut_hz: float = 0.5,
    high_cut_hz: float = 50.0,
    order: int = 2,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply band-pass filtering using MNE while preserving the notebook contract."""
    if sampling_rate_hz <= 0:
        raise ValueError(f"Sampling rate must be positive, got {sampling_rate_hz}.")
    if not (0.0 < low_cut_hz < high_cut_hz < (sampling_rate_hz / 2.0)):
        raise ValueError(
            "Invalid band-pass configuration: require "
            f"0 < low_cut_hz < high_cut_hz < Nyquist ({sampling_rate_hz / 2.0:.2f} Hz), got "
            f"low={low_cut_hz}, high={high_cut_hz}."
        )

    metadata = {
        "channel_names": [f"EEG {index:02d}" for index in range(signal_uv.shape[0])],
        "sampling_rate_hz": sampling_rate_hz,
    }
    info = _mne_info_for_signal(signal_uv=signal_uv, metadata=metadata)
    raw = mne.io.RawArray(signal_uv.astype(float), info, verbose="ERROR")

    raw.filter(
        l_freq=low_cut_hz,
        h_freq=high_cut_hz,
        method="iir",
        iir_params={"order": order, "ftype": "butter"},
        phase="zero",
        verbose="ERROR",
    )

    diagnostics = {
        "filter_name": "MNE IIR Butterworth Band-Pass",
        "filter_order": order,
        "filter_low_cut_hz": low_cut_hz,
        "filter_high_cut_hz": high_cut_hz,
        "filter_zero_phase": True,
        "filter_sampling_rate_hz": sampling_rate_hz,
        "filtered_channels": int(signal_uv.shape[0]),
        "skipped_channel_indices": [],
    }
    return raw.get_data(), diagnostics


def resample_to_target_hz(
    signal_uv: np.ndarray,
    metadata: dict[str, object],
    target_sampling_rate_hz: float = 128.0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Resample a channel-major EEG matrix to a target sampling rate using MNE."""
    current_sampling_rate_hz = float(metadata["sampling_rate_hz"])
    if current_sampling_rate_hz <= 0:
        raise ValueError(
            f"Current sampling rate must be positive, got {current_sampling_rate_hz}."
        )
    if target_sampling_rate_hz <= 0:
        raise ValueError(
            f"Target sampling rate must be positive, got {target_sampling_rate_hz}."
        )

    if np.isclose(current_sampling_rate_hz, target_sampling_rate_hz):
        resampled_signal = signal_uv.copy()
    else:
        info = _mne_info_for_signal(signal_uv=signal_uv, metadata=metadata)
        raw = mne.io.RawArray(signal_uv.astype(float), info, verbose="ERROR")
        raw.resample(sfreq=float(target_sampling_rate_hz), npad="auto", verbose="ERROR")
        resampled_signal = raw.get_data()

    expected_samples = int(
        round(signal_uv.shape[1] * target_sampling_rate_hz / current_sampling_rate_hz)
    )
    if resampled_signal.shape[0] != signal_uv.shape[0]:
        raise ValueError(
            "Resampled channel count mismatch: "
            f"expected {signal_uv.shape[0]}, got {resampled_signal.shape[0]}."
        )
    if expected_samples > 0 and resampled_signal.shape[1] != expected_samples:
        raise ValueError(
            "Resampled sample count mismatch: "
            f"expected {expected_samples}, got {resampled_signal.shape[1]}."
        )

    updated_metadata = dict(metadata)
    updated_metadata["sampling_rate_hz"] = float(target_sampling_rate_hz)
    updated_metadata["total_samples"] = int(resampled_signal.shape[1])
    if "calibrated_shape" in updated_metadata:
        updated_metadata["calibrated_shape"] = resampled_signal.shape
    if "standardized_monopolar_shape" in updated_metadata:
        updated_metadata["standardized_monopolar_shape"] = resampled_signal.shape
    if "bipolar_shape" in updated_metadata:
        updated_metadata["bipolar_shape"] = resampled_signal.shape
    updated_metadata["resampling_diagnostics"] = {
        "resample_method": "mne.io.RawArray.resample",
        "source_sampling_rate_hz": current_sampling_rate_hz,
        "target_sampling_rate_hz": float(target_sampling_rate_hz),
        "source_samples": int(signal_uv.shape[1]),
        "target_samples": int(resampled_signal.shape[1]),
        "resampled_channels": int(resampled_signal.shape[0]),
    }

    return resampled_signal, updated_metadata


def _find_true_runs(boolean_values: np.ndarray) -> list[tuple[int, int]]:
    """Find contiguous runs of True values in a 1D boolean array."""
    true_indices = np.flatnonzero(boolean_values)
    if true_indices.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    run_start = int(true_indices[0])
    run_end = int(true_indices[0])

    for index in true_indices[1:]:
        index = int(index)
        if index == run_end + 1:
            run_end = index
            continue
        runs.append((run_start, run_end))
        run_start = index
        run_end = index
    runs.append((run_start, run_end))
    return runs


def mask_flat_zero_windows(
    signal_uv: np.ndarray,
    metadata: dict[str, object],
    window_seconds: float = 10.0,
    zero_abs_tolerance_uv: float = 0.5,
    min_consecutive_windows: int = 1,
    require_all_channels_flat: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mask 10-second flat-zero-like windows so they are excluded before mel creation."""
    if signal_uv.ndim != 2:
        raise ValueError(f"Expected channel-major 2D signal, got shape {signal_uv.shape}.")

    sampling_rate_hz = float(metadata.get("sampling_rate_hz", np.nan))
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError(
            f"Invalid sampling_rate_hz={sampling_rate_hz} for flat-zero detection."
        )
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be > 0, got {window_seconds}.")
    if zero_abs_tolerance_uv < 0:
        raise ValueError(
            f"zero_abs_tolerance_uv must be >= 0, got {zero_abs_tolerance_uv}."
        )
    if min_consecutive_windows <= 0:
        raise ValueError(
            f"min_consecutive_windows must be >= 1, got {min_consecutive_windows}."
        )

    n_channels, n_samples = signal_uv.shape
    window_size_samples = int(round(window_seconds * sampling_rate_hz))
    if window_size_samples <= 0:
        raise ValueError(
            f"Computed invalid window_size_samples={window_size_samples}."
        )

    n_full_windows = n_samples // window_size_samples
    if n_full_windows == 0:
        diagnostics = {
            "flat_zero_enabled": True,
            "flat_zero_window_seconds": float(window_seconds),
            "flat_zero_window_size_samples": int(window_size_samples),
            "flat_zero_zero_abs_tolerance_uv": float(zero_abs_tolerance_uv),
            "flat_zero_min_consecutive_windows": int(min_consecutive_windows),
            "flat_zero_require_all_channels_flat": bool(require_all_channels_flat),
            "flat_zero_total_windows_checked": 0,
            "flat_zero_candidate_window_indices": [],
            "flat_zero_kept_window_indices": [],
            "flat_zero_masked_window_runs": [],
            "flat_zero_masked_sample_ranges": [],
            "flat_zero_masked_samples_total": 0,
            "flat_zero_masked_fraction": 0.0,
            "flat_zero_channel_flat_window_counts": {},
            "flat_zero_policy": "mask_to_nan",
        }
        return signal_uv.astype(float, copy=True), diagnostics

    n_channels, n_samples = signal_uv.shape
    channel_window_flat = np.zeros((n_channels, n_full_windows), dtype=bool)
    window_flat = np.zeros(n_full_windows, dtype=bool)

    for window_index in range(n_full_windows):
        sample_start = window_index * window_size_samples
        sample_end = sample_start + window_size_samples
        window_slice = signal_uv[:, sample_start:sample_end]

        per_channel_flat: list[bool] = []
        for channel_index in range(n_channels):
            channel_window = window_slice[channel_index]
            finite_values = channel_window[np.isfinite(channel_window)]
            if finite_values.size == 0:
                is_flat = True
            else:
                is_flat = bool(np.max(np.abs(finite_values)) <= zero_abs_tolerance_uv)
            per_channel_flat.append(is_flat)

        channel_window_flat[:, window_index] = np.array(per_channel_flat, dtype=bool)
        if require_all_channels_flat:
            window_flat[window_index] = bool(np.all(channel_window_flat[:, window_index]))
        else:
            window_flat[window_index] = bool(np.any(channel_window_flat[:, window_index]))

    candidate_runs = _find_true_runs(window_flat)
    kept_runs = [
        (start_idx, end_idx)
        for start_idx, end_idx in candidate_runs
        if (end_idx - start_idx + 1) >= min_consecutive_windows
    ]

    masked_signal_uv = signal_uv.astype(float, copy=True)
    masked_sample_ranges: list[tuple[int, int]] = []
    for start_window_index, end_window_index in kept_runs:
        sample_start = start_window_index * window_size_samples
        sample_end_exclusive = (end_window_index + 1) * window_size_samples
        masked_signal_uv[:, sample_start:sample_end_exclusive] = np.nan
        masked_sample_ranges.append((sample_start, sample_end_exclusive))

    masked_samples_total = int(sum(end - start for start, end in masked_sample_ranges))
    diagnostics = {
        "flat_zero_enabled": True,
        "flat_zero_window_seconds": float(window_seconds),
        "flat_zero_window_size_samples": int(window_size_samples),
        "flat_zero_zero_abs_tolerance_uv": float(zero_abs_tolerance_uv),
        "flat_zero_min_consecutive_windows": int(min_consecutive_windows),
        "flat_zero_require_all_channels_flat": bool(require_all_channels_flat),
        "flat_zero_total_windows_checked": int(n_full_windows),
        "flat_zero_candidate_window_indices": np.flatnonzero(window_flat).astype(int).tolist(),
        "flat_zero_kept_window_indices": sorted(
            {
                window_index
                for start_idx, end_idx in kept_runs
                for window_index in range(start_idx, end_idx + 1)
            }
        ),
        "flat_zero_masked_window_runs": [
            (int(start_idx), int(end_idx)) for start_idx, end_idx in kept_runs
        ],
        "flat_zero_masked_sample_ranges": [
            (int(start), int(end)) for start, end in masked_sample_ranges
        ],
        "flat_zero_masked_samples_total": int(masked_samples_total),
        "flat_zero_masked_fraction": float(masked_samples_total / n_samples),
        "flat_zero_channel_flat_window_counts": {
            str(channel_index): int(channel_window_flat[channel_index].sum())
            for channel_index in range(n_channels)
        },
        "flat_zero_policy": "mask_to_nan",
    }
    return masked_signal_uv, diagnostics

all_records = discover_icare_segments(DATA_ROOT)
print(f"Discovered paired segments: {len(all_records)}")

for signal_type in INCLUDED_SIGNAL_TYPES:
    count = sum(record.signal_type == signal_type for record in all_records)
    print(f"  {signal_type}: {count}")

if PREFERRED_PATIENTS:
    selected_examples = [
        record
        for record in all_records
        if record.signal_type in INCLUDED_SIGNAL_TYPES
        and record.patient_id in PREFERRED_PATIENTS
        and (PREFERRED_SEGMENT_ID is None or record.segment_id == PREFERRED_SEGMENT_ID)
    ]
else:
    selected_examples = select_static_examples(
        records=all_records,
        signal_types=INCLUDED_SIGNAL_TYPES,
        max_examples_per_type=MAX_EXAMPLES_PER_TYPE,
        preferred_patients=PREFERRED_PATIENTS,
        preferred_segment_id=PREFERRED_SEGMENT_ID,
    )

print(f"\nSelected static examples: {len(selected_examples)}")
selected_patient_ids = sorted({record.patient_id for record in selected_examples})
print(f"Selected patients: {selected_patient_ids if selected_patient_ids else 'None'}")
for record in selected_examples[:10]:
    print(
        f"  {record.patient_id}_{record.segment_id}_{record.hour_token}_{record.signal_type}"
    )
if len(selected_examples) > 10:
    print(f"  ... and {len(selected_examples) - 10} more")

print(f"\nPreprocessing mode: {PREPROCESS_MODE}")
print(
    f"Target sampling rate={TARGET_SAMPLING_RATE_HZ}Hz | "
    f"Band-pass filter enabled={ENABLE_BANDPASS_FILTER} "
    f"(order={FILTER_ORDER}, low={FILTER_LOW_HZ}Hz, high={FILTER_HIGH_HZ}Hz)"
)
print(
    f"Flat-zero elimination enabled={ENABLE_FLAT_ZERO_ELIMINATION} | "
    f"window={FLAT_ZERO_WINDOW_SECONDS}s | "
    f"tol={FLAT_ZERO_ABS_TOLERANCE_UV}uV | "
    f"min_consecutive={FLAT_ZERO_MIN_CONSECUTIVE_WINDOWS} | "
    f"all_channels={FLAT_ZERO_REQUIRE_ALL_CHANNELS_FLAT}"
)
print(
    f"Mel enabled={ENABLE_MEL_SPECTROGRAM} | mel_bins={MEL_N_MELS} | "
    f"hop={MEL_HOP_SECONDS}s | f_range=[{MEL_F_MIN_HZ}, {MEL_F_MAX_HZ}]Hz | "
    f"target_shape=({MEL_N_MELS}, {MEL_EXPECTED_TIME_STEPS})"
)

if not selected_examples:
    print("No examples selected. Adjust filters in the previous cell.")

expected_bipolar_channel_names = [f"{left}-{right}" for left, right in BIPOLAR_18_PAIRS]
segment_spectrograms_by_channel: dict[str, dict[str, np.ndarray]] = {}
segment_spectrogram_tensors: dict[str, np.ndarray] = {}

for record in selected_examples:
    print("\n" + "=" * 100)
    print(f"Loading {record.hea_path.name}")

    try:
        signal_uv, metadata = load_icare_segment(record.hea_path, record.mat_path)

        if PREPROCESS_MODE == "bipolar":
            signal_uv, metadata = prepare_bipolar_segment(signal_uv, metadata)

            if list(metadata["channel_names"]) != expected_bipolar_channel_names:
                raise ValueError(
                    "Bipolar channel order mismatch. "
                    f"Expected {expected_bipolar_channel_names} but got {metadata['channel_names']}."
                )

            pre_filter_signal_uv = signal_uv.copy()
            post_filter_signal_uv = signal_uv.copy()

            if ENABLE_BANDPASS_FILTER:
                post_filter_signal_uv, filter_diagnostics = apply_bandpass_butterworth(
                    signal_uv=signal_uv,
                    sampling_rate_hz=float(metadata["sampling_rate_hz"]),
                    low_cut_hz=FILTER_LOW_HZ,
                    high_cut_hz=FILTER_HIGH_HZ,
                    order=FILTER_ORDER,
                )
                metadata["filter_diagnostics"] = filter_diagnostics

            signal_uv = post_filter_signal_uv if ENABLE_BANDPASS_FILTER else pre_filter_signal_uv
            signal_uv, metadata = resample_to_target_hz(
                signal_uv=signal_uv,
                metadata=metadata,
                target_sampling_rate_hz=TARGET_SAMPLING_RATE_HZ,
            )

            if ENABLE_FLAT_ZERO_ELIMINATION:
                signal_uv, flat_zero_diagnostics = mask_flat_zero_windows(
                    signal_uv=signal_uv,
                    metadata=metadata,
                    window_seconds=FLAT_ZERO_WINDOW_SECONDS,
                    zero_abs_tolerance_uv=FLAT_ZERO_ABS_TOLERANCE_UV,
                    min_consecutive_windows=FLAT_ZERO_MIN_CONSECUTIVE_WINDOWS,
                    require_all_channels_flat=FLAT_ZERO_REQUIRE_ALL_CHANNELS_FLAT,
                )
                metadata["flat_zero_diagnostics"] = flat_zero_diagnostics

        elif PREPROCESS_MODE != "monopolar":
            raise ValueError(
                f"Unsupported PREPROCESS_MODE={PREPROCESS_MODE}. Use 'monopolar' or 'bipolar'."
            )

        segment_token = (
            f"{record.patient_id}_{record.segment_id}_{record.hour_token}_{record.signal_type}"
        )

        if ENABLE_MEL_SPECTROGRAM:
            spectrograms_by_channel, spectrogram_tensor, spectrogram_diagnostics = (
                create_bipolar_mel_spectrograms(
                    signal_uv=signal_uv,
                    metadata=metadata,
                    expected_channel_names=expected_bipolar_channel_names,
                    n_mels=MEL_N_MELS,
                    n_fft=MEL_N_FFT,
                    hop_seconds=MEL_HOP_SECONDS,
                    f_min_hz=MEL_F_MIN_HZ,
                    f_max_hz=MEL_F_MAX_HZ,
                    expected_sampling_rate_hz=TARGET_SAMPLING_RATE_HZ,
                    expected_time_steps=MEL_EXPECTED_TIME_STEPS,
                    pad_value=MEL_PAD_VALUE_DB,
                )
            )
            metadata["spectrogram_diagnostics"] = spectrogram_diagnostics
            segment_spectrograms_by_channel[segment_token] = spectrograms_by_channel
            segment_spectrogram_tensors[segment_token] = spectrogram_tensor

            first_channel_name = metadata["channel_names"][0]
            first_channel_shape = spectrograms_by_channel[first_channel_name].shape
            print(
                "spectrogram="
                f"channels={len(spectrograms_by_channel)} | "
                f"sample_channel_shape={first_channel_shape} | "
                f"stacked_shape={spectrogram_tensor.shape}"
            )

        summarize_segment(record, metadata)
        if "resampling_diagnostics" in metadata:
            diagnostics = metadata["resampling_diagnostics"]
            print(
                "resample="
                f"{diagnostics['resample_method']} | "
                f"source_fs={diagnostics['source_sampling_rate_hz']}Hz | "
                f"target_fs={diagnostics['target_sampling_rate_hz']}Hz | "
                f"source_samples={diagnostics['source_samples']} | "
                f"target_samples={diagnostics['target_samples']}"
            )
        if "filter_diagnostics" in metadata:
            diagnostics = metadata["filter_diagnostics"]
            print(
                "filter="
                f"{diagnostics['filter_name']} | "
                f"order={diagnostics['filter_order']} | "
                f"band=[{diagnostics['filter_low_cut_hz']}, {diagnostics['filter_high_cut_hz']}] Hz | "
                f"zero_phase={diagnostics['filter_zero_phase']} | "
                f"filtered_channels={diagnostics['filtered_channels']}"
            )
        if "flat_zero_diagnostics" in metadata:
            diagnostics = metadata["flat_zero_diagnostics"]
            print(
                "flat_zero="
                f"windows_checked={diagnostics['flat_zero_total_windows_checked']} | "
                f"candidates={len(diagnostics['flat_zero_candidate_window_indices'])} | "
                f"kept={len(diagnostics['flat_zero_kept_window_indices'])} | "
                f"masked_samples={diagnostics['flat_zero_masked_samples_total']} | "
                f"masked_fraction={diagnostics['flat_zero_masked_fraction']:.4f}"
            )

        
        n_channels = pre_filter_signal_uv.shape[0]

    except Exception as error:
        print(
            f"Failed for {record.patient_id}_{record.segment_id}_{record.signal_type}: {error}"
        )

output_dir = EXPORT_ROOT / EXPORT_SPLIT
output_dir.mkdir(parents=True, exist_ok=True)

manifest_path = output_dir / "manifest.csv"


def to_uint8_image(spec_2d: np.ndarray) -> np.ndarray:
    spec = spec_2d.astype(np.float32)
    if PER_IMAGE_NORMALIZE:
        vmin = float(np.nanmin(spec))
        vmax = float(np.nanmax(spec))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            scaled = np.zeros_like(spec, dtype=np.float32)
        else:
            scaled = (spec - vmin) / (vmax - vmin)
    else:
        clipped = np.clip(spec, DB_MIN, DB_MAX)
        scaled = (clipped - DB_MIN) / (DB_MAX - DB_MIN)

    img = (scaled * 255.0).clip(0, 255).astype(np.uint8)
    return img


rows = []
num_saved = 0
saved_by_patient: dict[str, int] = {}

# Uses your existing dictionary:
# segment_spectrograms_by_channel[segment_token][channel_name] -> (360, 360)
for segment_token, channel_map in segment_spectrograms_by_channel.items():
    patient_id = segment_token.split("_", 1)[0]
    if PREFERRED_PATIENTS and patient_id not in PREFERRED_PATIENTS:
        continue

    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    for channel_name, spec in channel_map.items():
        if spec.shape != (360, 360):
            continue

        safe_channel = re.sub(r"[^A-Za-z0-9._-]", "-", channel_name)
        filename = f"{segment_token}__{safe_channel}.png"
        file_path = patient_dir / filename

        img_u8 = to_uint8_image(spec)
        Image.fromarray(img_u8, mode="L").save(file_path)

        rows.append({
            "image_path": str(file_path.as_posix()),
            "image_path_relative": str(file_path.relative_to(output_dir).as_posix()),
            "patient_id": patient_id,
            "segment_token": segment_token,
            "channel_name": channel_name,
            "height": 360,
            "width": 360,
        })
        num_saved += 1
        saved_by_patient[patient_id] = saved_by_patient.get(patient_id, 0) + 1

with manifest_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "image_path",
            "image_path_relative",
            "patient_id",
            "segment_token",
            "channel_name",
            "height",
            "width",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {num_saved} images to: {output_dir}")
for patient_id in sorted(saved_by_patient):
    print(f"  {patient_id}: {saved_by_patient[patient_id]} images")
print(f"Manifest: {manifest_path}")