"""
Signal (audio) preprocessing / encoders ------------------------
Waveform to windowed spectral features and back again.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile
from scipy.signal import decimate

from polyergalio.encoders.encoders import Processor
from polyergalio.models.constants import EPSILON
from polyergalio.utilities import rolling_windows_nd, standardize_data


def downsample_sequence(
    data: NDArray, factor: int, mask: Optional[NDArray] = None
) -> tuple[NDArray, Optional[NDArray]]:
    """
    Downsample one step along the sequence axis, with anti-aliasing.

    Parameters
    ----------
    data : (batch, sequence, hidden)
    factor : integer decimation factor. scipy recommends chaining factors of
        13 or less rather than using one large factor.
    mask : optional (batch, sequence) or (batch, sequence, 1), 1 for real
        content, 0 for padding. A downsampled position is marked real if any
        of the original positions it covers were real -- with right-padding,
        that is exactly the positions before the true length.

    Returns
    -------
    downsampled data, and the corresponding downsampled mask (or None)
    """
    assert data.ndim == 3, f"expected (batch, sequence, hidden), got {data.shape}"
    assert factor >= 1, "factor must be a positive integer"

    if factor == 1:
        return data, mask

    downsampled = decimate(data, factor, axis=1, zero_phase=True)

    if mask is None:
        return downsampled, None

    positions = mask[..., 0] if mask.ndim == 3 else mask
    sequence = positions.shape[1]
    pad = (-sequence) % factor
    if pad:
        positions = np.pad(positions, ((0, 0), (0, pad)))
    blocks = positions.reshape(positions.shape[0], -1, factor)
    downsampled_mask = (blocks.sum(axis=-1) > 0).astype(mask.dtype)
    downsampled_mask = downsampled_mask[:, :downsampled.shape[1]]

    return downsampled, downsampled_mask


def progressive_downsample(
    data: NDArray, factors: list[int], mask: Optional[NDArray] = None
) -> list[tuple[NDArray, Optional[NDArray]]]:
    """
    BUILD a resolution pyramid by chaining downsample_sequence

    Each level downsamples the PREVIOUS level's output

    Parameters
    ----------
    data : (batch, sequence, hidden)
    factors : per-level downsampling factor, e.g. [2, 2, 2] for a
        1x, 1/2x, 1/4x, 1/8x pyramid
    mask : optional (batch, sequence) or (batch, sequence, 1)

    Returns
    -------
    list of (data, mask) pairs, from index 0 (full resolution) through the
    last entry (most downsampled)
    """
    levels = [(data, mask)]
    current_data, current_mask = data, mask
    for factor in factors:
        current_data, current_mask = downsample_sequence(current_data, factor, current_mask)
        levels.append((current_data, current_mask))
    return levels


def read_wav(
        file_path: str,
        channel: Optional[int] = 0,
        standardize: bool = True
) -> tuple[int, NDArray]:
    """
    Read a wav file and pull out one channel.

    Parameters
    ----------
    file_path : path to the wav file
    channel : channel index, or None to keep every channel
    standardize : zero mean and unit variance the samples

    Returns
    -------
    (sample_rate, waveform)
    """
    sample_rate, waveform = wavfile.read(file_path)

    if channel is not None and waveform.ndim > 1:
        waveform = waveform[:, channel]

    waveform = waveform.astype(np.float64)
    if standardize:
        waveform = standardize_data(waveform, axis=0)

    return sample_rate, waveform


def window_size_from_ms(sample_rate: int, window_ms: float) -> int:
    """number of samples spanned by a window of window_ms milliseconds"""
    return int(window_ms * sample_rate / 1000)


def build_windows(
    waveform: NDArray, window_size: int, num_overlap: int = 0
) -> NDArray:
    """
    Cut the waveform into overlapping windows, shaped (num_frames, window_size).
    Every downstream stage works from these windowed timesteps, so the spectrogram and the
    time domain windows always describe the same segments of signal.
    """
    if waveform.ndim == 2:  # if we have 2 dimensions (batch, amplitude)
        axis = 1
    elif waveform.ndim == 1:
        axis = 0
    return rolling_windows_nd(
        data=waveform, window_size=window_size, num_overlap=num_overlap, axis=axis
    )


def windowed_spectrum(
        windows: NDArray,
        window_kernel: Optional[NDArray] = None
) -> NDArray:
    """
    Real FFT of each window, tapered first to stop the window edges ringing.

    Returns the complex spectrum, shaped (..., window_size // 2 + 1)
    """
    if window_kernel is None:
        window_kernel = np.blackman(windows.shape[-1])

    return np.fft.rfft(windows * window_kernel, axis=-1)


def _paired_bins(num_frequencies: int, window_size: Optional[int]) -> slice:
    """
    which rfft bins stand for a conjugate pair, and so carry double the energy
    of a two sided spectrum
    """
    if window_size is None:
        window_size = 2 * (num_frequencies - 1)

    nyquist_present = window_size % 2 == 0
    return slice(1, -1 if nyquist_present else None)


def to_power(
    spectrum: NDArray, window_size: Optional[int] = None, one_sided: bool = True
) -> NDArray:
    """
    Power spectrum, |X|^2 for every bin.

    Parameters
    ----------
    spectrum : complex rfft output
    window_size : length of the window the spectrum came from. Defaults to the
        even case, so pass it explicitly whenever the window length is odd.
    one_sided : double the paired bins so the total matches the energy of the
        full two sided spectrum
    """
    power = np.abs(spectrum) ** 2

    if one_sided:
        power[..., _paired_bins(spectrum.shape[-1], window_size)] *= 2

    return power


def to_decibels(power: NDArray, epsilon: float = EPSILON) -> NDArray:
    """decibels from a power spectrum, 10 * log_base_10, floored by epsilon"""
    return 10 * np.log10(power + epsilon)


def from_decibels(decibels: NDArray, epsilon: float = EPSILON) -> NDArray:
    """inverse of to_decibels"""
    return np.maximum(10 ** (decibels / 10) - epsilon, 0.0)


def frequency_axis(window_size: int, sample_rate: int) -> NDArray:
    """centre frequency in Hz of each rfft bin"""
    return np.fft.rfftfreq(window_size, d=1.0 / sample_rate)


# -------------    windows back to a waveform    --------------------
def overlap_add(
    windows: NDArray, num_overlap: int = 0, window_kernel: Optional[NDArray] = None
) -> NDArray:
    """
    Fold overlapping windows back into one signal, weighted so that the taper
    applied on the way in is divided out on the way back.
    """
    num_windows, window_size = windows.shape
    stride = window_size - num_overlap

    if window_kernel is None:
        window_kernel = np.ones(window_size)

    length = stride * (num_windows - 1) + window_size
    signal = np.zeros(length)
    weight = np.zeros(length)

    for i in range(num_windows):
        start = i * stride
        signal[start: start + window_size] += windows[i] * window_kernel
        weight[start: start + window_size] += window_kernel ** 2

    return signal / np.maximum(weight, EPSILON)


# -------------    plotting    -------------------------------------
def plot_waveform(waveform: NDArray, sample_rate: Optional[int] = None) -> None:
    """time domain view of one channel"""
    xs = np.arange(waveform.shape[0])
    label = "Sample"
    if sample_rate is not None:
        xs = xs / sample_rate
        label = "Time (seconds)"

    plt.figure(figsize=(10, 4))
    plt.plot(xs, waveform)
    plt.title("Audio waveform")
    plt.xlabel(label)
    plt.ylabel("Amplitude")
    plt.axis("tight")
    plt.show()


def plot_spectrogram(
    decibels: NDArray, freqs: NDArray, sample_rate: Optional[int] = None
) -> None:
    """decibel spectrogram, frames on x and frequency on y"""
    time_axis = np.arange(decibels.shape[0])
    label = "Frame"

    plt.figure(figsize=(15, 8))
    plt.pcolormesh(time_axis, freqs, decibels.T, shading="auto")
    plt.xlabel(label)
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram")
    plt.colorbar(label="dB")
    plt.show()


def plot_window_spectra(
    spectrum: NDArray,
    freqs: NDArray,
    index: int = 0,
    window_size: Optional[int] = None,
) -> None:
    """
    Amplitude, power and decibel view of a single frame. Replaces the older
    plot_fft_windows, which assumed three named axes and a fixed subplot grid.
    """
    magnitude = np.abs(spectrum[index])
    power = to_power(spectrum[index: index + 1], window_size=window_size)[0]
    decibels = to_decibels(power)

    panels = (
        ("Amplitude spectrum", magnitude, "amplitude", plt.plot),
        ("Power spectrum", power, "power", plt.plot),
        ("Decibel spectrum", decibels, "dB", plt.plot),
    )

    plt.figure(figsize=(14, 10))
    for position, (title, values, ylabel, draw) in enumerate(panels, start=1):
        plt.subplot(len(panels), 1, position)
        draw(freqs, values)
        plt.title(f"{title}, frame {index}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel(ylabel)
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
class AudioProcessor(Processor):
    """
    Waveform to a windowed spectral representation.

    channel is the channel to pull out of a multi channel file.
    """

    def __init__(
        self,
        sample_rate: int,
        window_ms: float = 20.0,
        overlap_ratio: float = 1 / 3,
        use_decibels: bool = True,
        target: str = "waveform",
        channel: int = 0,
    ):
        """
        Parameters
        ----------
        sample_rate : samples per second of the source signal
        window_ms : frame length in milliseconds
        overlap_ratio : fraction of a frame shared with the next frame
        use_decibels : encode to decibels rather than raw power
        target : name of the signal being encoded
        channel : channel index within a multi channel source
        """
        super().__init__(target=target)
        self.channel = channel
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.overlap_ratio = overlap_ratio
        self.use_decibels = use_decibels

        self.window_size: int = window_size_from_ms(sample_rate, window_ms)
        self.num_overlap: int = int(self.window_size * overlap_ratio)
        self.window_kernel: NDArray = np.blackman(self.window_size)
        self.freqs: NDArray = frequency_axis(self.window_size, sample_rate)
        self.spectrum: Optional[NDArray] = None


    def fit(self, values: NDArray) -> "AudioProcessor":
        """record the observed range, which inverse() needs to undo scaling"""
        self.obs_min_max = (float(np.min(values)), float(np.max(values)))
        self.fitted = True
        return self

    def encode(self, values: NDArray) -> dict[str,NDArray]:
        """
        Waveform to (num_frames, num_frequencies). The complex spectrum is
        cached on the way through so inverse() can reuse its phase.
        """
        windows = build_windows(values, self.window_size, self.num_overlap)
        power = to_power(windows, window_size=self.window_size)

        self.spectrum = windowed_spectrum(power, self.window_kernel)

        if self.use_decibels:
            return {"spectrum": to_decibels(self.spectrum), "amplitude": to_decibels(power)}

        return {"spectrum": self.spectrum, "amplitude": power}


    def fit_encode(self, values: NDArray) -> NDArray:
        return self.fit(values).encode(values)

    def inverse(self, values: NDArray) -> NDArray:
        """
        Spectral features back to a waveform.

        Magnitude alone does not determine a signal, so the phase cached by the
        last encode() is reused when its shape matches. Without that cache the
        reconstruction is zero phase and only approximate.
        """
        power = from_decibels(values) if self.use_decibels else values

        two_sided = power.copy()
        two_sided[..., _paired_bins(power.shape[-1], self.window_size)] /= 2

        magnitude = np.sqrt(two_sided)

        if self.spectrum is not None and self.spectrum.shape == magnitude.shape:
            phase = np.exp(1j * np.angle(self.spectrum))
        else:
            phase = 1.0

        windows = np.fft.irfft(magnitude * phase, n=self.window_size, axis=-1)
        return overlap_add(windows, self.num_overlap, self.window_kernel)

    @property
    def metadata(self) -> dict:
        return {
            "target": self.target,
            "channel": self.channel,
            "sample_rate": self.sample_rate,
            "window_ms": self.window_ms,
            "window_size": self.window_size,
            "num_overlap": self.num_overlap,
            "num_frequencies": len(self.freqs),
            "use_decibels": self.use_decibels,
        }

    def __str__(self):
        units = "dB" if self.use_decibels else "power"
        return (
            f"AudioProcessor at {self.sample_rate} Hz, {self.window_size} sample "
            f"windows overlapping {self.num_overlap}, encoding to {units}"
        )
