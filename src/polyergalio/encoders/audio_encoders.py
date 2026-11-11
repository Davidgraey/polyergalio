"""
Signal (audio) preprocessing / encoders ------------------------
Waveform to windowed spectral features and back again.
"""

from typing import Optional, Sequence

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
    construct a resolution pyramid by chaining downsample_sequence

    NOTE - Each level downsamples the PREVIOUS level's output

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
    spectrum: NDArray,
    window_size: Optional[int] = None,
    one_sided: bool = True,
    window_kernel: Optional[NDArray] = None,
) -> NDArray:
    """
    Power spectrum, |X|squared for every bin, optionally normalized by window energy.

    Parameters
    ----------
    spectrum : complex rfft output
    window_size : length of the window the spectrum came from. Defaults to the even case, so pass it explicitly
        whenever the window length is odd.
    one_sided : double the paired bins so the total matches the energy of the full two sided spectrum
    window_kernel : taper the spectrum was computed with. Power is divided by its energy, sum(kernel**2), so
        unit variance noise has unit power per bin whatever the window length or taper.
    """
    power = np.abs(spectrum) ** 2

    if window_kernel is not None:
        power = power / np.sum(window_kernel ** 2)

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


def band_edges(
    freqs: NDArray,
    num_bands: int,
    min_frequency: Optional[float] = None,
    max_frequency: Optional[float] = None,
    min_bins: int = 1,
) -> NDArray:
    """
    Bin index edges of log spaced bands. Where log spacing would give a band fewer than `min_bins`
    bins (the low frequencies), bands are widened to `min_bins`, so the lowest bands keep full
    bin resolution and the log spacing takes over as the bins get relatively narrower.

    Parameters
    ----------
    freqs : centre frequency in Hz of each rfft bin
    num_bands : number of bands
    min_frequency : lowest frequency covered, defaults to the first bin above DC
    max_frequency : highest frequency covered, defaults to Nyquist
    min_bins : fewest rfft bins a band may span

    Returns
    -------
    (num_bands + 1,) integer bin edges, band i spans bins edges[i]:edges[i + 1]
    """
    low = 1 if min_frequency is None else max(1, int(np.searchsorted(freqs, min_frequency)))
    high = len(freqs) if max_frequency is None else int(np.searchsorted(freqs, max_frequency, side="right"))
    assert high - low >= num_bands * min_bins, (
        f"{num_bands} bands of {min_bins} bins need {num_bands * min_bins} bins, "
        f"only {high - low} between {freqs[low]:.1f} and {freqs[high - 1]:.1f} Hz"
    )

    targets = np.geomspace(freqs[low], freqs[high - 1], num_bands + 1)
    edges = np.searchsorted(freqs, targets).astype(int)
    edges[0] = low

    for i in range(1, num_bands + 1):
        edges[i] = max(edges[i], edges[i - 1] + min_bins)
    edges[-1] = high
    for i in range(num_bands - 1, 0, -1):
        edges[i] = min(edges[i], edges[i + 1] - min_bins)

    return edges


def band_matrix(edges: NDArray, num_frequencies: int) -> NDArray:
    """(num_bands, num_frequencies) matrix whose rows average the bins of each band"""
    matrix = np.zeros((len(edges) - 1, num_frequencies))
    for band, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        matrix[band, start:stop] = 1.0 / (stop - start)
    return matrix


def band_pool(power: NDArray, weights: NDArray) -> NDArray:
    """
    Average power across the bins of each band. Pool before taking decibels: averaging power
    removes the per-bin chi-squared scatter that averaging decibels would keep.

    Parameters
    ----------
    power : (..., num_frequencies)
    weights : (num_bands, num_frequencies) from band_matrix

    Returns
    -------
    (..., num_bands)
    """
    return power @ weights.T


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
    Waveform to a windowed spectral representation, windowed power, or windowed amplitude
    Windows are tiled across the `overlap_ratio`

    Spectral -
    Uses a Blackman kernel to reduce artifacts

    channel is the channel to pull out of a multi-channel file (L, R or other)
    """
    state_names = ("mean", "standard_deviation", "band_mean", "band_std")

    def __init__(
        self,
        sample_rate: int,
        window_ms: float = 20.0,
        overlap_ratio: float = 1 / 5,
        use_decibels: bool = True,
        target: str = "waveform",
        channel: int = 0,
        standardize: bool = True,
        num_bands: int = 64,
        min_frequency: Optional[float] = None,
        max_frequency: Optional[float] = 10_000.0,
        min_bins: int = 1,
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
        num_bands : number of log spaced bands in the "bands" encoding
        min_frequency : lowest band frequency in Hz, defaults to the first bin above DC
        max_frequency : highest band frequency in Hz
        min_bins : fewest rfft bins a band may span, low bands are widened to this
        """
        super().__init__(target=target)
        self.channel = channel
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.overlap_ratio = overlap_ratio
        self.use_decibels = use_decibels

        self.mean = 0
        self.standard_deviation = 1
        self.seen_samples = 0

        self.window_size: int = window_size_from_ms(sample_rate, window_ms)
        self.num_overlap: int = int(self.window_size * (overlap_ratio / 2))
        self.window_kernel: NDArray = np.blackman(self.window_size)
        self.freqs: NDArray = frequency_axis(self.window_size, sample_rate)
        self.spectrum: Optional[NDArray] = None
        self.standardize = standardize

        self.num_bands = num_bands
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_bins = min_bins
        edges = band_edges(self.freqs, num_bands, min_frequency, max_frequency, min_bins)
        self.band_weights: NDArray = band_matrix(edges, len(self.freqs))
        self.band_frequencies: NDArray = self.band_weights @ self.freqs
        self.band_mean = np.zeros(num_bands)
        self.band_std = np.ones(num_bands)


    def fit(self, values: NDArray | Sequence[NDArray]) -> "AudioProcessor":
        """
        Record the waveform mean and std, and the per band mean and std of the band decibels.

        Parameters
        ----------
        values : one waveform, or a list of waveforms, all at this processor's sample rate.
            Statistics are computed across all of them, replacing any earlier fit.
        """
        waveforms = [values] if isinstance(values, np.ndarray) else list(values)

        count = sum(waveform.size for waveform in waveforms)
        self.mean = float(sum(np.nansum(waveform) for waveform in waveforms) / count)
        squares = sum(np.nansum((waveform - self.mean) ** 2) for waveform in waveforms)
        self.standard_deviation = float(np.sqrt(squares / count))
        self.seen_samples = count
        self.obs_min_max = (
            float(min(np.min(waveform) for waveform in waveforms)),
            float(max(np.max(waveform) for waveform in waveforms)),
        )

        self.band_mean = np.zeros(self.num_bands)
        self.band_std = np.ones(self.num_bands)
        frames = 0
        band_sum = np.zeros(self.num_bands)
        band_squares = np.zeros(self.num_bands)
        for waveform in waveforms:
            bands = self.encode(waveform)["bands"].reshape(-1, self.num_bands)
            frames += len(bands)
            band_sum += bands.sum(axis=0)
            band_squares += (bands ** 2).sum(axis=0)

        self.band_mean = band_sum / frames
        self.band_std = np.sqrt(np.maximum(band_squares / frames - self.band_mean ** 2, 0.0))
        self.fitted = True
        return self

    def encode(self, values: NDArray) -> dict[str,NDArray]:
        """
        Waveform to windowed features. The complex spectrum is cached on the
        way through so inverse() can reuse its phase.

        Returns
        -------
        "spectrum" : (num_frames, num_frequencies), window energy normalized power
        "bands" : (num_frames, num_bands), log spaced band power, standardized per band
        "power" : (num_frames, window_size), squared samples
        "amplitude" : (num_frames, window_size), samples
        """
        _values =  (values - self.mean) / (self.standard_deviation + EPSILON)

        windows = build_windows(_values, self.window_size, self.num_overlap)

        self.spectrum = windowed_spectrum(windows, self.window_kernel)
        spectral_power = to_power(
            self.spectrum, window_size=self.window_size, window_kernel=self.window_kernel
        )
        bands = band_pool(spectral_power, self.band_weights)
        power = windows ** 2

        if self.use_decibels:
            spectral_power = to_decibels(spectral_power)
            bands = to_decibels(bands)
            power = to_decibels(power)

        bands = (bands - self.band_mean) / (self.band_std + EPSILON)

        return {"spectrum": spectral_power, "bands": bands, "power": power, "amplitude": windows}

    def windowed_spectral(self, values: NDArray, minimum_window_size: int = 100) -> NDArray:
        """
        Dyadic multi-resolution spectrum. Level k cuts the signal into 2**k
        windows (1/1, 1/2 | 1/2, 1/4 | 1/4 | 1/4 | 1/4, ...) down to
        `minimum_window_size` samples.

        Each window is zero padded to the full signal length before the FFT, so
        every level shares one frequency axis whatever the window length.

        Parameters
        ----------
        values : (..., length)
        minimum_window_size : smallest window length, in samples

        Returns
        -------
        complex spectrum, (..., num_windows, length // 2 + 1), coarsest level first
        """
        length = values.shape[-1]
        assert length >= minimum_window_size, "signal is shorter than minimum_window_size"

        levels = []
        count = 1

        while length // count >= minimum_window_size:
            size = length // count
            windows = values[..., : count * size].reshape(*values.shape[:-1], count, size)
            tapered = windows * np.blackman(size)
            padded = np.pad(tapered, [(0, 0)] * (tapered.ndim - 1) + [(0, length - size)])
            levels.append(np.fft.rfft(padded, axis=-1))
            count *= 2

        return np.concatenate(levels, axis=-2)

    def fit_encode(self, values: NDArray) -> dict[str, NDArray]:
        return self.fit(values).encode(values)

    def inverse(self, values: NDArray) -> NDArray:
        """
        Spectral features back to a waveform.

        Magnitude alone does not determine a signal, so the phase cached by the
        last encode() is reused when its shape matches. Without that cache the
        reconstruction is zero phase and only approximate.
        """
        power = from_decibels(values) if self.use_decibels else values

        two_sided = power * np.sum(self.window_kernel ** 2)
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
            "num_bands": self.num_bands,
            "band_frequencies": self.band_frequencies.tolist(),
            "use_decibels": self.use_decibels,
        }

    def __str__(self):
        units = "dB" if self.use_decibels else "power"
        return (
            f"AudioProcessor at {self.sample_rate} Hz, {self.window_size} sample "
            f"windows overlapping {self.num_overlap}, encoding to {units}"
        )

class FrequencyMatroyskaProcessor(Processor):
    """
    Waveform to a stacked multi-resolution spectrum, (batch, stacked_windows, frequencies).

    The signal is cut into 1, 2, 4, ... windows down to `minimum_window_ms`.
    Each window is Blackman tapered and zero padded, so all windows share one
    frequency axis.

    Alongside the dB SPL spectrum, window energy normalized power is pooled into log spaced
    bands and standardized per band.
    """
    state_names = ("mean", "standard_deviation", "band_mean", "band_std")

    def __init__(
        self,
        sample_rate: int,
        minimum_window_ms: float = 20.0,
        use_decibels: bool = True,
        target: str = "waveform",
        channel: int = 0,
        n_fft: Optional[int] = None,
        num_bands: int = 64,
        min_frequency: Optional[float] = None,
        max_frequency: Optional[float] = 10_000.0,
        min_bins: int = 1,
    ):
        """
        Parameters
        ----------
        sample_rate : samples per second of the source signal
        minimum_window_ms : length of the smallest window in milliseconds
        use_decibels : encode to decibels rather than raw power
        target : name of the signal being encoded
        channel : channel index within a multi channel source
        n_fft : FFT length every window is zero padded to, defaults to sample_rate (1 Hz bins)
        num_bands : number of log spaced bands in the "bands" encoding
        min_frequency : lowest band frequency in Hz, defaults to the first bin above DC
        max_frequency : highest band frequency in Hz
        min_bins : fewest bins a band may span, low bands are widened to this
        """
        super().__init__(target=target)
        self.channel = channel
        self.sample_rate = sample_rate
        self.minimum_window_ms = minimum_window_ms
        self.minimum_window_size: int = window_size_from_ms(sample_rate, minimum_window_ms)
        self.use_decibels = use_decibels
        self.n_fft: int = sample_rate if n_fft is None else n_fft

        self.volts_to_pascals = 10.0
        self.p_ref = 20e-6
        self.noise_floor_threshold_db = 40
        
        
        self.mean = 0
        self.standard_deviation = 1
        self.seen_samples = 0
        self.frequencies: NDArray = frequency_axis(self.n_fft, sample_rate)

        self.num_bands = num_bands
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_bins = min_bins
        edges = band_edges(self.frequencies, num_bands, min_frequency, max_frequency, min_bins)
        self.band_weights: NDArray = band_matrix(edges, len(self.frequencies))
        self.band_frequencies: NDArray = self.band_weights @ self.frequencies
        self.band_mean = np.zeros(num_bands)
        self.band_std = np.ones(num_bands)

    def fit(self, values: NDArray | Sequence[NDArray]) -> "FrequencyMatroyskaProcessor":
        """
        Record the waveform mean and std, and the per band mean and std of the band decibels.

        Parameters
        ----------
        values : one waveform, or a list of waveforms, all at this processor's sample rate.
            Statistics are computed across all of them, replacing any earlier fit.
        """
        waveforms = [values] if isinstance(values, np.ndarray) else list(values)

        count = sum(waveform.size for waveform in waveforms)
        self.mean = float(sum(np.nansum(waveform) for waveform in waveforms) / count)
        squares = sum(np.nansum((waveform - self.mean) ** 2) for waveform in waveforms)
        self.standard_deviation = float(np.sqrt(squares / count))
        self.seen_samples = count
        self.obs_min_max = (
            float(min(np.min(waveform) for waveform in waveforms)),
            float(max(np.max(waveform) for waveform in waveforms)),
        )

        self.band_mean = np.zeros(self.num_bands)
        self.band_std = np.ones(self.num_bands)
        frames = 0
        band_sum = np.zeros(self.num_bands)
        band_squares = np.zeros(self.num_bands)
        for waveform in waveforms:
            bands = self.encode(waveform)["bands"].reshape(-1, self.num_bands)
            frames += len(bands)
            band_sum += bands.sum(axis=0)
            band_squares += (bands ** 2).sum(axis=0)

        self.band_mean = band_sum / frames
        self.band_std = np.sqrt(np.maximum(band_squares / frames - self.band_mean ** 2, 0.0))
        self.fitted = True
        return self

    def encode(self, signal: NDArray) -> dict[str, NDArray]:
        """
        Waveform to a stacked spectrum.

        Parameters
        ----------
        signal : (batch, length) or (length,)

        Returns
        -------
        "spectrum" : (batch, stacked_windows, n_fft // 2 + 1), dB SPL
        "bands" : (batch, stacked_windows, num_bands), log spaced band power, standardized per band
        "frequency_bins" : (n_fft // 2 + 1,), frequency in Hz of each bin
        "resolutions" : window count of each level
        """
        signal = np.atleast_2d(signal)

        # standardized = (signal - self.mean) / (self.standard_deviation + EPSILON)
        spectrum, power, resolutions = self.windowed_spectral(signal, self.minimum_window_size, self.n_fft)

        bands = band_pool(power, self.band_weights)
        if self.use_decibels:
            bands = to_decibels(bands)
        bands = (bands - self.band_mean) / (self.band_std + EPSILON)

        return {
            "spectrum": spectrum,
            "bands": bands,
            "frequency_bins": self.frequencies,
            "resolutions": resolutions
        }

    def windowed_spectral(
        self, values: NDArray, minimum_window_size: int = 100, n_fft: Optional[int] = None
    ) -> tuple[NDArray, NDArray, list[int]]:
        """
        Dyadic multi-resolution spectrum. Level k cuts the signal into 2**k
        windows (1/1, 1/2 | 1/2, 1/4 | 1/4 | 1/4 | 1/4, ...) down to
        `minimum_window_size` samples.

        The signal is zero padded to a multiple of the finest window count, so
        every level divides evenly. Each tapered window is then zero padded to
        `n_fft`, or folded down to it when longer (time aliasing, which samples
        the window's spectrum at `n_fft` evenly spaced frequencies), so every
        level shares one frequency axis whatever the signal length.

        Parameters
        ----------
        values : (..., length)
        minimum_window_size : smallest window length, in samples
        n_fft : FFT length, fixes the number of frequency bins. Defaults to the padded signal length.

        Returns
        -------
        spectrum, (..., num_windows, n_fft // 2 + 1), dB SPL, coarsest level first
        power, same shape, normalized by window energy and signal variance so unit variance noise has unit
            power in every bin at every level
        resolutions, window count of each level
        """
        length = values.shape[-1]
        assert length >= minimum_window_size, "signal is shorter than minimum_window_size"

        # fit into the power of 2--
        max_count = 1 << ((length // minimum_window_size).bit_length() - 1)

        padded_length = -(-length // max_count) * max_count

        n_fft = padded_length if n_fft is None else n_fft

        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, padded_length - length)])

        levels = []
        powers = []
        count = 1
        resolutions = []
        while count <= max_count:
            size = padded_length // count
            windows = values.reshape(*values.shape[:-1], count, size)

            tapered = windows * np.hanning(size)
            folds = -(-size // n_fft)
            padded = np.pad(tapered, [(0, 0)] * (tapered.ndim - 1) + [(0, folds * n_fft - size)])
            padded = padded.reshape(*tapered.shape[:-1], folds, n_fft).sum(axis=-2)
            
            # -----
            fft_output = np.fft.rfft(padded, axis=-1)
            window_energy = np.sum(np.hanning(size) ** 2)
            powers.append(
                to_power(fft_output, window_size=n_fft) / (window_energy * (self.standard_deviation + EPSILON) ** 2)
            )
            frequencies = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)
            positive_indices = frequencies >= 0

            # positive half of magnitudes (over size)
            magnitudes = np.abs(fft_output) / size
            pressure_pascals = magnitudes * self.volts_to_pascals
            pressure_pascals = np.where(pressure_pascals == 0, 1e-15, pressure_pascals)
            magnitude_db_spl = 20 * np.log10(pressure_pascals / self.p_ref)
            masked_db_spl = np.where(
                magnitude_db_spl < self.noise_floor_threshold_db,
                self.noise_floor_threshold_db,
                magnitude_db_spl,
            )
            
            freqs_masked = frequencies[positive_indices]
            db_masked = masked_db_spl[..., positive_indices]
            # print(freqs_masked.shape, db_masked.shape)
            
            levels.append(db_masked)
            resolutions.append(count)

            count *= 2
            

        return np.concatenate(levels, axis=-2), np.concatenate(powers, axis=-2), resolutions

    def fit_encode(self, values: NDArray) -> dict[str, NDArray]:
        return self.fit(values).encode(values)

    def inverse(self, values: NDArray) -> NDArray:
        """
        """
        return ()

    @property
    def metadata(self) -> dict:
        return {
            "target": self.target,
            "channel": self.channel,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "num_bands": self.num_bands,
            "band_frequencies": self.band_frequencies.tolist(),
            "minimum_window_size": self.minimum_window_size,
            "use_decibels": self.use_decibels,
        }

    def __str__(self):
        units = "dB" if self.use_decibels else "power"
        return (
            f"FrequencyMatroyskaProcessor at {self.sample_rate} Hz, windows down to "
            f"{self.minimum_window_size} samples, encoding to {units}"
        )
