import math

import librosa.onset
import numpy as np
import scipy.signal.windows
from numpy.lib import stride_tricks

import pyaudio
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt


class AudioHandler(object):
    def __init__(self):
        # pyaudio
        self.filter_coef = None
        self.threshold = None
        self.rms = None
        self.spectrogram_t = None
        self.spectrogram_f = None
        self.spectrogram_A = None
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 22050
        self.CHUNK = 1024 * 1
        self.WINDOW_SIZE = 512
        self.HOP_LENGTH = 512
        self.BUFFER_BLOCKS = int(math.ceil(self.RATE / 2 / self.CHUNK))
        self.p = None
        self.stream = None

        # Create a buffer for storing 500ms worth of signal
        self.data_buffer = np.zeros(self.BUFFER_BLOCKS * self.CHUNK)
        self.times = None

        self.ring_buffer_index = 0
        self.ring_buffer_full = False

        # Create a highpass filter that cuts off a 200Hz
        self.filter_coef = self.create_butter_highpass(250)

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        # Store the waveform data for this update
        wf_data = np.frombuffer(in_data, dtype=np.float32)
        filtered_data = filtfilt(self.filter_coef[0], self.filter_coef[1], wf_data)

        if self.ring_buffer_full:
            self.data_buffer = np.roll(self.data_buffer, -frame_count)
            self.data_buffer[-frame_count::] = filtered_data
        else:
            index = self.ring_buffer_index * frame_count
            self.data_buffer[index:index + frame_count] = filtered_data
            self.ring_buffer_index += 1
            if self.ring_buffer_index == self.BUFFER_BLOCKS:
                # Buffer has filled, mark it full and collect a baseline of noise
                self.ring_buffer_full = True
                print(f'Ring buffer filled')

        # Compute the STFT of the buffer after this update for use by the following analysis
        Sc = librosa.stft(y=self.data_buffer, n_fft=2048, hop_length=self.HOP_LENGTH, center=True,
                     window=scipy.signal.windows.blackman)
        self.times = librosa.times_like(X=Sc, sr=self.RATE, n_fft=2048, hop_length=self.HOP_LENGTH)
        S = np.abs(Sc)

        # Use the STFT to compute onsets
        onset_env = librosa.onset.onset_strength(S=S, sr=self.RATE, center=True)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.RATE, hop_length=self.HOP_LENGTH,
                                            backtrack=True)

        onstm = librosa.frames_to_time(onsets, sr=self.RATE)

        # Calculate RMS energy per frame. The frame length from the default value
        # in order to avoid ending up with too much smoothing
        self.rms = librosa.feature.rms(S=S, hop_length=self.HOP_LENGTH)[0, ]
        envtm = librosa.frames_to_time(np.arange(len(self.rms)), sr=self.RATE)

        if self.threshold is not None:
            thresh_idx = (self.rms > self.threshold).astype(int)
            bool_indices = [tm in envtm[thresh_idx] for tm in onstm]
            indices = np.where(bool_indices)[0]
            correctedonstm = onstm[indices]
            if len(correctedonstm) > 0:
                print(f'Thresh Index: {thresh_idx}')
                print(f'Bool Indices: {bool_indices}')
                print(f'Indices: {indices}')
                print(f'Threshold Times: {correctedonstm}')
        # self.spectrogram_f, self.spectrogram_t, self.spectrogram_A = signal.spectrogram(self.data_buffer, self.RATE,
        #                                                                                detrend=False,
        #                                                                                mode='psd')

        return None, pyaudio.paContinue

    def capture_rms_baseline(self):
        noise_median = np.percentile(self.rms[::], 50)
        sigma = np.percentile(self.rms[::], 84.1) - noise_median
        # Set the minimum RMS energy threshold that is needed in order to declare
        # an "onset" event to be equal to 5 sigma above the median
        self.threshold = noise_median + 5 * sigma
        print(f'RMS Threshold: {self.threshold}')

    def create_butter_highpass(self, cutoff, order=5):
        """
        Design a highpass filter.
        Args:
        - cutoff (float) : the cutoff frequency of the filter.
        - fs     (float) : the sampling rate.
        - order    (int) : order of the filter, by default defined to 5.
        """
        # calculate the Nyquist frequency
        nyq = 0.5 * self.RATE
        # design filter
        high = cutoff / nyq
        return butter(order, high, btype='high', analog=False)

    # def determine_window_range(self):
    #     # Total number of blocks in the ring buffer
    #     max_count = self.ring_buffer_view.shape[0]
    #
    #     # Determine the number of filled blocks in the ring buffer
    #     if self.ring_buffer_full:
    #         count = max_count
    #     else:
    #         # The number of full blocks on the front side of the buffer
    #         count = 0 if self.ring_buffer_index == 0 else self.ring_buffer_index - 1
    #
    #     # Determine indices. This should start at the next write position (oldest data) and go to the most recent block
    #     return np.append(np.arange(self.ring_buffer_index, count), np.arange(self.ring_buffer_index))

    def get_latest_waveform(self):
        return self.data_buffer, np.linspace(self.times[0], self.times[-1], len(self.data_buffer))

    def get_latest_rms(self):
        return self.rms, self.times

    def get_latest_spectrogram(self):
        return [self.spectrogram_t, self.spectrogram_f, self.spectrogram_A]
