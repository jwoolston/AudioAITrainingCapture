import math

import librosa.onset
import numpy as np
import scipy.signal.windows
from numpy.lib import stride_tricks

import pyaudio
from scipy import signal
from scipy.fftpack import fft


class AudioHandler(object):
    def __init__(self):
        # pyaudio
        self.threshold = None
        self.rms = None
        self.spectrogram_t = None
        self.spectrogram_f = None
        self.spectrogram_A = None
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 1
        self.WINDOW_SIZE = 512
        self.BUFFER_BLOCKS = int(math.ceil(self.RATE / 2 / self.CHUNK))
        self.p = None
        self.stream = None

        # Create a buffer for storing 500ms worth of signal
        self.data_buffer = np.zeros(self.BUFFER_BLOCKS * self.CHUNK)

        self.ring_buffer_index = 0
        self.ring_buffer_full = False

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

        if self.ring_buffer_full:
            self.data_buffer = np.roll(self.data_buffer, -frame_count)
            self.data_buffer[-frame_count::] = wf_data
            self.ring_buffer_index += 1
            if self.ring_buffer_index == self.BUFFER_BLOCKS:
                print(f'Ring buffer filled second time.')
                noise_median = np.percentile(self.rms[::], 50)
                sigma = np.percentile(self.rms[::], 84.1) - noise_median
                # Set the minimum RMS energy threshold that is needed in order to declare
                # an "onset" event to be equal to 5 sigma above the median
                self.threshold = noise_median + 5 * sigma
                print(f'RMS Threshold: {self.threshold}')
        else:
            index = self.ring_buffer_index * frame_count
            self.data_buffer[index:index + frame_count] = wf_data
            self.ring_buffer_index += 1
            if self.ring_buffer_index == self.BUFFER_BLOCKS:
                # Buffer has filled, mark it full and collect a baseline of noise
                self.ring_buffer_full = True
                self.ring_buffer_index = 0
                print(f'Ring buffer filled')

        onsets = librosa.onset.onset_detect(y=self.data_buffer, sr=self.RATE, backtrack=False)
        onstm = librosa.frames_to_time(onsets, sr=self.RATE)

        print(f'Onset Times: {onstm}')

        # Calculate RMS energy per frame. I shortened the frame length from the
        # default value in order to avoid ending up with too much smoothing
        self.rms = librosa.feature.rms(y=self.data_buffer, frame_length=1024)[0, ]
        envtm = librosa.frames_to_time(np.arange(len(self.rms)), sr=self.RATE)
        print(f'Frame Times: {envtm}')

        if self.threshold is not None:
            thresh_idx = [self.rms > self.threshold]
            print(f'Event Indices: {thresh_idx}')
            for tm in onstm:
                print(f'Found: {tm in envtm[thresh_idx]}')
            indices = [tm in envtm[thresh_idx] for tm in onstm]
            print(f'Indices: {indices}')
            correctedonstm = onstm[indices]
            print(f'Threshold Times: {correctedonstm}')
        # self.spectrogram_f, self.spectrogram_t, self.spectrogram_A = signal.spectrogram(self.data_buffer, self.RATE,
        #                                                                                detrend=False,
        #                                                                                mode='psd')

        return None, pyaudio.paContinue

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
        return self.data_buffer

    def get_latest_rms(self):
        return self.rms

    def get_latest_spectrogram(self):
        return [self.spectrogram_t, self.spectrogram_f, self.spectrogram_A]
