import math

import librosa.onset
import numpy as np
import scipy.signal.windows
import pyaudio
from scipy.signal import butter, filtfilt
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition


class Detection(object):
    def __init__(self, times, buffer, rate, hop_len, n_fft):
        self.times = times.copy()
        self.buffer = buffer.copy()
        self.rate = rate
        self.hop_length = hop_len
        self.n_fft = n_fft


class AudioHandler(QThread):
    resume = pyqtSignal()
    update = pyqtSignal()
    detection = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        # State management
        self.do_work = True
        self.pause_flag = False
        self.pause_mutex = QMutex()
        self.pause_wait = QWaitCondition()

        # pyaudio
        self.filter_coef = None
        self.threshold = None
        self.rms = None
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 22050
        self.CHUNK = 1024 * 1
        self.WINDOW_SIZE = 512
        self.HOP_LENGTH = 512
        self.BUFFER_BLOCKS = int(math.ceil(self.RATE / self.CHUNK))
        self.p = None
        self.stream = None
        self.detection_block_counter = 0
        self.accumulating = False

        # Create a buffer for storing 500ms worth of signal
        self.data_buffer = np.zeros(self.BUFFER_BLOCKS * self.CHUNK)
        self.times = None

        self.ring_buffer_index = 0
        self.ring_buffer_full = False

        # Create a highpass filter that cuts off a 200Hz
        self.filter_coef = self.create_butter_highpass(250)

    def __del__(self):
        self.stop()
        self.wait()

    def run(self):
        self.work()

    def stop(self):
        print('Stopping audio handler')
        self.do_work = False

    def pause(self):
        self.reset_stream()
        self.pause_mutex.lock()
        self.pause_flag = True
        self.pause_wait.wait(self.pause_mutex)
        self.pause_mutex.unlock()

    def resume(self):
        print('Resuming audio handler')
        self.pause_mutex.lock()
        self.pause_flag = False
        self.pause_wait.wakeAll()
        self.pause_mutex.unlock()

    def get_stream(self):
        return self.p.open(format=self.FORMAT,
                           channels=self.CHANNELS,
                           rate=self.RATE,
                           input=True,
                           output=False,
                           frames_per_buffer=self.CHUNK)

    def reset_stream(self):
        self.stream.close()
        self.stream = None

    def work(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.get_stream()

        while self.do_work:
            wf_data = np.frombuffer(self.stream.read(self.CHUNK), dtype=np.float32)
            filtered_data = filtfilt(self.filter_coef[0], self.filter_coef[1], wf_data)

            if self.ring_buffer_full:
                self.data_buffer = np.roll(self.data_buffer, -self.CHUNK)
                self.data_buffer[-self.CHUNK::] = filtered_data
                if self.accumulating:
                    self.detection_block_counter += 1
            else:
                index = self.ring_buffer_index * self.CHUNK
                self.data_buffer[index:index + self.CHUNK] = filtered_data
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

            # Calculate RMS energy per frame. The frame length from the default value
            # in order to avoid ending up with too much smoothing
            self.rms = librosa.feature.rms(S=S, hop_length=self.HOP_LENGTH)[0,]

            self.update.emit()

            if self.threshold is not None and not self.accumulating:
                thresh_idx = np.asarray(self.rms > self.threshold).astype(int)
                # print(f'Thresh Index: {thresh_idx}')
                nz_count = np.count_nonzero(thresh_idx)
                if nz_count > 0:
                    self.accumulating = True
                    offset = np.argmax(thresh_idx != 0)
                    print(f'Offset: {offset}')
                    self.detection_block_counter = 1 #(len(thresh_idx) - offset)

            if self.accumulating and self.detection_block_counter == self.BUFFER_BLOCKS:
                print(f'Buffer accumulated')
                self.detection.emit(Detection(self.times, self.data_buffer, self.RATE, self.HOP_LENGTH, 2048))
                self.pause()
                self.zero_buffer()
                self.stream = self.get_stream()

        # Cleanup the stream
        self.reset_stream()
        self.p.terminate()

    def zero_buffer(self):
        self.data_buffer = np.zeros(self.data_buffer.shape)
        self.ring_buffer_index = 0
        self.ring_buffer_full = False
        self.accumulating = False
        self.detection_block_counter = 0

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

    def get_latest_waveform(self):
        if self.times is None:
            return None
        return self.data_buffer, np.linspace(self.times[0], self.times[-1], len(self.data_buffer))

    def get_latest_rms(self):
        if self.times is None:
            return None
        return self.rms, self.times
