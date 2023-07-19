import numpy as np
from numpy.lib import stride_tricks

import pyaudio
from scipy import signal
from scipy.fftpack import rfft


class AudioHandler(object):
    def __init__(self):
        # pyaudio
        self.spectrogram_t = None
        self.spectrogram_f = None
        self.spectrogram_A = None
        self.wf_data = None
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024 * 1
        self.p = None
        self.stream = None
        self.traces = dict()
        self.reference_db = 1  # 32768

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
        self.wf_data = np.frombuffer(in_data, dtype=np.int16)
        self.spectrogram_f, self.spectrogram_t, self.spectrogram_A = signal.spectrogram(self.wf_data, self.RATE,
                                                                                        detrend=False,
                                                                                        mode='psd')

        return None, pyaudio.paContinue

    def get_latest_waveform(self):
        return self.wf_data

    def get_latest_spectrogram(self):
        return [self.spectrogram_t, self.spectrogram_f, self.spectrogram_A]
