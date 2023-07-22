import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRectF
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np

import sys


class Gui(object):
    def __init__(self, audio_handler):
        self.audio_handler = audio_handler
        self.SIZE_X = self.audio_handler.CHUNK * self.audio_handler.BUFFER_BLOCKS
        self.FS = self.audio_handler.RATE

        # Start by initializing Qt (only once per application)
        self.app = pg.mkQApp("Hit Detect Training Capture")

        # Define a top-level widget to hold everything
        self.win = QtWidgets.QWidget()
        self.win.setWindowTitle('Capture Analyzer')

        # Create a grid layout to manage the widgets size and position
        self.layout = QtWidgets.QGridLayout()
        self.win.setLayout(self.layout)
        self.win.resize(1440, 1024)

        # Plot Widgets
        self.waveform = pg.PlotWidget(title='WAVEFORM')
        self.rms_plot = pg.PlotWidget(title='RMS')
        self.spectrogram = pg.PlotWidget(title='SPECTROGRAM')

        # Control Widgets
        self.baseline_rms_capture = QtWidgets.QPushButton('Capture Noise RMS')
        self.baseline_rms_capture.clicked.connect(self.capture_rms_baseline)

        self.layout.addWidget(self.waveform, 0, 0)
        self.layout.addWidget(self.rms_plot, 1, 0)
        self.layout.addWidget(self.baseline_rms_capture, 2, 0)
        self.layout.addWidget(self.spectrogram, 3, 0)

        self.rms_x = None
        self.rms_trace = None
        self.waveform_trace = None
        self.spectrogram_trace = False

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.setConfigOptions(antialias=True)
        # Item for displaying image data
        self.spectrogram_img = pg.ImageItem()
        self.spectrogram.addItem(self.spectrogram_img)

        self.spectrogram.setMouseEnabled(x=False, y=False)
        self.spectrogram.hideButtons()
        self.spectrogram.addColorBar(self.spectrogram_img, values=(0, 30_000), label='Spectral Power', limits=(0, None),
                                     colorMap='inferno')

        self.win.show()

    def draw(self):
        waveform = self.audio_handler.get_latest_waveform()
        rms = self.audio_handler.get_latest_rms()
        spectrogram = self.audio_handler.get_latest_spectrogram()
        if waveform is not None:
            self.update_waveform(waveform)
        if rms is not None:
            self.update_rms(rms)
        if spectrogram is not None:
            self.update_spectrogram(spectrogram)

    def capture_rms_baseline(self):
        print("Capturing RMS baseline.")
        self.audio_handler.capture_rms_baseline()

    def update_waveform(self, waveform):
        self.set_waveform_data(waveform[1], waveform[0])

    def update_rms(self, rms):
        self.set_rms_data(rms[1], rms[0])

    def update_spectrogram(self, spectrogram):
        self.set_spectrogram_data(spectrogram[0], spectrogram[1], spectrogram[2])

    def run(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.draw)
        timer.start(32)
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.instance().exec()

    def stop(self):
        self.win.close()

    def set_waveform_data(self, x, y):
        if self.waveform_trace is None:
            self.waveform_trace = self.waveform.plot(pen='c', width=3)
            self.waveform.setXRange(x[0], x[-1], padding=0.005)
            self.waveform.setYRange(-1, 1, padding=0)
            self.waveform.setLabel('left', "Amplitude", units='V')
            self.waveform.setLabel('bottom', "Time", units='s')
        self.waveform_trace.setData(x, y)

    def set_rms_data(self, x, y):
        if self.rms_trace is None:
            self.rms_trace = self.rms_plot.plot(pen='r', width=3)
            self.rms_plot.setXRange(x[0], x[-1])
            self.rms_plot.setYRange(0, 0.1)
            self.rms_plot.setLogMode(x=False, y=True)
            self.rms_plot.setLabel('left', "RMSe")
            self.rms_plot.setLabel('bottom', "Time", units='s')
        self.rms_trace.setData(x, y)

    def set_spectrogram_data(self, time, freq, amplitude):
        if amplitude is None:
            # There is no data for us to update
            return
        if not self.spectrogram_trace:
            self.spectrogram_trace = True

        # Sxx contains the amplitude for each pixel
        rect = QRectF(0, 0, time[-1], freq[-1])

        self.spectrogram_img.setImage(amplitude, rect=rect)

        # Limit panning/zooming to the spectrogram
        self.spectrogram.setLimits(xMin=0, xMax=time[-1], yMin=0, yMax=freq[-1])
        # Add labels to the axis
        self.spectrogram.setLabel('bottom', "Time", units='s')
        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in
        # this case kHz)
        self.spectrogram.setLabel('left', "Frequency", units='Hz')
