import PyQt5
import pyqtgraph
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRectF
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt


import sys


class Gui(object):
    def __init__(self, audio_handler):
        self.audio_handler = audio_handler
        self.SIZE_X = self.audio_handler.CHUNK
        self.FS = self.audio_handler.RATE

        self.app = pg.mkQApp("Hit Detect Training Capture")
        self.win = pg.GraphicsLayoutWidget(show=True, title='Capture Analyzer')
        self.win.setWindowTitle('Capture Analyzer')
        self.win.resize(1440, 1024)

        self.waveform = self.win.addPlot(
            title='WAVEFORM', row=1, col=1  # axisItems={'bottom': wf_xaxis},
        )
        # A plot area (ViewBox + axes) for displaying the image
        self.spectrogram = self.win.addPlot(
            title='SPECTROGRAM', row=2, col=1
        )

        self.waveform_x = np.arange(0, self.SIZE_X)
        self.waveform_trace = None
        self.spectrogram_trace = False

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        # Item for displaying image data
        self.spectrogram_img = pg.ImageItem()
        self.spectrogram.addItem(self.spectrogram_img)
        self.spectrogram_img.setColorMap(pyqtgraph.colormap.get(name="jet", source='matplotlib'))
        #self.spectrogram_hist = self.spectrogram_img.getHistogram()
        # Add a histogram with which to control the gradient of the image
        #self.spectrogram_hist = pg.HistogramLUTItem()
        # Link the histogram to the image
        #self.spectrogram_hist.setImageItem(self.spectrogram_img)

        #self.spectrogram_hist.gradient.restoreState(
        #    {'mode': 'rgb',
        #     'ticks': [(0.5, (0, 182, 188, 255)),
        #               (1.0, (246, 111, 0, 255)),
        #               (0.0, (75, 0, 113, 255))]})

        # If you don't add the histogram to the window, it stays invisible, but I find it useful.
        #self.win.addItem(self.spectrogram_hist, row=2, col=2)

    def draw(self):
        waveform = self.audio_handler.get_latest_waveform()
        spectrogram = self.audio_handler.get_latest_spectrogram()
        if waveform is not None:
            self.update_waveform(waveform)
        if spectrogram is not None:
            self.update_spectrogram(spectrogram)

    def update_waveform(self, waveform):
        self.set_waveform_data(waveform)

    def update_spectrogram(self, spectrogram):
        self.set_spectrogram_data(spectrogram[0], spectrogram[1], spectrogram[2])

    def run(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.draw)
        timer.start(20)
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.instance().exec_()

    def stop(self):
        self.win.close()

    def set_waveform_data(self, data):
        if self.waveform_trace is None:
            self.waveform_trace = self.waveform.plot(pen='c', width=3)
            self.waveform.setYRange(-32767, 32768, padding=0)
            self.waveform.setXRange(0, self.SIZE_X, padding=0.005)
        self.waveform_trace.setData(self.waveform_x, data)

    def set_spectrogram_data(self, time, freq, amplitude):
        if amplitude is None:
            # There is no data for us to update
            return
        if not self.spectrogram_trace:
            self.spectrogram_trace = True

        # Sxx contains the amplitude for each pixel
        image_width = time[-1]
        image_height = freq[-1]
        pixel_size_x = image_width / (len(time) - 1)
        pixel_size_y = image_height / (len(freq) - 1)
        rect = QRectF(0, 0, image_width, image_height)

        self.spectrogram_img.setImage(amplitude, rect=rect)#scale=[time[-1] / len(time), len(freq) / freq[-1]])
        #self.spectrogram_hist.setLevels(np.min(amplitude), np.max(amplitude))

        # Scale the X and Y Axis to time and frequency (standard is pixels)
        # self.spectrogram.setYRange(0, freq[-1])
        # Limit panning/zooming to the spectrogram
        self.spectrogram.setLimits(xMin=0, xMax=time[-1], yMin=0, yMax=freq[-1])
        # Add labels to the axis
        self.spectrogram.setLabel('bottom', "Time", units='s')
        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in
        # this case kHz)
        self.spectrogram.setLabel('left', "Frequency", units='Hz')
