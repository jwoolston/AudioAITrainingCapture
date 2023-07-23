import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QWindow
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel, QMessageBox
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg
import numpy as np

import sys

from AudioHandler import AudioHandler


class DetectionDialog(QDialog):
    def __init__(self, detection, parent=None):
        super().__init__(parent)
        self.time = detection.time
        self.freq = detection.freq
        self.power = detection.power
        self.setWindowTitle("Detection Received")

        buttons = QDialogButtonBox.Apply | QDialogButtonBox.Discard

        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QGridLayout()
        self.spectrogram = pg.PlotWidget(title='SPECTROGRAM')
        self.layout.addWidget(self.spectrogram, 0, 0)
        self.layout.addWidget(self.buttonBox, 1, 0)
        self.setLayout(self.layout)

        self.spectrogram_trace = False
        # Item for displaying image data
        self.spectrogram_img = pg.ImageItem()
        self.spectrogram.addItem(self.spectrogram_img)

        self.spectrogram.setMouseEnabled(x=False, y=False)
        self.spectrogram.hideButtons()
        self.spectrogram.addColorBar(self.spectrogram_img, values=(0, 30_000), label='Spectral Power', limits=(0, None),
                                     colorMap='inferno')

        if not self.spectrogram_trace:
            self.spectrogram_trace = True

        # Sxx contains the amplitude for each pixel
        rect = QRectF(0, 0, self.time[-1], self.freq[-1])

        self.spectrogram_img.setImage(self.power, rect=rect)

        # Limit panning/zooming to the spectrogram
        self.spectrogram.setLimits(xMin=0, xMax=self.time[-1], yMin=0, yMax=self.freq[-1])
        # Add labels to the axis
        self.spectrogram.setLabel('bottom', "Time", units='s')
        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in
        # this case kHz)
        self.spectrogram.setLabel('left', "Frequency", units='Hz')

class Gui(QtWidgets.QWidget):
    stop_audio = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create the audio thread
        self.audio_handler = AudioHandler()
        self.SIZE_X = self.audio_handler.CHUNK * self.audio_handler.BUFFER_BLOCKS
        self.FS = self.audio_handler.RATE

        # Prepare the window
        self.setWindowTitle('Capture Analyzer')
        self.resize(1440, 1024)

        # Create a grid layout to manage the widgets size and position
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

        # Plot Widgets
        self.waveform = pg.PlotWidget(title='WAVEFORM')
        self.rms_plot = pg.PlotWidget(title='RMS')

        # Control Widgets
        self.baseline_rms_capture = QtWidgets.QPushButton('Capture Noise RMS')
        self.baseline_rms_capture.clicked.connect(self.capture_rms_baseline)

        # Add the widgets to the window
        self.layout.addWidget(self.waveform, 0, 0)
        self.layout.addWidget(self.rms_plot, 1, 0)
        self.layout.addWidget(self.baseline_rms_capture, 2, 0)

        self.rms_x = None
        self.rms_trace = None
        self.waveform_trace = None

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.setConfigOptions(antialias=True)

    def draw(self):
        waveform = self.audio_handler.get_latest_waveform()
        rms = self.audio_handler.get_latest_rms()

        if waveform is not None:
            self.update_waveform(waveform)
        if rms is not None:
            self.update_rms(rms)

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
        # Setup audio handler thread
        # Make thread execute audio handler run() method when started
        # Allow the audio handler loop to exit
        self.stop_audio.connect(self.audio_handler.stop)
        # Tell the audio handler to shut down when the thread is quit
        # self.audio_thread.quit.connect(self.audio_handler.stop())
        # Receive updates
        self.audio_handler.update.connect(self.draw)
        # Receive detections
        self.audio_handler.detection.connect(self.detection_received)
        self.audio_handler.start()

        # Draw update timer, ~30FPS
        timer = QtCore.QTimer()
        timer.timeout.connect(self.draw)
        timer.start(32)
        # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        #    self.app.instance().exec()

    def stop(self):
        self.stop_audio.emit()
        self.close()

    def detection_received(self, detection):
        print(f'Detection received!')
        detection_dialog = DetectionDialog(detection, parent=self)
        button = detection_dialog.exec()
        if button == QMessageBox.Apply:
            print("Success!")
        elif button == QMessageBox.Discard:
            print("Cancel!")
        else:
            print("Unknown button response")
        self.audio_handler.resume()

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
