import librosa
import numpy as np
import pyqtgraph as pg
import scipy
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QFont, QIntValidator
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QWidget, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout, QCheckBox
from scipy import signal


# A simple widget consisting of a QLabel and a QLineEdit that
# uses a QIntValidator to ensure that only integer inputs are
# accepted. This class could be implemented in a separate
# script called, say, labelled_int_field.py
class LabelledIntField(QWidget):
    def __init__(self, title, initial_value=None):
        QWidget.__init__(self)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel()
        self.label.setText(title)
        self.label.setFixedWidth(100)
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        layout.addWidget(self.label)

        self.lineEdit = QLineEdit(self)
        self.lineEdit.setFixedWidth(40)
        self.lineEdit.setValidator(QIntValidator())
        if initial_value is not None:
            self.lineEdit.setText(str(initial_value))
        layout.addWidget(self.lineEdit)
        layout.addStretch()

    def setLabelWidth(self, width):
        self.label.setFixedWidth(width)

    def setInputWidth(self, width):
        self.lineEdit.setFixedWidth(width)

    def getValue(self):
        return int(self.lineEdit.text())


# A simple widget consisting of a QLabel and a QLineEdit that
# uses a QIntValidator to ensure that only integer inputs are
# accepted. This class could be implemented in a separate
# script called, say, labelled_int_field.py
class LabelledCheckboxField(QWidget):
    def __init__(self, title, initial_value=None):
        QWidget.__init__(self)
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel()
        self.label.setText(title)
        self.label.setFixedWidth(100)
        self.label.setFont(QFont("Arial", weight=QFont.Bold))
        layout.addWidget(self.label)

        self.checkbox = QCheckBox(self)
        if initial_value is not None:
            self.checkbox.setChecked(initial_value)
        layout.addWidget(self.checkbox)
        layout.addStretch()

    def getValue(self):
        return self.checkbox.isChecked()


class DetectionDialog(QDialog):
    def __init__(self, detection, parent=None):
        super().__init__(parent)
        self.on_edge = None
        self.is_head = None
        self.score = None
        self.data = detection.buffer
        self.rate = detection.rate
        self.hop_length = detection.hop_length
        self.n_fft = detection.n_fft
        self.times = np.linspace(detection.times[0], detection.times[-1], len(self.data))
        self.freq = librosa.fft_frequencies(sr=self.rate, n_fft=self.n_fft)

        self.S = np.abs(librosa.stft(y=self.data, hop_length=self.hop_length, n_fft=self.n_fft, center=True,
                                     window=scipy.signal.windows.blackman))
        self.power = librosa.feature.melspectrogram(S=self.S, sr=self.rate, hop_length=self.hop_length,
                                                    n_fft=self.n_fft, center=True)
        self.setWindowTitle("Detection Received")

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QGridLayout()
        self.waveform = pg.PlotWidget(title='WAVEFORM')
        self.spectrogram = pg.PlotWidget(title='SPECTROGRAM')
        self.layout.addWidget(self.waveform, 0, 0)
        self.layout.addWidget(self.spectrogram, 1, 0)
        self.add_int_inputs_panel(self.layout, 2, 0)
        self.layout.addWidget(self.buttonBox, 3, 0)
        self.setLayout(self.layout)

        # Waveform plot

        self.waveform_trace = self.waveform.plot(pen='c', width=3)
        # self.waveform.setXRange(self.times[0], self.times[-1], padding=0.005)
        # self.waveform.setYRange(-1, 1, padding=0)
        self.waveform.setLabel('left', "Amplitude", units='V')
        self.waveform.setLabel('bottom', "Time", units='s')
        self.waveform_trace.setData(self.times, self.data)

        # Item for displaying image data
        self.spectrogram_img = pg.ImageItem()
        self.spectrogram.addItem(self.spectrogram_img)

        self.spectrogram.setMouseEnabled(x=False, y=False)
        self.spectrogram.hideButtons()
        self.spectrogram_img.setImage(self.power)

        color_map = pg.colormap.get("inferno")
        # generate an adjustable color bar, initially spanning min to max power:
        bar = pg.ColorBarItem(values=(np.min(self.power), np.max(self.power)), colorMap=color_map)

        # Sxx contains the amplitude for each pixel
        rect = QRectF(self.times[0], self.freq[0], self.times[-1], self.freq[-1])

        self.spectrogram_img.setImage(self.power, rect=rect)

        # link color bar and color map to spectrogram, and show it in plotItem:
        bar.setImageItem(self.spectrogram_img, insert_in=self.spectrogram.plotItem)

        # Limit panning/zooming to the spectrogram
        self.spectrogram.setLimits(xMin=self.times[0], xMax=self.times[-1], yMin=self.freq[0], yMax=self.freq[-1])
        # Add labels to the axis
        self.spectrogram.setLabel('bottom', "Time", units='s')
        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in
        # this case kHz)
        self.spectrogram.setLabel('left', "Frequency", units='Hz')

        self.score.lineEdit.setFocus()

    def add_int_inputs_panel(self, parent, row, column):
        hlayout = QHBoxLayout()
        self.score = LabelledIntField('Score', 0)
        self.is_head = LabelledCheckboxField('Is Head', False)
        self.on_edge = LabelledCheckboxField('On Edge', False)

        hlayout.addWidget(self.score)
        hlayout.addWidget(self.is_head)
        hlayout.addWidget(self.on_edge)
        hlayout.addStretch()
        parent.addLayout(hlayout, row, column)
