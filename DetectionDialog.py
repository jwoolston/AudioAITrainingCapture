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

    def set_label_width(self, width):
        self.label.setFixedWidth(width)

    def set_input_width(self, width):
        self.lineEdit.setFixedWidth(width)

    def get_value(self):
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

    def get_value(self):
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
        self.times = np.linspace(detection.times[0], detection.times[-1], np.shape(self.data)[0])
        self.freq = librosa.fft_frequencies(sr=self.rate, n_fft=self.n_fft)

        self.left_S = np.abs(librosa.stft(y=self.data[:, 0], hop_length=self.hop_length, n_fft=self.n_fft, center=True,
                                          window=scipy.signal.windows.blackman))
        self.left_power = librosa.feature.melspectrogram(S=self.left_S, sr=self.rate, hop_length=self.hop_length,
                                                         n_fft=self.n_fft, center=True)
        self.right_S = np.abs(librosa.stft(y=self.data[:, 1], hop_length=self.hop_length, n_fft=self.n_fft, center=True,
                                          window=scipy.signal.windows.blackman))
        self.right_power = librosa.feature.melspectrogram(S=self.right_S, sr=self.rate, hop_length=self.hop_length,
                                                         n_fft=self.n_fft, center=True)
        self.setWindowTitle("Detection Received")

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(buttons)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QtWidgets.QGridLayout()
        self.waveform = pg.PlotWidget(title='WAVEFORM')
        self.left_spectrogram = pg.PlotWidget(title='LEFT SPECTROGRAM')
        self.right_spectrogram = pg.PlotWidget(title='RIGHT SPECTROGRAM')
        self.layout.addWidget(self.waveform, 0, 0)
        self.layout.addWidget(self.left_spectrogram, 1, 0)
        self.layout.addWidget(self.right_spectrogram, 1, 1)
        self.add_int_inputs_panel(self.layout, 2, 0)
        self.layout.addWidget(self.buttonBox, 3, 0)
        self.setLayout(self.layout)

        # Waveform plot

        self.left_waveform_trace = self.waveform.plot(pen='w', width=3)
        self.right_waveform_trace = self.waveform.plot(pen='r', width=3)
        self.waveform.setLabel('left', "Amplitude", units='V')
        self.waveform.setLabel('bottom', "Time", units='s')
        self.left_waveform_trace.setData(self.times, self.data[:, 0])
        self.right_waveform_trace.setData(self.times, self.data[:, 1])

        # Item for displaying image data
        self.left_spectrogram_img = pg.ImageItem()
        self.left_spectrogram.addItem(self.left_spectrogram_img)
        self.right_spectrogram_img = pg.ImageItem()
        self.right_spectrogram.addItem(self.right_spectrogram_img)

        self.left_spectrogram.setMouseEnabled(x=False, y=False)
        self.left_spectrogram.hideButtons()
        self.left_spectrogram_img.setImage(self.left_power)
        self.right_spectrogram.setMouseEnabled(x=False, y=False)
        self.right_spectrogram.hideButtons()
        self.right_spectrogram_img.setImage(self.right_power)

        color_map = pg.colormap.get("inferno")
        # generate an adjustable color bar, initially spanning min to max power:
        # Use the same power range for both
        _min = min(np.min(self.left_power), np.min(self.right_power))
        _max = max(np.max(self.left_power), np.max(self.right_power))
        left_bar = pg.ColorBarItem(values=(_min, _max), colorMap=color_map)
        right_bar = pg.ColorBarItem(values=(_min, _max), colorMap=color_map)

        # Sxx contains the amplitude for each pixel
        rect = QRectF(self.times[0], self.freq[0], self.times[-1], self.freq[-1])

        self.left_spectrogram_img.setImage(self.left_power, rect=rect)
        self.right_spectrogram_img.setImage(self.right_power, rect=rect)

        # link color bar and color map to spectrogram, and show it in plotItem:
        left_bar.setImageItem(self.left_spectrogram_img, insert_in=self.left_spectrogram.plotItem)
        right_bar.setImageItem(self.right_spectrogram_img, insert_in=self.right_spectrogram.plotItem)

        # Limit panning/zooming to the spectrogram
        self.left_spectrogram.setLimits(xMin=self.times[0], xMax=self.times[-1], yMin=self.freq[0], yMax=self.freq[-1])
        self.right_spectrogram.setLimits(xMin=self.times[0], xMax=self.times[-1], yMin=self.freq[0], yMax=self.freq[-1])
        # Add labels to the axis
        self.left_spectrogram.setLabel('bottom', "Time", units='s')
        self.right_spectrogram.setLabel('bottom', "Time", units='s')
        # If you include the units, Pyqtgraph automatically scales the axis and adjusts the SI prefix (in
        # this case kHz)
        self.left_spectrogram.setLabel('left', "Frequency", units='Hz')
        self.right_spectrogram.setLabel('left', "Frequency", units='Hz')

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
