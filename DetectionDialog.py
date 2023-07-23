import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRectF
from PyQt5.QtWidgets import QDialog, QDialogButtonBox


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
