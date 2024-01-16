from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QFileDialog, QComboBox
from pyqtgraph.Qt import QtCore
import pyqtgraph as pg

from SampleWriter import SampleWriter, Sample
from AudioHandler import AudioHandler
from DetectionDialog import DetectionDialog


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
        self.cartridge_combobox = QComboBox()
        self.cartridge_combobox.addItem('380 ACP')
        self.cartridge_combobox.addItem('38 Special')
        self.cartridge_combobox.addItem('357 Magnum')
        self.cartridge_combobox.addItem('9mm')
        self.cartridge_combobox.addItem('40 S&W')
        self.cartridge_combobox.addItem('45 ACP')
        self.cartridge_combobox.addItem('44 Magnum')
        self.cartridge_combobox.addItem('300 Blackout')
        self.cartridge_combobox.currentIndexChanged.connect(self.update_writer_directory)

        self.cartridge_array = ['380_acp', '38_special', '357_magnum', '9mm', '40_sw', '45_acp', '44_magnum',
                                '300_blackout']

        self.baseline_rms_capture = QtWidgets.QPushButton('Capture Noise RMS')
        self.baseline_rms_capture.clicked.connect(self.capture_rms_baseline)

        # Add the widgets to the window
        self.layout.addWidget(self.waveform, 0, 0, 1, 4)
        self.layout.addWidget(self.rms_plot, 1, 0, 1, 4)
        self.layout.addWidget(self.cartridge_combobox, 2, 2, 1, 1)
        self.layout.addWidget(self.baseline_rms_capture, 2, 3, 1, 1)

        self.rms_trace = None
        self.left_waveform_trace = None
        self.right_waveform_trace = None
        self.processing_detection = False

        # Interpret image data as row-major instead of col-major
        pg.setConfigOptions(imageAxisOrder='row-major')
        pg.setConfigOptions(antialias=True)

        self.sample_writer = SampleWriter(self.cartridge_array[self.cartridge_combobox.currentIndex()])

    def draw(self):
        if not self.processing_detection:
            waveform = self.audio_handler.get_latest_waveform()
            rms = self.audio_handler.get_latest_rms()

            if waveform is not None:
                self.update_waveform(waveform)
            if rms is not None:
                self.update_rms(rms)

    def capture_rms_baseline(self):
        print("Capturing RMS baseline.")
        self.audio_handler.capture_rms_baseline()

    def update_writer_directory(self, index):
        self.sample_writer.set_new_directory(self.cartridge_array[index])

    def update_waveform(self, waveform):
        self.set_waveform_data(waveform[1], waveform[0])

    def update_rms(self, rms):
        self.set_rms_data(rms[1], rms[0])

    def run(self):
        file_dialog = QFileDialog(parent=self)
        file_dialog.setFileMode(QFileDialog.Directory)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        # file_dialog.exec()
        # directory = file_dialog.getSaveFileName()
        # print(f'Save directory: {directory}')
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

    def stop(self):
        self.stop_audio.emit()
        self.sample_writer.close()
        self.close()

    def detection_received(self, detection):
        print(f'Detection received!')
        detection_dialog = DetectionDialog(detection, parent=self)
        self.processing_detection = True
        button = detection_dialog.exec()
        print(f'Dialog Button: {button}')
        if button == QDialog.DialogCode.Accepted:
            score = detection_dialog.score.get_value()
            is_head = detection_dialog.is_head.get_value()
            is_edge = detection_dialog.on_edge.get_value()
            cartridge = self.cartridge_combobox.currentText()
            print(f'Success! Score: {score} Is Head: {is_head} Is Edge: {is_edge} Cartridge: {cartridge}')
            sample = Sample(detection.buffer, detection.channels, detection.sample_width, detection.rate, score, is_head, is_edge, cartridge)
            self.sample_writer.write_sample(sample)
        elif button == QDialog.DialogCode.Rejected:
            print("Cancel!")
        else:
            print("Unknown button response")
        self.audio_handler.resume()
        self.processing_detection = False

    def set_waveform_data(self, x, y):
        if self.left_waveform_trace is None:
            self.left_waveform_trace = self.waveform.plot(pen='w', width=3)
            self.waveform.setXRange(x[0], x[-1], padding=0.005)
            self.waveform.setYRange(-1, 1, padding=0)
            self.waveform.setLabel('left', "Amplitude", units='V')
            self.waveform.setLabel('bottom', "Time", units='s')
        if self.right_waveform_trace is None:
            self.right_waveform_trace = self.waveform.plot(pen='r', width=3)
        self.left_waveform_trace.setData(x, y[:, 0])
        self.right_waveform_trace.setData(x, y[:, 1])

    def set_rms_data(self, x, y):
        if self.rms_trace is None:
            self.rms_trace = self.rms_plot.plot(pen='r', width=3)
            self.rms_plot.setXRange(x[0], x[-1])
            self.rms_plot.setYRange(0, 0.1)
            self.rms_plot.setLogMode(x=False, y=True)
            self.rms_plot.setLabel('left', "RMSe")
            self.rms_plot.setLabel('bottom', "Time", units='s')
        self.rms_trace.setData(x, y)
