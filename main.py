# This is a sample Python script.
import signal
import sys

import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication

from AudioHandler import AudioHandler
from Gui import Gui


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    gui.stop()
    sys.exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #app = QApplication(sys.argv)
    #app.setApplicationDisplayName("Hit Detect Training Capture")
    #window = Gui()
    #window.show()
    #sys.exit(app.exec_())

    # Start by initializing Qt (only once per application)
    app = pg.mkQApp("Hit Detect Training Capture")

    # Define a top-level widget to hold everything
    gui = Gui()
    gui.show()
    signal.signal(signal.SIGINT, signal_handler)
    gui.run()
    app.exec()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
