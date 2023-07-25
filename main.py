import signal
import sys

import pyqtgraph as pg

from Gui import Gui


def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    gui.stop()
    sys.exit(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Start by initializing Qt (only once per application)
    app = pg.mkQApp("Hit Detect Training Capture")

    # Define a top-level widget to hold everything
    signal.signal(signal.SIGINT, signal_handler)
    gui = Gui()
    gui.show()
    gui.run()
    app.exec()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
