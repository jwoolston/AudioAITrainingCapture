import os
from datetime import datetime

import csv

import scipy

# field names
fields = ['category', 'is_head', 'is_edge', 'src_file']


class Sample(object):
    def __init__(self, buffer, channels, sample_width, rate, score, is_head, is_edge, cartridge):
        self.buffer = buffer.copy()
        self.channels = channels
        self.sample_width = sample_width
        self.rate = rate
        self.score = score
        self.is_head = is_head
        self.is_edge = is_edge
        self.cartridge = cartridge


class SampleWriter(object):
    def __init__(self, root_directory, directory):
        self.directory = directory
        now = datetime.now()
        self.root_dir = root_directory
        # name of csv file
        csv_filename = f"hits.csv"
        os.makedirs(self.root_dir, exist_ok=True)
        self.csvfile = open(os.path.join(self.root_dir, csv_filename), 'a', newline='')
        self.csv_writer = csv.writer(self.csvfile)
        # writing the fields
        self.csv_writer.writerow(fields)

    def close(self):
        self.csvfile.flush()
        self.csvfile.close()

    def set_new_directory(self, directory):
        self.directory = directory

    def write_sample(self, sample):
        print(f'Writing sample to disk.')
        path = str(int(datetime.timestamp(datetime.now())))
        score_str = str(sample.score)
        if sample.is_head:
            score_str += 'H'
        working_dir = os.path.join(self.root_dir, self.directory, score_str)
        os.makedirs(working_dir, exist_ok=True)
        sample_filename = os.path.join(working_dir, f'{path}.wav')

        scipy.io.wavfile.write(sample_filename, sample.rate, sample.buffer)

        csv_row = [score_str, str(sample.is_head), str(sample.is_edge), sample_filename]
        self.csv_writer.writerow(csv_row)
