import os
from datetime import datetime

import numpy as np
import wave
import csv
import json

import pyaudio

# field names
fields = ['category', 'src_file']


class Sample(object):
    def __init__(self, buffer, channels, rate, score, is_head, is_edge, cartridge):
        self.buffer = buffer.copy()
        self.channels = channels
        self.rate = rate
        self.score = score
        self.is_head = is_head
        self.is_edge = is_edge
        self.cartridge = cartridge


class SampleWriter(object):
    def __init__(self, directory):
        self.directory = directory
        now = datetime.now()
        self.root_dir = f'{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}'
        # name of csv file
        csv_filename = f"hits_{datetime.timestamp(now)}.csv"
        self.csvfile = open(csv_filename, 'w')
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

        c = np.empty((np.shape(sample.buffer)[0] * np.shape(sample.buffer)[1]), dtype=sample.buffer.dtype)
        c[0::2] = sample.buffer[:, 0]
        c[1::2] = sample.buffer[:, 1]

        with wave.open(sample_filename, 'wb') as wf:
            wf.setnchannels(sample.channels)
            wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paFloat32))
            wf.setframerate(sample.rate)
            wf.writeframes(c)

        csv_row = [score_str, sample_filename]
        self.csv_writer.writerow(csv_row)

        meta_data_json = {
            "rate": sample.rate,
            "score": sample.score,
            "is_head": sample.is_head,
            "is_edge": sample.is_edge,
            "cartridge": sample.cartridge
        }

        # Serializing json
        sample_filename = os.path.join(working_dir, f'{path}.json')
        with open(sample_filename, 'w') as metadata:
            json_object = json.dumps(meta_data_json, indent=4)
            metadata.write(json_object)
