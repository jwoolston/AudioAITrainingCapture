import os
from datetime import datetime

import aifc
import json
import music_tag


class Sample(object):
    def __init__(self, buffer, rate, score, is_head, is_edge, cartridge):
        self.buffer = buffer.copy()
        self.rate = rate
        self.score = score
        self.is_head = is_head
        self.is_edge = is_edge
        self.cartridge = cartridge


class SampleWriter(object):
    def __init__(self, directory):
        self.directory = directory

    def set_new_directory(self, directory):
        self.directory = directory

    def write_sample(self, sample):
        print(f'Writing sample to disk.')
        path = str(int(datetime.timestamp(datetime.now())))
        os.makedirs(self.directory, exist_ok=True)
        sample_filename = os.path.join(self.directory, f'{path}.aiff')

        with aifc.open(sample_filename, 'wb') as out:
            out.setnchannels(1)
            out.setsampwidth(4)
            out.setframerate(sample.rate)
            out.setcomptype(b'NONE', b'NONE')
            out.writeframes(sample.buffer)

        meta_data_json = {
            "rate": sample.rate,
            "score": sample.score,
            "is_head": sample.is_head,
            "is_edge": sample.is_edge,
            "cartridge": sample.cartridge
        }

        # Serializing json
        json_object = json.dumps(meta_data_json, indent=4)

        file = music_tag.load_file(sample_filename)
        if file is not None:
            file["comment"] = json_object
            file.save()
