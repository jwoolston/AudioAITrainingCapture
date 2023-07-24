import scipy.io.wavfile as wave
import json


class Sample(object):
    def __init__(self, buffer, rate, score, is_head, is_edge, cartridge):
        self.buffer = buffer.copy()
        self.rate = rate
        self.score = score
        self.is_head = is_head
        self.is_edge = is_edge
        self.cartridge = cartridge


def write_sample(sample):
    print(f'Writing sample to disk.')
    wave.write('sample.wav', sample.rate, sample.buffer)
    meta_data_json = {
        "rate": sample.rate,
        "score": sample.score,
        "is_head": sample.is_head,
        "is_edge": sample.is_edge,
        "cartridge": sample.cartridge
    }

    # Serializing json
    json_object = json.dumps(meta_data_json, indent=4)

    # Writing to sample.json
    with open("meta_data.json", "w") as outfile:
        outfile.write(json_object)


class SampleWriter(object):
    def __init__(self):
        pass
