from itertools import islice, chain
import numpy as np


def evaluation_generator(data, batch_size=100):
    print(len(data))
    def generate():
        for idx in range(0, len(data), batch_size):
            print(idx)
            yield data[idx:(idx + batch_size)]
    return generate

def training_generator(data, batch_size=100, random=np.random):
    assert batch_size > 0 and len(data) > 0
    for batch_idxs in random_index(len(data), batch_size, random):
        yield data[batch_idxs]



def random_index(max_index, batch_size, random=np.random):
    def random_ranges():
        while True:
            indices = np.arange(max_index)
            random.shuffle(indices)
            yield indices

    def batch_slices(iterable):
        while True:
            yield np.array(list(islice(iterable, batch_size)))

    eternal_random_indices = chain.from_iterable(random_ranges())
    return batch_slices(eternal_random_indices)
