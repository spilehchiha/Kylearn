import tensorflow as tf


def model_vars(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def getter_ema(ema, getter, name, *args, **kwargs):
    """Exponential moving average getter for variable scopes.

    Args:
        ema: ExponentialMovingAverage object, where to get variable moving averages.
        getter: default variable scope getter.
        name: variable name.
        *args: extra args passed to default getter.
        **kwargs: extra args passed to default getter.

    Returns:
        If found the moving average variable, otherwise the default variable.
    """
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var

def get_variable_num():
    total_parameters = 0
    for variable in tf.trainable_variables():
        print(variable)
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
        print(total_parameters)


from itertools import islice, chain

import numpy as np


def evaluation_epoch_generator(data, batch_size=100):
    def generate():
        for idx in range(0, len(data), batch_size):
            yield data[idx:(idx + batch_size)]
    return generate


def training_batches(data, batch_size=100, n_labeled_per_batch=50, random=np.random):
    n_unlabeled_per_batch = batch_size - n_labeled_per_batch
    labeled_data, unlabeled_data = split_labeled(data)
    return combine_batches(
        batches(labeled_data, n_labeled_per_batch, random),
        batches(unlabeled_data, n_unlabeled_per_batch, random))



def split_labeled(data):
    is_labeled = (data['y'] != -1)
    return data[is_labeled], data[~is_labeled]

def combine_batches(*batch_generators):
    return (np.concatenate(batches) for batches in zip(*batch_generators))

def batches(data, batch_size=100, random=np.random):
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

def training_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "training_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }