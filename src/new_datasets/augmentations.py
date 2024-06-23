import numpy as np
from scipy.io import wavfile
from scipy.signal import resample


def stretch_audio(data, time_stretch_factors):

    segments = np.array_split(data, len(time_stretch_factors))

    stretched_audio_segments = []
    for segment, stretch_factor in zip(segments, time_stretch_factors):
        new_length = int(segment.shape[0] * stretch_factor)
        stretched_data = resample(segment, new_length)
        stretched_data = np.round(stretched_data).astype(np.int16)

        stretched_audio_segments.append(stretched_data)

    stretched_audio = np.concatenate(stretched_audio_segments)
    return stretched_audio


def shift_audio(data, shift_factor):
    # shift audio by a factor to right
    shift = int(shift_factor * len(data))
    shifted_audio = np.roll(data, shift)
    return shifted_audio


def stacker(feats, stack_order):
    """
    Concatenating consecutive audio frames
    Args:
    feats - numpy.ndarray of shape [T, F]
    stack_order - int (number of neighboring frames to concatenate
    Returns:
    feats - numpy.ndarray of shape [T', F']
    """
    feat_dim = feats.shape[1]
    if len(feats) % stack_order != 0:
        res = stack_order - len(feats) % stack_order
        res = np.zeros([res, feat_dim]).astype(feats.dtype)
        feats = np.concatenate([feats, res], axis=0)
    feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order * feat_dim)
    return feats
