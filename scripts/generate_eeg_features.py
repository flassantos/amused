import argparse
import math
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
from biosppy.signals import tools as st
from biosppy.signals.bvp import bvp
from biosppy.signals.eeg import get_power_features
from tqdm import tqdm


def get_eeg_features(signal=None, sampling_rate=1000.0, size=0.25, overlap=0.5):
    """Process raw EEG signals and extract relevant signal features using
    default parameters.


    Parameters
    ----------
    signal : array
        Raw EEG signal matrix; each column is one EEG channel.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : float, optional
        Window size (seconds).
    overlap : float, optional
        Window overlap (0 to 1).

    Returns
    -------
    features_ts : array
        Features time axis reference (seconds).
    theta : array
        Average power in the 4 to 8 Hz frequency band; each column is one EEG
        channel.
    alpha_low : array
        Average power in the 8 to 10 Hz frequency band; each column is one EEG
        channel.
    alpha_high : array
        Average power in the 10 to 13 Hz frequency band; each column is one EEG
        channel.
    beta : array
        Average power in the 13 to 25 Hz frequency band; each column is one EEG
        channel.
    gamma : array
        Average power in the 25 to 40 Hz frequency band; each column is one EEG
        channel.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)
    signal = np.reshape(signal, (signal.shape[0], -1))

    sampling_rate = float(sampling_rate)
    nch = signal.shape[1]


    # high pass filter
    b, a = st.get_filter(
        ftype="butter",
        band="highpass",
        order=8,
        frequency=4,
        sampling_rate=sampling_rate,
    )

    aux, _ = st._filter_signal(b, a, signal=signal, check_phase=True, axis=0)

    # low pass filter
    b, a = st.get_filter(
        ftype="butter",
        band="lowpass",
        order=16,
        frequency=40,
        sampling_rate=sampling_rate,
    )

    filtered, _ = st._filter_signal(b, a, signal=aux, check_phase=True, axis=0)

    # band power features
    out = get_power_features(
        signal=filtered, sampling_rate=sampling_rate, 
        size=size, overlap=overlap
    )
    ts_feat = out["ts"]
    theta = out["theta"]
    alpha_low = out["alpha_low"]
    alpha_high = out["alpha_high"]
    beta = out["beta"]
    gamma = out["gamma"]

    return ts_feat, theta, alpha_low, alpha_high, beta, gamma


def normalize_a_b(x, a=0, b=1):
    return a + (x - x.min()) * (a - b) / (x.min() - x.max())


def read_features_and_eeg(basepath, normalize=True):
    # bitalino mac address
    bitalino_mac = '98:D3:91:FD:7A:28'

    # read everything from bitalino
    cols = ["nSeq", "I1", "I2", "O1", "O2", "A1", "A2", "A3", "A4", "A5", "A6"]
    df_eeg = pd.read_csv(basepath + '/bit.txt', sep='\t', index_col=False, names=cols, skiprows=3)

    # select only A1 and A2
    labels = ["A1", "A2"]
    df_eeg = df_eeg[labels]

    # normalize values between 0 and 1
    if normalize:
        for l in labels:
            df_eeg[l] = normalize_a_b(df_eeg[l], a=0, b=1)

    # read only the first row of the bit.txt file to get the metadata
    with open(basepath + '/bit.txt', 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i == 1:
                metadata_eeg = eval(line[2:].strip())
                metadata_eeg = metadata_eeg[bitalino_mac]
                break

    # read the features.csv table representing the log
    df_f = pd.read_csv(basepath + '/dataset.csv', sep='\t', index_col=0)

    return df_f, df_eeg, metadata_eeg


def compute_valence_and_arousal(signal, alpha_low, alpha_high):
    # compute valence and arousal according to Sandra's paper
    valence = 0.5 * alpha_low[:, 0].mean() + 0.5 * alpha_high[:, 0].mean()
    arousal = signal[:, 1].mean()
    return valence, arousal, math.degrees(math.atan(arousal / valence))


def crop_eeg(np_singals, bitalino_start_time, video_start_time='00:00:00', diff_log=0):
    # np_singals
    # array of shape (N, 2)

    # bitalino_start_time:
    # horario que comeca o bitalino

    # video_start_time:
    # horario que o computador esta marcando quando comeca o video

    # diff_log:
    # esse eh a diferenca de tempo entre o comeco do video (gravacao da tela) e o comeco do log (gravacao do wildfire)
    # se o valor for positivo, siginfica que o wildfire comecou ANTES que o video
    # se o valor for negativo, significa que comecou DEPOIS

    # conver to datetime objects
    v_format =  '%H:%M:%S.%f' if '.' in video_start_time else '%H:%M:%S'
    tv = datetime.strptime(video_start_time, v_format)
    tb = datetime.strptime(bitalino_start_time, '%H:%M:%S.%f')

    # add the delta to video_start_time to get log_start_time
    if diff_log < 0:
        log_start_time = tv + timedelta(seconds=abs(diff_log))
    else:
        log_start_time = tv - timedelta(seconds=abs(diff_log))

    # convert log_start_time back to a string if needed
    log_start_time = log_start_time.strftime('%H:%M:%S')
    tl = datetime.strptime(log_start_time, '%H:%M:%S')

    # print stuff
    print('Video start time:', video_start_time)
    print('Log start time:', log_start_time)
    print('Bit start time:', bitalino_start_time)

    # CASO 1 (B -> L -> V) or (B -> V -> L) or (V -> B -> L)
    # cut     |~~~|            |~~~~~~~~~|           |~~~|
    if tb <= tl:

        delta_lb = (tl - tb).total_seconds()
        print('CASO 1:', delta_lb)

        new_np_signals = np_singals[int(delta_lb * 1000):]

    # CASO 2 (L -> B -> V) or (L -> V -> B) or (V -> L -> B)
    else:
        # add trash
        delta_bl = (tb - tl).total_seconds()
        print('CASO 2:', delta_bl)

        L = int(delta_bl * 1000)
        z = np.zeros((L, 2))

        new_np_signals = np.concatenate([z, np_singals], axis=0)

    return pd.DataFrame(new_np_signals, columns=['A1', 'A2'])


def hh_mm_ss_to_seconds(s):
    # convert hh:mm:ss.ms to seconds
    p = s.split(':')
    return float(p[0]) * 3600 + float(p[1]) * 60 + float(p[2])


def create_suff_stats(d, suffix=None):
    d_new = {}
    s = f'_{suffix}' if suffix is not None else ''
    for k, v in d.items():
        d_new[f'{k}_min{s}'] = np.min(v)
        d_new[f'{k}_max{s}'] = np.max(v)
        d_new[f'{k}_avg{s}'] = np.mean(v)
        d_new[f'{k}_std{s}'] = np.std(v)
    return d_new


def get_eeg_features_for_interval(signals, a, b, suffix=None, fallback=None, return_trimmed_signal=False):
    lim_a = int(a * 1000)
    lim_b = int(b * 1000)
    if lim_a == lim_b:
        # print('empty interval')
        if return_trimmed_signal:
            return fallback, np.array([])
        return fallback

    signal = signals[lim_a:lim_b]
    a1_signal = signal[:, 0]
    size = min(0.25, len(a1_signal) / 1000)

    try:
        features_ts, theta, alpha_low, alpha_high, beta, gamma = get_eeg_features(
            signal=a1_signal,
            sampling_rate=1000.0,
            size=size,
            overlap=0.5
        )
    except:
        # print('error eeg')
        if return_trimmed_signal:
            return fallback, a1_signal
        return fallback

    d = {
        'bit_eeg_signal': a1_signal,
        'bit_eeg_features_ts': features_ts,
        'bit_eeg_theta': theta,
        'bit_eeg_alpha_low': alpha_low,
        'bit_eeg_alpha_high': alpha_high,
        'bit_eeg_beta': beta,
        'bit_eeg_gamma': gamma,
    }
    d_feat = create_suff_stats(d, suffix=suffix)
    d_feat[f'bit_eeg_lim_a_{suffix}'] = lim_a
    d_feat[f'bit_eeg_lim_b_{suffix}'] = lim_b

    if return_trimmed_signal:
        return d_feat, a1_signal
    return d_feat


def get_bvp_features_for_interval(signals, a, b, suffix=None, fallback=None, return_trimmed_signal=False):
    lim_a = int(a * 1000)
    lim_b = int(b * 1000)
    if lim_a == lim_b:
        # print('empty interval')
        if return_trimmed_signal:
            return fallback, np.array([])
        return fallback

    signal = signals[lim_a:lim_b]
    a2_signal = signal[:, 1]
    size = min(0.25, len(a2_signal) / 1000)

    try:
        _, _, onsets, _, hr = bvp(
            signal=a2_signal,
            sampling_rate=1000.0,
            show=False
        )
    except:
        # print('error bvp')
        if return_trimmed_signal:
            return fallback, a2_signal
        return fallback

    d = {
        'bit_bvp_signal': a2_signal,
        'bit_bvp_onsets': onsets,
        'bit_bvp_hr': hr,
    }
    d_feat = create_suff_stats(d, suffix=suffix)
    d_feat[f'bit_bvp_lim_a_{suffix}'] = lim_a
    d_feat[f'bit_bvp_lim_b_{suffix}'] = lim_b

    if return_trimmed_signal:
        return d_feat, a2_signal
    return d_feat


def save_signals_to_h5(trimmed_signals, filename):
    print('Saving trimmed signals to:', filename)
    with h5py.File(filename, 'w') as file:
        for i, item in enumerate(trimmed_signals):
            group = file.create_group(f'item_{i}')
            group.attrs['interval'] = item['interval']
            group.attrs['level'] = item['level']
            group.create_dataset('eeg_signal', data=np.array(item['eeg_signal']))
            group.create_dataset('bvp_signal', data=np.array(item['bvp_signal']))


def load_h5_signals(filename):
    print('Loading trimmed signals from:', filename)
    loaded_data = []
    with h5py.File(filename, 'r') as file:
        for key in file.keys():
            group = file[key]
            item = {
                'interval': group.attrs['interval'],
                'level': group.attrs['level'],
                'eeg_signal': np.array(group['eeg_signal']),
                'bvp_signal': np.array(group['bvp_signal'])
            }
            loaded_data.append(item)
    return loaded_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Dir path")
    parser.add_argument("--video-start-time", type=str, default=None, help="timestamp start of video")
    parser.add_argument("--diff-log", type=float, required=True, default=0,
                        help="Difference between the begin of the log and the video recording in seconds. "
                             "se o valor for positivo, siginfica que o wildfire comecou ANTES que o video. "
                             "se o valor for negativo, significa que o wildfire comecou DEPOIS que o video")
    args = parser.parse_args()

    # read features, eeg, and metadata
    df_f, df_eeg, metadata_eeg = read_features_and_eeg(args.input_dir)

    print(args.input_dir)
    # pprint(metadata_eeg)

    # crop or extend eeg
    df_eeg = crop_eeg(
        np_singals=df_eeg.to_numpy(),
        bitalino_start_time=metadata_eeg['time'],
        video_start_time=args.video_start_time if args.video_start_time is not None else metadata_eeg['time'],
        diff_log=args.diff_log
    )

    # add new columns to df_f
    signals = df_eeg.to_numpy()
    prev_time = 0
    prev_tabchange_time = 0
    new_data = []

    trimmed_signals = []
    prev_feats_prev_tabchange_eeg = {}
    prev_feats_prev_tabchange_bvp = {}
    prev_feats_prev_event_eeg = {}
    prev_feats_prev_event_bvp = {}

    n = len(df_f)
    for i, row in tqdm(df_f.iterrows(), total=len(df_f)):
        d = row.to_dict()

        # get time in seconds (row['time'] is 'yyyy-mm-dd hh:mm:ss')
        curr_time = hh_mm_ss_to_seconds(row['time'].split()[1])

        # deal with tabchange-level
        # print(f'[{i+1}/{n}] Cutting tabchange from {prev_tabchange_time:.2f} to {curr_time:.2f}')
        feats_tabchange_eeg, a1_signal_trimmed = get_eeg_features_for_interval(
            signals,
            a=prev_tabchange_time,
            b=curr_time,
            suffix='tabchange',
            fallback=prev_feats_prev_tabchange_eeg,
            return_trimmed_signal=True
        )
        d.update(feats_tabchange_eeg)

        feats_tabchange_bvp, a2_signal_trimmed = get_bvp_features_for_interval(
            signals,
            a=prev_tabchange_time,
            b=curr_time,
            suffix='tabchange',
            fallback=prev_feats_prev_tabchange_bvp,
            return_trimmed_signal=True
        )
        d.update(feats_tabchange_bvp)

        trimmed_signals.append(
            {
                'interval': f'{prev_tabchange_time}-{curr_time}',
                'eeg_signal': a1_signal_trimmed.tolist(),
                'bvp_signal': a2_signal_trimmed.tolist(),
                'level': 'tabchange',
            }
        )

        # deal with event-level
        # print(f'[{i+1}/{n}] Cutting event from {prev_time} to {curr_time}')
        feats_event_eeg, a1_signal_trimmed = get_eeg_features_for_interval(
            signals,
            a=prev_time,
            b=curr_time,
            suffix='event',
            fallback=prev_feats_prev_event_eeg,
            return_trimmed_signal=True
        )
        d.update(feats_event_eeg)

        feats_event_bvp, a2_signal_trimmed = get_bvp_features_for_interval(
            signals,
            a=prev_time,
            b=curr_time,
            suffix='event',
            fallback=prev_feats_prev_event_bvp,
            return_trimmed_signal=True
        )
        d.update(feats_event_bvp)

        trimmed_signals.append(
            {
                'interval': f'{prev_time}-{curr_time}',
                'eeg_signal': a1_signal_trimmed.tolist(),
                'bvp_signal': a2_signal_trimmed.tolist(),
                'level': 'event',
            }
        )

        # update row
        new_data.append(d)

        # update prevs
        if row['event'] == 'tabchange':
            prev_tabchange_time = curr_time
            prev_feats_prev_tabchange_eeg = feats_tabchange_eeg
            prev_feats_prev_tabchange_bvp = feats_tabchange_bvp
        prev_time = curr_time
        prev_feats_prev_event_eeg = feats_event_eeg
        prev_feats_prev_event_bvp = feats_event_bvp

    # save
    output_filename = args.input_dir + '/dataset_with_bit.csv'
    print('Saving features WITH bitalino signals to:', output_filename)
    df_f_new = pd.DataFrame(new_data)
    df_f_new.to_csv(output_filename, sep='\t')

    output_filename = args.input_dir + '/trimmed_signals.h5'
    # print('Saving trimmed bitalino signals to:', output_filename)
    save_signals_to_h5(trimmed_signals, output_filename)
