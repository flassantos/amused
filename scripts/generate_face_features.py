import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm


def read_feats_with_bit_csv(basepath):
    filename = basepath+'/dataset_with_bit.csv'
    if os.path.exists(filename):
        df_f = pd.read_csv(filename, sep='\t', index_col=0)
    else:
        filename_old = basepath+'/dataset.csv'
        df_f = pd.read_csv(filename_old, sep='\t', index_col=0)
    return df_f


def read_face_emo_json(basepath):
    # emotion index:
    # 0: anger
    # 1: disgust
    # 2: fear
    # 3: enjoyment
    # 4: contempt
    # 5: sadness
    # 6: surprise
    with open(basepath+'/face.json', 'r') as file:
        loaded_data = json.load(file)
    return loaded_data


def mm_ss_ms_to_s(mm_ss_ms):
    m, s, ms = mm_ss_ms.split(':')
    return float(m) * 60 + float(s) + float(ms) / 1000


def convert_to_timestamp(total_seconds):
    # Extract minutes, seconds, and milliseconds from total_seconds
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - minutes * 60 - seconds) * 1000)
    # Format the string as 'mm:ss:ms'
    timestamp = f"{minutes:02}:{seconds:02}:{milliseconds:03}"
    return timestamp


def crop_face(data_face, diff_log=-24, diff_face=-10):
    # diff_log:
        # esse eh a diferenca de tempo entre o comeco do video (gravacao da tela) e o comeco do log (gravacao do wildfire)
        # se o valor for positivo, siginfica que o wildfire comecou ANTES que o video
        # se o valor for negativo, significa que comecou DEPOIS
    
    # diff_face:
        # esse eh a diferenca de tempo entre o comeco do video (gravacao da tela) e o comeco da gravacao da face)
        # se o valor for positivo, siginfica que a face comecou ANTES que o video
        # se o valor for negativo, significa que a face comecou DEPOIS

    # if and elses are here only for illustrative purposes, 
    # note that the delta equation is the same regardless of the case
    if diff_log < diff_face < 0:
        print('CASO 1 (F -> L -> V)')
        delta = diff_log - diff_face

    elif diff_face < diff_log < 0:
        print('CASO 2 (L -> F -> V)')
        delta = diff_log - diff_face
        
    elif diff_log < 0 < diff_face:
        print('CASO 3 (F -> V -> L)')
        delta = diff_log - diff_face

    print('Delta:', delta)

    def get_data_face_new(data_face_trimmed, delta):
        data_face_new = [] 
        for item in data_face_trimmed:
            d = {
                's': item['s'] + delta,
                'timestamp': convert_to_timestamp(item['s'] + delta),
                'emotions': item['emotions'],
                's_old': item['s'],
                'timestamp_old': item['timestamp']
                
            }
            data_face_new.append(d)
        return data_face_new
    

    # se delta for negativo, significa que a face veio antes que o log
    # entao, vou cortar o que veio antes, e ajustar o tempo dos que vem depois
    # com timestamp - abs(delta)
    if delta < 0:
        for crop_index, item in enumerate(data_face):
            if item['s'] > abs(delta):
                break
        print(f'Cutting face data from index {crop_index} and adding delta to timestamps...')
        data_face_new = get_data_face_new(data_face[crop_index:], delta)

    # caso contrario, a face vem depois que o log
    # entao vou adicionar o delta para todos os itens: timetamp + abs(delta)
    else:
        print('No need to cut. Adding delta to timestamps...')
        data_face_new = get_data_face_new(data_face, delta)
        

    return data_face_new


def get_emotions_from_data(data):
    emotions = [item['emotions'] for item in data]
    return np.array(emotions)


def save_signals_to_h5(trimmed_signals, filename):
    print('Saving trimmed signals to:', filename)
    with h5py.File(filename, 'w') as file:
        for i, item in enumerate(trimmed_signals):
            group = file.create_group(f'item_{i}')
            group.attrs['interval'] = item['interval']
            group.attrs['level'] = item['level']
            emotions = get_emotions_from_data(item['face_trimmed'])
            group.create_dataset('emotions', data=emotions)


def load_h5_signals(filename):
    print('Loading trimmed signals from:', filename)
    loaded_data = []
    with h5py.File(filename, 'r') as file:
        for key in file.keys():
            group = file[key]
            item = {
                'interval': group.attrs['interval'],
                'level': group.attrs['level'],
                'emotions': np.array(group['emotions']),
            }
            loaded_data.append(item)
    return loaded_data


def hh_mm_ss_to_seconds(s):
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


def get_sliced_data(data, lim_a, lim_b):
    idx_a = None
    idx_b = None
    
    for i, d in enumerate(data_face_new):
        if d['s'] > lim_a and idx_a is None:
            idx_a = i
            
        if d['s'] > lim_b and idx_b is None:
            idx_b = i
            break
    
    if idx_a is None:
        return [], idx_a, idx_b
    else:
        return data[idx_a:idx_b], idx_a, idx_b


def get_features_for_interval(data, a, b, suffix=None, fallback=None, return_trimmed_signal=False):
    lim_a = float(a)
    lim_b = float(b)
    data_sliced, idx_a, idx_b = get_sliced_data(data, lim_a, lim_b)
    
    if len(data_sliced) == 0 or lim_a == lim_b:
        # print('empty list')
        if return_trimmed_signal:
            return fallback, data_sliced
        return fallback
    
    # print(lim_a, lim_b, data_sliced[0]['s'], data_sliced[-1]['s'], idx_a, idx_b)

    emotions = get_emotions_from_data(data_sliced)

    num_faces = emotions.shape[0]
    num_emotions = emotions.shape[1]
    emo_names = ['anger', 'disgust', 'fear', 'enjoyment', 'contempt', 'sadness', 'surprise']
    assert len(emo_names) == num_emotions

    d_feat = {}
    emo_avg_probas = emotions.mean(axis=0)
    for i, emo_name in enumerate(emo_names):
        d_feat[f'face_avg_{i}_{emo_name}_proba_{suffix}'] = emo_avg_probas[i]

    emo_max_prob = np.argmax(emo_avg_probas)
    d_feat[f'face_avg_most_likely_{suffix}'] = emo_max_prob

    emo_avg_entropy = (-emo_avg_probas * np.log(emo_avg_probas + 1e-9)).sum()
    d_feat[f'face_avg_entropy_{suffix}'] = emo_avg_entropy

    emo_most_freq = stats.mode(emo_avg_probas).mode
    d_feat[f'face_avg_most_freq_{suffix}'] = emo_most_freq

    d_feat[f'face_lim_a_{suffix}'] = lim_a
    d_feat[f'face_lim_b_{suffix}'] = lim_b
    
    if return_trimmed_signal:
        return d_feat, data_sliced
    return d_feat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True, help="Dir path")
    parser.add_argument("--diff-log", type=float, required=True, default=0,
                        help="Difference between the begin of the log and the video recording in seconds. "
                             "se o valor for positivo, siginfica que o wildfire comecou ANTES que o video. "
                             "se o valor for negativo, significa que o wildfire comecou DEPOIS que o video")
    parser.add_argument("--diff-face", type=float, required=True, default=0,
                        help="Difference between the begin of the face and the video recording in seconds. "
                             "se o valor for positivo, siginfica que a face comecou ANTES que o video. "
                             "se o valor for negativo, significa que a face comecou DEPOIS que o video")
    args = parser.parse_args()

    print(args.input_dir)
    df = read_feats_with_bit_csv(args.input_dir)

    data = read_face_emo_json(args.input_dir)
    data_face = [{'timestamp': k, 's': mm_ss_ms_to_s(k), 'emotions': v} for k, v in data.items()]
    data_face = list(sorted(data_face, key=lambda x: x['s']))
    data_face_new = crop_face(data_face, diff_log=args.diff_log, diff_face=args.diff_face)

    prev_time = 0
    prev_tabchange_time = 0
    prev_feats_prev_tabchange = {}
    prev_feats_prev_event = {}

    new_data = []
    trimmed_signals = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        d = row.to_dict()
        
        # get time in seconds (row['time'] is 'yyyy-mm-dd hh:mm:ss')
        curr_time = hh_mm_ss_to_seconds(row['time'].split()[1])


        # deal with tabchange-level
        # print(f'[{i+1}/{len(df)}] Cutting tabchange from {prev_tabchange_time:.2f} to {curr_time:.2f}')
        feats_tabchange, data_face_trimmed = get_features_for_interval(
            data_face_new, 
            a=prev_tabchange_time, 
            b=curr_time, 
            suffix='tabchange', 
            fallback=prev_feats_prev_tabchange,
            return_trimmed_signal=True
        )
        d.update(feats_tabchange)
        trimmed_signals.append(
            {
                'interval': f'{prev_tabchange_time}-{curr_time}',
                'face_trimmed': data_face_trimmed,
                'level': 'tabchange',
            }
        )

        # print(f'[{i+1}/{len(df)}] Cutting event from {prev_time} to {curr_time}')
        feats_event, data_face_trimmed = get_features_for_interval(
            data_face_new, 
            a=prev_time, 
            b=curr_time, 
            suffix='event', 
            fallback=prev_feats_prev_event,
            return_trimmed_signal=True
        )
        d.update(feats_event)
        trimmed_signals.append(
            {
                'interval': f'{prev_time}-{curr_time}',
                'face_trimmed': data_face_trimmed,
                'level': 'event',
            }
        )

        # update row
        new_data.append(d)

        # update prevs
        if row['event'] == 'tabchange':
            prev_tabchange_time = curr_time
            prev_feats_prev_tabchange = feats_tabchange
        prev_time = curr_time
        prev_feats_prev_event = feats_event


    # save
    output_filename = args.input_dir + '/dataset_with_bit_and_face.csv'
    print('Saving features WITH face emotions to:', output_filename)
    df_f_new = pd.DataFrame(new_data)
    df_f_new.to_csv(output_filename, sep='\t')

    output_filename = args.input_dir + '/trimmed_face.h5'
    print('Saving trimmed face emotions to:', output_filename)
    save_signals_to_h5(trimmed_signals, output_filename)
