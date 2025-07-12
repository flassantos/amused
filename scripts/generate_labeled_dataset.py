import argparse
import datetime
import json
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd

from utils import (
    get_events_from_json,
    group_events_by_tabchange_session,
    move_tabchange_event_to_first_subevent,
    collapse_events,
    annotate_labels_in_selected_events,
)


def read_json(fname):
    """
    Read the JSON file and return the data as a list of dictionaries
    """
    with open(fname, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data


def read_tasks_json(fname):
    def fix_ms_to_hms(x):
        if x.count(':') == 1:
            return '00:' + x
        return x

    def fix_datetime(x):
        x = fix_ms_to_hms(x)
        y = pd.to_datetime(x, format='%H:%M:%S', origin='unix')
        return str(y).replace('1900', '1970')

    df_tmp = pd.read_json(fname)
    # eeg_offset = float(df_tmp['eegBvpOffset'].iloc[0])
    # face_offset = float(df_tmp['faceOffset'].iloc[0])
    tasks = df_tmp['tasks']
    df_tasks = pd.DataFrame.from_records(tasks)
    df_tasks = df_tasks.rename(columns={"startTime": "tempo_gold", "finished": "tarefa_concluida"})
    df_tasks['tempo_gold'] = df_tasks['tempo_gold'].map(fix_datetime)
    return df_tasks


def create_empty_annotations_dataframe(num_tasks=10):
    """
    Create an empty annotations dataframe for when no annotations are provided
    """
    data = []
    for i in range(num_tasks):
        # Create empty task entries with placeholder timestamps
        task_entry = {
            'tempo_gold': f'1970-01-01 00:0{i % 6}:00.000000',  # Dummy timestamps
            'tarefa_concluida': 'NaN',
            'valence': 5,  # Neutral valence
            'arousal': 5   # Neutral arousal
        }
        data.append(task_entry)
    
    return pd.DataFrame(data)


def is_valid_event(event):
    """
    An event is valid if it's a keypress event, unless it's an enter keypress event
    """
    if event['evt'] != 'keypress':
        return True
    return event['evt_data']['keyCode'] == 13


def get_episode_level_features(grouped_events):
    features = []
    for i, episode in enumerate(grouped_events):
        d = dict()

        # ids
        d['pid'] = i
        d['eid'] = 0
        d['task_id'] = episode[0].get('task_id', 0)  # Use get() for safety

        # url and xpath
        d['url'] = episode[0]['url']
        d['xpath'] = episode[0]['xpath']

        # time and unix_time
        d['time'] = episode[0]['time']
        d['unix_time'] = episode[0]['unix_time']

        # event type
        d['event'] = episode[0]['evt']

        # dom object
        d['dom_object'] = 'none'

        # add entire dict as a feature for this episode
        features.append(d)

    return features


def get_event_level_featutes(grouped_events):
    def get_dom_object_from_xpath(xpath):
        last_dom_object = xpath.strip().split('>')[-1].strip()
        if ':' in last_dom_object:
            last_dom_object = last_dom_object.split(':')[0].strip()
        return last_dom_object

    features = []
    for i, episode in enumerate(grouped_events):
        features_evt = []

        # compute features for individual events
        counter = 0
        for j, event in enumerate(episode):

            event_type = event['evt']
            if event_type == 'keypress':
                # if it's not an enter press, skip it
                if event['evt_data']['keyCode'] != 13:
                    continue

            d = dict()

            # id
            d['pid'] = i
            d['eid'] = counter
            d['task_id'] = event.get('task_id', 0)  # Use get() for safety

            # url and xpath
            d['url'] = event['url'].strip()
            d['xpath'] = event['xpath'].strip()

            # time and unix_time
            d['time'] = event['time']
            d['unix_time'] = event['unix_time']

            # event type
            d['event'] = event_type

            # dom object
            d['dom_object'] = get_dom_object_from_xpath(d['xpath'])

            # click-specific features
            if event_type == 'click':
                # 'left' if 0, 'middle' if 1, 'right' if 2, 'none' if other
                click_button = 'none'
                if event['evt_data']['button'] == 0:
                    click_button = 'left'
                elif event['evt_data']['button'] == 1:
                    click_button = 'middle'
                elif event['evt_data']['button'] == 2:
                    click_button = 'right'
                d['click_button'] = click_button
                d['click_pos_x'] = event['evt_data']['clientX']
                d['click_pos_y'] = event['evt_data']['clientY']
                d['click_text'] = event['evt_data']['innerText']

            # change-specific features
            elif event_type == 'change':
                d['change_value'] = event['evt_data']['value']
                d['change_text'] = event['evt_data']['innerText']
                d['num_key_presses'] = event['key_presses']

            # scroll-specific features
            elif event_type == 'scroll':
                d['scroll_left_start'] = event['evt_data']['scrollLeftStart']
                d['scroll_left_end'] = event['evt_data']['scrollLeftEnd']
                d['scroll_top_start'] = event['evt_data']['scrollTopStart']
                d['scroll_top_end'] = event['evt_data']['scrollTopEnd']
                d['scroll_time'] = event['evt_data']['scrollTime']
                d['num_scrolls'] = event['evt_data']['totalScrolls']

            counter += 1
            features_evt.append(d)

        # add new list of dicts as features for this episode
        features.append(features_evt)

    return features


def add_features_to_dataframe(episode_level_features, event_level_features):
    d_epi = set(episode_level_features[0].keys())
    d_evt = set()
    for epi in event_level_features:
        for d in epi:
            d_evt = d_evt.union(set(d.keys()))
    columns_to_add_to_epi = d_evt - d_epi
    columns_to_add_to_evt = d_epi - d_evt
    columns_all = d_epi.union(d_evt)

    # fill empty keys with None:
    for i in range(len(episode_level_features)):
        for c in columns_to_add_to_epi:
            episode_level_features[i][c] = None
        for j in range(len(event_level_features[i])):
            for c in columns_to_add_to_evt:
                event_level_features[i][j][c] = None

    # get a new_list with episode level and event_level concatenated
    full_log = []
    for i in range(len(episode_level_features)):
        full_log.append(episode_level_features[i])
        full_log.extend(event_level_features[i][1:])  # ignore the first tabchange

    # make sure we have a list of dicts
    for entry in full_log:
        assert isinstance(entry, dict)

    # create dataframe
    columns = [
        # overall
        'pid',
        'eid',
        'event',
        'task_id',

        # times
        'unix_time',
        'time',

        # dom object
        'dom_object',

        # url and xpath
        'url',
        'xpath',

        # labels
        'labels_0',
        'labels_1',
        'labels_2',
        'labels_valid',

        # click
        'click_text',
        'click_button',
        'click_pos_x',
        'click_pos_y',

        # change
        'change_text',
        'change_value',
        'num_key_presses',

        # scroll
        'scroll_left_end',
        'scroll_left_start',
        'scroll_top_end',
        'scroll_top_start',
        'scroll_time',
        'num_scrolls',
    ]
    df = pd.DataFrame(full_log, columns=columns)
    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generate labeled dataset from interaction logs and expert annotations"
    )
    parser.add_argument("--input-json", type=str, required=True, 
                       help="Path to interaction log .json file")
    parser.add_argument("--input-tasks", type=str, required=False, default=None,
                       help="Path to expert annotations .json file (optional)")
    parser.add_argument("--no-annotations", action='store_true',
                       help="Generate dataset without annotations (empty labels)")
    parser.add_argument("--diff-seconds", type=float, required=True, default=0,
                        help="Difference between the begin of the log and the video recording in seconds. "
                             "If positive, the video starts `diff-seconds` AFTER the log. "
                             "If negative, the video starts `diff-seconds` BEFORE the log.")
    parser.add_argument("--output", type=str, default=None, 
                       help="Path for output .csv file")
    parser.add_argument("--print-events", action='store_true', 
                       help="Print the whole list of events")
    parser.add_argument("--nonverbose", action='store_true', 
                       help="Only print success/failure messages")
    parser.add_argument("--valid-urls", type=str, nargs='+', default=None, 
                       help="Valid urls to consider")
    args = parser.parse_args()

    verbose = not args.nonverbose

    # Validate arguments
    if args.no_annotations and args.input_tasks:
        parser.error("Cannot specify both --no-annotations and --input-tasks")
    if not args.no_annotations and not args.input_tasks:
        parser.error("Must specify either --input-tasks or --no-annotations")

    # define the valid urls used in the experiments (without www)
    valid_urls = args.valid_urls
    if args.valid_urls is None or len(args.valid_urls) == 0 or args.valid_urls[0].strip() == '' or args.valid_urls[0].strip().lower() == 'none':
        valid_urls = ['perspectivesocialnetwork.com', 'social-network.link', 'love-social.firebaseapp.com']

    # read data
    data_log = read_json(args.input_json)

    # get the Unix-like timestamp of the first event
    start_unix_time = data_log[0]['time']

    # select the subset of relevant events, filter out irrelevant data, and add other helpful info
    selected_events = get_events_from_json(data_log, start_unix_time=start_unix_time, valid_urls=valid_urls)

    # group events by tabchange, such that each tabchange delimits a list
    grouped_selected_events = group_events_by_tabchange_session(selected_events)

    # make the tabchange event to be the first item of each sublist
    grouped_selected_events = move_tabchange_event_to_first_subevent(grouped_selected_events)

    # the format of grouped_selected_events is:
    # [list_of_events_of_tabchange_id_1, ..., list_of_events_of_tabchange_id_n]

    # print useful information
    if verbose:
        print('Number of episodes:', len(grouped_selected_events))
        print('Number of events in each episode:', list(map(len, grouped_selected_events)))

    # print events history
    if args.print_events and verbose:
        from rich import print as rprint
        print('Full event history: ')
        rprint(grouped_selected_events)

    # Handle annotations
    if args.no_annotations:
        if verbose:
            print("Running without annotations - creating empty labels")
        # Create empty annotations
        ann_df = create_empty_annotations_dataframe(num_tasks=len(grouped_selected_events))
        annotator_id = 1
        
        # Initialize empty labels for all events
        ann_grouped_selected_events = deepcopy(grouped_selected_events)
        for episode in ann_grouped_selected_events:
            for event in episode:
                event['labels'] = []
                event['task_id'] = 0
        
        last_episode = []
        last_smells = []
        
    else:
        # read annotation
        ann_df = read_tasks_json(args.input_tasks)
        tempo_gold = ann_df['tempo_gold']
        annotator_id = 1

        # create a copy of grouped_selected_events recursively
        ann_grouped_selected_events = deepcopy(grouped_selected_events)

        # annotate events by fuzzy-matching timestamps
        smells = ['tarefa_concluida']
        last_episode, last_smells = annotate_labels_in_selected_events(
            ann_df, ann_grouped_selected_events, smells, verbose=verbose)

    # create features from grouped_selected_events
    episode_level_features = get_episode_level_features(ann_grouped_selected_events)
    ann_grouped_selected_events = collapse_events(ann_grouped_selected_events)
    event_level_features = get_event_level_featutes(ann_grouped_selected_events)
    log_df = add_features_to_dataframe(episode_level_features, event_level_features)

    # add labels to the dataframe
    if args.no_annotations:
        # Add empty labels for all rows
        log_df[f'labels_{annotator_id-1}'] = [[] for _ in range(len(log_df))]
    else:
        labels = [list(set(evt['labels'])) for epi in ann_grouped_selected_events for evt in epi]
        diff_size = len(log_df) - len(labels)
        log_df[f'labels_{annotator_id-1}'] = labels + [[]]*diff_size

        # add one more row if the last_episode is an empty list:
        if len(last_episode) == 0 and len(last_smells) > 0:
            if verbose:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('LAST SMELLS WERE NOT USED! Total: ', len(last_smells))
                print('ANNOTATE MANUALLY!')
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            vals = dict(zip(list(log_df.columns), [None]*len(log_df.columns)))
            vals[f'labels_{annotator_id-1}'] = last_smells
            try:
                log_df = log_df.append(vals)
            except:
                new_df = pd.DataFrame([vals], columns=log_df.columns)
                log_df = pd.concat([log_df, new_df], ignore_index=True)

    # add sam information
    if 'valence' in ann_df.columns and 'arousal' in ann_df.columns:
        valence = ann_df['valence'].tolist()
        arousal = ann_df['arousal'].tolist()
        log_df['sam_valence'] = log_df.apply(
            lambda row: valence[row['task_id'] - 1] if row['task_id'] > 0 and row['task_id'] <= len(valence) else None,
            axis=1
        )
        log_df['sam_arousal'] = log_df.apply(
            lambda row: arousal[row['task_id'] - 1] if row['task_id'] > 0 and row['task_id'] <= len(arousal) else None,
            axis=1
        )
    else:
        if verbose:
            print("No valence/arousal data found in annotations")
        log_df['sam_valence'] = None
        log_df['sam_arousal'] = None

    # save dataframe as csv
    if args.output is None:
        output_filename = '/'.join(args.input_json.split('/')[:-1])
        output_filename += '/dataset.csv'
    else:
        output_filename = args.output

    log_df.to_csv(output_filename, sep='\t')
    
    if args.no_annotations:
        print(f'Unlabeled dataset saved to: {output_filename}')
        print(f'Dataset contains {len(log_df)} events across {len(grouped_selected_events)} episodes')
    else:
        print(f'Labeled dataset saved to: {output_filename}')
        print(f'Dataset contains {len(log_df)} events with expert annotations')
