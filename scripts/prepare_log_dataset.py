#!/usr/bin/env python
# coding: utf-8
"""
Dataset Preparation Pipeline

This script processes raw user interaction datasets by cleaning, normalizing,
and enriching the data for analysis. It handles events such as clicks, changes,
scrolls, and tab changes, while anonymizing sensitive information.

Usage:
    python prepare_dataset.py <input_path> [<output_path>]
"""

import sys
import os
import re
import hashlib
from collections import Counter
from datetime import datetime
from pprint import pprint

import pandas as pd
import numpy as np
import scipy
import strsimpy


# ============= DATA LOADING FUNCTIONS =============

def read_feats_csv(dirname):
    """
    Read dataset from the most complete CSV file available in the directory.

    Args:
        dirname: Directory containing the dataset files

    Returns:
        pandas.DataFrame: The loaded dataset with basic metadata added
    """
    filename_options = [
        os.path.join(dirname, 'dataset_with_bit_and_face_and_sam.csv'),
        os.path.join(dirname, 'dataset_with_bit_and_face.csv'),
        os.path.join(dirname, 'dataset_with_bit.csv'),
        os.path.join(dirname, 'dataset.csv')
    ]

    df_f = None
    for filename in filename_options:
        if os.path.exists(filename):
            print(f"Reading dataset from {filename}")
            df_f = pd.read_csv(filename, sep='\t', index_col=0)
            break

    if df_f is None:
        raise FileNotFoundError(f"No dataset file found in {dirname}")

    # Add metadata placeholders
    df_f['dirname'] = dirname
    df_f['network'] = 'SN_1'
    df_f['username'] = 'none'

    return df_f


# ============= TEXT PROCESSING HELPERS =============

def trim(text):
    """Remove multiple spaces from text."""
    return re.sub(r'\ +', ' ', text)


def remove_punctuation(text):
    """Replace punctuation with spaces."""
    import string
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)


def str_to_list(x):
    """Convert string representation of a list to an actual list."""
    try:
        return eval(x) if isinstance(x, str) else []
    except (ValueError, SyntaxError):
        return []


def create_secure_hash(text):
    """
    Create a deterministic hash for a given text with added salt.

    Args:
        text: Text to be hashed

    Returns:
        str: Hashed value or None if input is NaN
    """
    if pd.isna(text):
        return None
    salt = "32;lgd312$%&$^s0i423vvSDF6s4!@#$"
    salted_text = text + salt
    return hashlib.sha256(salted_text.encode("utf-8")).hexdigest()[:32]


# ============= EVENT FILTERING FUNCTIONS =============

def filter_click_text(text):
    """
    Filter and normalize click text content.

    Performs modifications like:
    - Converting emails to a placeholder
    - Converting phone numbers to placeholders
    - Converting CPF numbers to placeholders
    - Removing invalid click texts

    Args:
        text: Raw click text

    Returns:
        str: Filtered and normalized text
    """
    text = str(text).lower()

    # Regular expression patterns
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    phone_pattern = re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3,4}\b')
    phone_pattern2 = re.compile(r'\b\d{4}[-.\s]?\d{4,5}\b')
    cpf_pattern = re.compile(r'\b\d{3}.?d{3}.?d{3}-?d{2}\b')

    # Basic filters
    if len(text) > 100:
        return ''
    if text in ['nan', 'empty']:
        return ''
    if re.search(r'^\(\d+\)$', text):
        return '(0)'
    if re.search(r'^\d+$', text):
        return '0'

    # Pattern-based replacements
    if re.search(r'^curtir(\s*\(\d+\))?$', text):
        return 'curtir'
    if email_pattern.search(text):
        return 'email@email.com'
    if phone_pattern.search(text) or phone_pattern2.search(text):
        return '000-000-000'
    if cpf_pattern.search(text):
        return '000.000.000-00'

    # Specific text replacements
    if text.startswith('sign up'):
        return 'sign up'
    if text.startswith('selecione seu sexo'):
        return 'selecione seu sexo'
    if '\nchange image\n' in text:
        return 'change image'
    if 'likes:\n 0\ncomments:\n' in text:
        return 'likes and comments:'
    if text == 'loadingâ\x80¦':
        return 'loading'
    if len(text) == 1 and text.isalpha():
        return ''
    if re.search(r'^\d+:\d+$', text):
        return ''
    if 'sejam bem vindos' in text:
        return ''
    if text in ['fotos', 'foto']:
        return 'foto'
    if '\n' in text:
        return ''
    if text == 'postmessagemodal 0':
        return 'postmessagemodal'
    if text == 'slave delete 0':
        return 'delete'

    # Final cleanup
    text = text.replace('ã\xa0', 'a')
    text = trim(remove_punctuation(text)).strip()
    return text


def filter_change_text(text):
    """
    Filter and normalize change text content.

    Args:
        text: Raw change text

    Returns:
        str: Filtered and normalized text
    """
    text = str(text).lower()

    # Regular expression patterns
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    phone_pattern = re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3,4}\b')
    phone_pattern2 = re.compile(r'\b\d{4}[-.\s]?\d{4,5}\b')
    cpf_pattern = re.compile(r'\b\d{3}.?d{3}.?d{3}-?d{2}\b')

    # Basic filters
    if len(text) > 100:
        return ''
    if text in ['nan', 'empty']:
        return ''
    if re.search(r'^\(\d+\)$', text):
        return '(0)'
    if re.search(r'^\d+$', text):
        return '0'

    # Pattern-based replacements
    if email_pattern.search(text):
        return 'email@email.com'
    if phone_pattern.search(text) or phone_pattern2.search(text):
        return '000-000-000'
    if cpf_pattern.search(text):
        return '000.000.000-00'

    # Specific text replacements
    if text.startswith('sign up'):
        return 'sign up'
    if text.startswith('selecione seu sexo'):
        return 'selecione seu sexo'

    # Final cleanup
    text = text.replace('ã\xa0', 'a')
    text = trim(remove_punctuation(text)).strip()
    return text


def filter_url_text(text, user_parts_to_filter=None):
    """
    Filter and normalize URL content, anonymizing user-specific parts.

    Args:
        text: Raw URL
        user_parts_to_filter: Set to collect user parts for later checking

    Returns:
        str: Filtered and normalized URL
    """
    text = str(text).lower()
    return text


def replace_span_inside_button_in_xpath(xpath, current_dom):
    """
    Replace <span> elements inside <button> or <a> elements with their parent.

    Args:
        xpath: XPath string
        current_dom: Current DOM object type

    Returns:
        str: Replaced DOM object if needed, or original
    """
    parts = xpath.split('>')
    max_previous_parts = parts[-4:-1]  # Last 3 elements before the current one
    has_button = any('button' in p.strip() for p in max_previous_parts)
    has_a = any('a' in p.strip() for p in max_previous_parts)
    is_replaceable = current_dom == 'span'

    if is_replaceable and has_button:
        return 'button'
    elif is_replaceable and has_a:
        return 'a'
    return current_dom


def group_dom_object(dom):
    """
    Group similar DOM object types into more general categories.

    Args:
        dom: Original DOM object type

    Returns:
        str: Grouped DOM object type
    """
    if re.match(r'mudou_a_url\(\d+\)', dom):
        return 'none'
    elif dom in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        return 'h1'
    elif dom in ['img', 'svg', 'path', 'circle', 'use']:
        return 'img'
    elif dom in ['i', 'b']:
        return 'i'
    elif dom in ['ul', 'ol']:
        return 'ul'
    elif dom in ['li']:
        return 'li'
    elif dom in ['body', 'main', 'header']:
        return 'body'
    elif dom in ['div', 'center']:
        return 'div'
    elif dom in ['p', 'br']:
        return 'p'
    return dom


# ============= DATAFRAME PROCESSING FUNCTIONS =============

def get_column_names(dfs):
    """Get a list of all unique column names across all dataframes."""
    all_columns = []
    for df in dfs:
        for c in df.columns.tolist():
            if c not in all_columns:
                all_columns.append(c)
    return all_columns


def get_count_event_types(dfs):
    """Count the occurrences of each event type across all dataframes."""
    ct = Counter()
    for i, df in enumerate(dfs):
        e = df['event'].value_counts().to_dict()
        ct += Counter(e)
    return ct


def filter_events(df):
    """
    Filter out unwanted events and transform certain events.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame: Filtered DataFrame
    """
    # Remove rows where event is in the list
    for e in ['input', 'clipboard_paste', 'clipboard_cut']:
        if e in df['event'].unique():
            # Remove rows
            df = df[df['event'] != e]
            # Reset the eid count for each pid group
            df.loc[:, 'eid'] = df.groupby('pid').cumcount()

    # Transform copy to click
    if 'clipboard_copy' in df['event'].unique():
        df.loc[:, 'event'] = df['event'].map(lambda x: x.replace('clipboard_copy', 'click'))

    return df


def calculate_rid(group):
    """
    Calculate relative event ID within an episode.

    Args:
        group: DataFrame group by episode (pid)

    Returns:
        DataFrame: Group with added 'rid' column
    """
    max_eid = len(group) - 1
    group['rid'] = group['eid'] / max_eid if max_eid > 0 else 1
    return group


def calculate_episode_duration(group, event_type):
    """
    Calculate duration of events of a specific type per episode.

    Args:
        group: DataFrame group by episode (pid)
        event_type: Type of event (or 'total' for all events)

    Returns:
        DataFrame: Group with added duration column
    """
    if event_type != 'total':
        event_duration = group[group['event'] == event_type]['event_duration'].sum()
    else:
        event_duration = group['event_duration'].sum()

    group[f'episode_{event_type}_duration'] = event_duration
    return group


def calculate_episode_event_count(group, event_type):
    """
    Calculate number of events of a specific type per episode.

    Args:
        group: DataFrame group by episode (pid)
        event_type: Type of event (or 'total' for all events)

    Returns:
        DataFrame: Group with added count column
    """
    if event_type == 'scroll':
        group[f'episode_{event_type}_num'] = group[group['event'] == 'scroll']['num_scrolls'].sum()
    elif event_type == 'keypress':
        group[f'episode_{event_type}_num'] = group[group['event'] == 'change']['num_key_presses'].sum()
    elif event_type != 'total':
        group[f'episode_{event_type}_num'] = sum(group['event'] == event_type)
    else:
        group[f'episode_{event_type}_events'] = len(group)

    return group


def fix_negative_durations(df, k=2):
    """
    Fix cases of negative event durations by recalculating time differences.

    Args:
        df: DataFrame with potential negative durations
        k: Number of positions to look back for time reference

    Returns:
        tuple: (Fixed DataFrame, count of remaining negative durations)
    """
    orig_index = df[df['event_duration'] < 0].index
    if len(orig_index) == 0:
        return df, 0

    # Initialize arrays for time differences
    time_difference_seconds = []
    valid_indices = []

    # Process each index with negative duration
    for idx in orig_index:
        prev_idx = idx - k
        # Skip if previous index would be negative
        if prev_idx < 0:
            continue

        time_prev = pd.to_datetime(df.loc[prev_idx, 'time'])
        time_curr = pd.to_datetime(df.loc[idx, 'time'])

        # Calculate time difference
        time_difference = time_curr - time_prev
        time_diff_sec = time_difference.total_seconds()

        # Store results
        time_difference_seconds.append(time_diff_sec)
        valid_indices.append(idx)

    # Update event durations for valid indices
    if valid_indices:
        df.loc[valid_indices, 'event_duration'] = time_difference_seconds

    # Count remaining negative durations
    rem = sum(time_diff_sec < 0 for time_diff_sec in time_difference_seconds) if time_difference_seconds else 0
    return df, rem


def calculate_max_similarity(df, group_by=None):
    """
    Calculate maximum Levenshtein similarity for change values.

    Args:
        df: DataFrame with change values
        group_by: Column to group by for episode-level similarity

    Returns:
        DataFrame: With added similarity columns
    """
    norm_levenshtein = strsimpy.NormalizedLevenshtein()

    # For each text, compute its similarity to all other texts and get the maximum
    def compute_similarities(texts):
        max_similarities = []
        for i, text in enumerate(texts):
            if pd.isna(text):
                max_similarities.append(np.nan)
                continue

            similarities = [
                norm_levenshtein.similarity(text, other_text)
                for j, other_text in enumerate(texts)
                if i != j and not pd.isna(other_text)
            ]
            max_similarities.append(max(similarities) if similarities else 1.0)
        return max_similarities

    # Compute maximum similarity either per episode or per entire session
    if group_by:
        df['change_value_sim_episode'] = df.groupby(group_by)['change_value'].transform(
            lambda x: compute_similarities(x.tolist()))
    else:
        df['change_value_sim_session'] = compute_similarities(df['change_value'].tolist())

    return df


def has_tarefa_concluida(row, valid_cols=['labels_0', 'labels_1', 'labels_2']):
    """
    Check if any of the label columns contains 'tarefa_concluida'.

    Args:
        row: DataFrame row
        valid_cols: List of label columns to check

    Returns:
        bool: True if 'tarefa_concluida' is present
    """
    labels = [set(str_to_list(row[c])) for c in valid_cols]
    labels = [set(['none']) if len(l) == 0 else l for l in labels]

    # Unite all labels
    united_labels = set()
    for l in labels:
        united_labels = united_labels.union(l)

    return 'tarefa_concluida' in united_labels


# ============= FOR DF_ALL =============

# Define the function to generate new features
def generate_episode_level_global_duration_features(df, col_name):
    def gen_features(df_g, col):
        # Calculate the features
        max_val = df_g[col].max()
        min_val = df_g[col].min()
        last_val = df_g[col].iloc[-1]  # Last value
        first_val = df_g[col].iloc[0]  # First value
        second_val = df_g[col].iloc[1] if len(df_g[col]) > 1 else 0  # Second value
        avg_val = df_g[col].mean()

        # Assign the calculated features as new columns
        df_g[f'{col}_max'] = max_val
        df_g[f'{col}_min'] = min_val
        df_g[f'{col}_last'] = last_val
        df_g[f'{col}_first'] = first_val
        df_g[f'{col}_second'] = second_val
        df_g[f'{col}_avg'] = avg_val

        return df_g

    # Apply the function to each group
    df = df.groupby(['username', 'network', 'pid']).apply(lambda x: gen_features(x, col_name))

    # Reset index to avoid ambiguity
    df = df.reset_index(drop=True)
    return df


def generate_episode_level_local_duration_features(df, col_name='episode_total_duration', k_range=3):
    # Filter for 'eid' == 0
    sel = df['eid'] == 0

    # Generate columns for previous events
    for k in range(1, k_range + 1):
        df.loc[sel, f'{col_name}_prev_{k}'] = df.loc[sel, col_name].shift(k)

    # Generate columns for next events
    for k in range(1, k_range + 1):
        df.loc[sel, f'{col_name}_next_{k}'] = df.loc[sel, col_name].shift(-k)

    # Convert NaN to 0 for the corner cases
    df = df.fillna(0)

    return df

# ============= MAIN PROCESSING PIPELINE =============

def prepare_dataset(input_path, output_path=None):
    """
    Main function to prepare the dataset.

    Args:
        input_path: Path to input directory with dataset files
        output_path: Path to output directory (defaults to input directory)
    """
    # Set output path
    if output_path is None:
        output_path = os.path.dirname(os.path.abspath(input_path))

    # Load data
    dirname = os.path.dirname(input_path)
    dfs = [
        read_feats_csv(dirname)  # add only ony dataset for now
    ]

    # Initialize shared variables
    user_parts_to_filter = set()

    # Process columns per network
    valid_cols_per_network = {}
    for df in dfs:
        network = df['network'].iloc[0]
        username = df['username'].iloc[0]
        if network not in valid_cols_per_network:
            valid_cols_per_network[network] = {}
        labels_valid = ['labels_0']
        valid_cols_per_network[network][username] = list(sorted(labels_valid))

    # ===== Event Type Filtering =====
    print("Processing event types...")
    print(get_count_event_types(dfs))

    for i in range(len(dfs)):
        dfs[i] = filter_events(dfs[i])

    ct = get_count_event_types(dfs)
    for k, v in sorted(ct.items(), key=lambda x: -x[1]):
        print('{:10s} {:4d} ({:.1f}%)'.format(k + ':', v, 100 * v / sum(ct.values())))

    # ===== Column Cleanup =====
    print("\nCleaning up columns...")
    for i in range(len(dfs)):
        df = dfs[i]

        # Drop irrelevant columns
        df.drop(columns=['scroll_left_end', 'scroll_left_start', 'scroll_top_end', 'scroll_top_start'],
                inplace=True, errors='ignore')

        # Convert to int
        df['num_scrolls'] = df['num_scrolls'].astype('Int64')
        df['num_key_presses'] = df['num_key_presses'].astype('Int64')

        # Remove repetitive labels
        for l in [0, 1, 2]:
            df[f'labels_{l}'] = df[f'labels_{l}'].apply(lambda x: str(list(set(str_to_list(x)))))

        dfs[i] = df

    # ===== DOM Object Processing =====
    print("\nProcessing DOM objects...")
    ct = Counter()

    for i in range(len(dfs)):
        df = dfs[i]

        # Replace span inside button/a in xpath
        df['dom_object'] = df.apply(
            lambda row: replace_span_inside_button_in_xpath(row['xpath'], row['dom_object']), axis=1)

        # Group DOM objects
        df['dom_object'] = df['dom_object'].apply(group_dom_object)

        ct += Counter(df['dom_object'].value_counts().to_dict())
        dfs[i] = df

    for k, v in sorted(ct.items(), key=lambda x: -x[1]):
        print('{:9s} {:4d} ({:.1f}%)'.format(k + ':', v, 100 * v / sum(ct.values())))

    # ===== Relative Event ID Calculation =====
    print("\nCalculating relative event IDs...")
    for i in range(len(dfs)):
        dfs[i] = dfs[i].groupby('pid').apply(calculate_rid).reset_index(drop=True)
        dfs[i]['srid'] = np.linspace(0, 1, len(dfs[i]))

    # ===== Duration Calculation =====
    print("\nCalculating duration metrics...")
    for i in range(len(dfs)):
        df = dfs[i]

        # Convert time to datetime for delta computation
        df['datetime'] = pd.to_datetime(df['time'])

        # Calculate time differences
        df['event_duration'] = df['datetime'].diff().dt.total_seconds()
        df['event_duration'] = df['event_duration'].fillna(0)
        df['event_cum_duration'] = df.groupby('pid')['event_duration'].cumsum()

        # Calculate per-episode durations for different event types
        for event_type in ['click', 'change', 'scroll', 'tabchange', 'total']:
            df = df.groupby('pid').apply(
                lambda x: calculate_episode_duration(x, event_type)).reset_index(drop=True)

        dfs[i] = df

    # ===== Fix Negative Durations =====
    print("\nFixing negative durations...")
    error = False
    for i in range(len(dfs)):
        df = dfs[i]
        has_neg_duration = (df['event_duration'] < 0).any()

        if has_neg_duration:
            network = df['network'].iloc[0]
            username = df['username'].iloc[0]
            for k in [2, 3, 4, 5]:
                print(f'Fixing neg duration with k = {k} for {network}/{username}')
                df, rem = fix_negative_durations(df, k=k)
                if not rem:
                    break

        dfs[i] = df

    error = any([(df['event_duration'] < 0).any() for df in dfs])
    print('OK' if not error else 'ERROR')

    if error:
        # set duration to 0 for negative values
        for i in range(len(dfs)):
            dfs[i].loc[dfs[i]['event_duration'] < 0, 'event_duration'] = 0

    # ===== Event Count Calculation =====
    print("\nCalculating event counts...")
    for i in range(len(dfs)):
        df = dfs[i]

        for event_type in ['click', 'change', 'scroll', 'tabchange', 'keypress', 'total']:
            df = df.groupby('pid').apply(
                lambda x: calculate_episode_event_count(x, event_type)).reset_index(drop=True)

        dfs[i] = df

    # ===== Click Event Processing =====
    print("\nProcessing click events...")
    for i in range(len(dfs)):
        df = dfs[i]
        df['click_text'] = df['click_text'].map(filter_click_text)
        df['click_text_length'] = df['click_text'].map(lambda x: len(x) if not pd.isna(x) else 0)
        dfs[i] = df

    # ===== Change Event Processing =====
    print("\nProcessing change events...")
    for i in range(len(dfs)):
        df = dfs[i]

        # Apply the filtering function
        df['change_text'] = df['change_text'].apply(filter_change_text)
        df['change_text_length'] = df['change_text'].map(lambda x: len(x) if not pd.isna(x) else 0)

        # Calculate similarity values
        df = calculate_max_similarity(df)
        df = calculate_max_similarity(df, group_by='pid')

        # Compute value length and hash
        df['change_value_length'] = df['change_value'].map(lambda x: len(x) if not pd.isna(x) else 0)
        df['change_value'] = df['change_value'].map(lambda x: create_secure_hash(x))

        dfs[i] = df

    # ===== URL Processing =====
    print("\nProcessing URLs...")
    url_counter = Counter()
    for i in range(len(dfs)):
        df = dfs[i]
        df['url'] = df['url'].apply(lambda x: filter_url_text(x, user_parts_to_filter))
        url_counter += Counter(df['url'].value_counts().to_dict())
        dfs[i] = df

    print('All URLs:')
    for k, v in sorted(url_counter.items(), key=lambda x: (x[0].split('/')[2], -x[1])):
        print(repr("{}: {}".format(k, v)))

    # ===== Process 'tarefa_concluida' Label =====
    print("\nProcessing task completion labels...")
    for i in range(len(dfs)):
        df = dfs[i]

        network = df.iloc[0]['network']
        username = df.iloc[0]['username']
        valid_cols = ['labels_0']

        df['finished_task'] = df.apply(lambda x: has_tarefa_concluida(x, valid_cols), axis=1)

        # Remove 'tarefa_concluida' from labels
        for c in valid_cols:
            set_tarefa_concluida = set(['tarefa_concluida'])
            df[c] = df[c].apply(lambda x: str(list(set(str_to_list(x)) - set_tarefa_concluida)))

        dfs[i] = df

    # ===== Rename Columns for Consistency =====
    print("\nRenaming columns for consistency...")
    for i in range(len(dfs)):
        dfs[i] = dfs[i].rename(columns={
            # click events
            'click_text': 'event_click_text',
            'click_text_length': 'event_click_text_length',
            'click_button': 'event_click_button',
            'click_pos_x': 'event_click_pos_x',
            'click_pos_y': 'event_click_pos_y',

            # change events
            'change_text': 'event_change_text',
            'change_text_length': 'event_change_text_length',
            'change_value': 'event_change_value',
            'change_value_length': 'event_change_value_length',
            'change_value_sim_session': 'event_change_value_sim_session',
            'change_value_sim_episode': 'event_change_value_sim_episode',

            # scroll and keypress
            'scroll_time': 'event_scroll_time',
            'num_scrolls': 'event_scroll_num',
            'num_key_presses': 'event_keypress_num',
        })

    # ===== Remove Datetime Column =====
    print("\nRemoving datetime column...")
    for i in range(len(dfs)):
        dfs[i] = dfs[i].drop(columns=['datetime'], errors='ignore')

    # ===== Rename Bitalino and Face Columns =====
    print("\nRenaming sensor columns...")
    df = dfs[0]  # For the remaining steps, we work with a single DataFrame

    # Rename bitalino and face columns
    new_column_names = {}
    for c in df.columns.tolist():
        # bitalino features
        if c.startswith('bit_'):
            if c.endswith('_tabchange'):
                new_c = 'episode_' + c.replace('_tabchange', '')
                new_column_names[c] = new_c
            elif c.endswith('_event'):
                new_c = 'event_' + c.replace('_event', '')
                new_column_names[c] = new_c

        # face features
        if c.startswith('face_'):
            if c.endswith('_tabchange'):
                new_c = 'episode_' + c.replace('_tabchange', '')
                new_column_names[c] = new_c
            elif c.endswith('_event'):
                new_c = 'event_' + c.replace('_event', '')
                new_column_names[c] = new_c

    if new_column_names:
        df = df.rename(columns=new_column_names)

    # ===== Define Column Order =====
    print("\nSetting column order...")
    column_order = [
        'dirname',
        'network',
        'username',
        'pid',
        'eid',
        'rid',
        'srid',

        'event',

        'task_id',

        'unix_time',
        'time',

        'url',
        'dom_object',
        'xpath',

        'labels_0',
        'labels_1',
        'labels_2',
        'labels_union',
        'labels_inter',
        'labels_valid',
        'finished_task',

        'event_duration',
        'event_cum_duration',

        'episode_click_duration',
        'episode_change_duration',
        'episode_scroll_duration',
        'episode_tabchange_duration',
        'episode_total_duration',

        'episode_click_num',
        'episode_change_num',
        'episode_scroll_num',
        'episode_tabchange_num',
        'episode_keypress_num',
        'episode_total_events',

        'event_click_text',
        'event_click_text_length',
        'event_click_button',
        'event_click_pos_x',
        'event_click_pos_y',

        'event_change_text',
        'event_change_text_length',
        'event_change_value',
        'event_change_value_sim_session',
        'event_change_value_sim_episode',
        'event_change_value_length',

        'event_keypress_num',

        'event_scroll_time',
        'event_scroll_num',
    ]

    # Add sensor columns to order if they exist
    sensor_columns = [c for c in df.columns if c.startswith('episode_bit_') or
                      c.startswith('event_bit_') or
                      c.startswith('episode_face_') or
                      c.startswith('event_face_') or
                      c == 'sam_valence' or
                      c == 'sam_arousal']

    column_order.extend(sorted(sensor_columns))

    # Select columns that exist in the DataFrame and in the specified order
    final_columns = [c for c in column_order if c in df.columns]
    df = df[final_columns]

    # Generate relative task id (rtid)
    df['rtid'] = df['task_id'] / df['task_id'].max()

    # Reset index
    df_all = df.reset_index(drop=True)

    # Generate per min features
    # Avoid division by zero by ensuring episode_total_duration > 0
    df_all['episode_tabchange_num_per_min'] = df_all['episode_tabchange_num'] / (df_all['episode_total_duration'] / 60).replace(0, 1)
    df_all['episode_change_num_per_min'] = df_all['episode_change_num'] / (df_all['episode_total_duration'] / 60).replace(0, 1)
    df_all['episode_click_num_per_min'] = df_all['episode_click_num'] / (df_all['episode_total_duration'] / 60).replace(0, 1)
    df_all['episode_scroll_num_per_min'] = df_all['episode_scroll_num'] / (df_all['episode_total_duration'] / 60).replace(0, 1)
    df_all['episode_keypress_num_per_min'] = df_all['episode_keypress_num'] / (df_all['episode_total_duration'] / 60).replace(0, 1)
    df_all['episode_total_events_per_min'] = df_all['episode_total_events'] / (df_all['episode_total_duration'] / 60).replace(0, 1)

    # Generate HTML features
    df_all['xpath_depth'] = df_all['xpath'].str.count('>')
    df_all['xpath_div_count'] = df_all['xpath'].str.count('div')
    df_all['xpath_num_unique'] = df_all['xpath'].apply(lambda x: len(set(re.findall(r'\b\w+\b', x))))
    df_all['dom_object_prev'] = df_all['dom_object'].shift(1).fillna('none')
    df_all['dom_object_next'] = df_all['dom_object'].shift(-1).fillna('none')

    # List of columns to normalize with z-score
    zscore_columns = [
        # event durations
        'event_duration',
        'event_cum_duration',

        # scroll
        'event_scroll_time',
        'event_scroll_num',

        # keypress
        'event_keypress_num',

        # episode durations
        'episode_tabchange_duration',
        'episode_change_duration',
        'episode_click_duration',
        'episode_scroll_duration',
        'episode_total_duration',

        # episode counts
        'episode_tabchange_num',
        'episode_change_num',
        'episode_click_num',
        'episode_scroll_num',
        'episode_keypress_num',
        'episode_total_events',

        # num per min
        'episode_tabchange_num_per_min',
        'episode_change_num_per_min',
        'episode_click_num_per_min',
        'episode_scroll_num_per_min',
        'episode_keypress_num_per_min',
        'episode_total_events_per_min',

        # html features
        'xpath_depth',
        'xpath_div_count',
        'xpath_num_unique',

        # text lens
        'event_change_text_length',
        'event_change_value_length',
        'event_click_text_length'
    ]

    # Add face columns
    face_columns = [col for col in df_all.columns if 'face_avg' in col or 'face_entropy' in col]
    zscore_columns += face_columns

    # Group by 'network' and 'username' and apply normalization
    for col in zscore_columns:
        group_mean = df_all.groupby(['network', 'username'])[col].transform('mean')
        group_std = df_all.groupby(['network', 'username'])[col].transform('std').replace(0, 1)
        df_all[f'{col}_norm'] = (df_all[col] - group_mean) / group_std

    # List of columns to normalize with min-max normalization
    minmax_columns = [
        'event_click_pos_x',
        'event_click_pos_y',
    ]

    # Add all EEG and BVP columns to the list
    eeg_bvp_columns = [col for col in df_all.columns if ('bit_eeg' in col or 'bit_bvp' in col)]
    minmax_columns += eeg_bvp_columns

    # Normalize each feature within each group
    normalized_data = df_all.groupby(['network', 'username'])[minmax_columns].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)  # Avoid division by zero
    )

    # Rename the columns for normalized features
    normalized_data = normalized_data.add_suffix('_norm')

    # Concatenate the normalized data back to the original DataFrame
    df_all = pd.concat([df_all, normalized_data], axis=1)

    # Example of applying this function for 'episode_total_duration'
    df_all = generate_episode_level_global_duration_features(df_all, col_name='event_duration')
    df_all = generate_episode_level_global_duration_features(df_all, col_name='event_duration_norm')

    df_all = generate_episode_level_local_duration_features(df_all, col_name='episode_total_duration')
    df_all = generate_episode_level_local_duration_features(df_all, col_name='episode_total_events')
    df_all = generate_episode_level_local_duration_features(df_all, col_name='episode_total_duration_norm')
    df_all = generate_episode_level_local_duration_features(df_all, col_name='episode_total_events_norm')

    # ===== Save Prepared Dataset =====
    print("\nSaving prepared dataset...")

    # Drop the 'dirname' column
    df_all = df_all.drop(columns=['dirname'], errors='ignore')

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Save DataFrame to CSV
    csv_path = os.path.join(output_path, 'dataset_prepared.csv')
    df_all.to_csv(csv_path, sep='\t', index=False)
    print(f"Dataset successfully prepared and saved to: {csv_path}")

    pkl_path = os.path.join(output_path, 'df_all.pkl')
    df_all.to_pickle(pkl_path)
    print(f"Dataset successfully prepared and saved to: {pkl_path}")

    return df_all


# ============= SCRIPT ENTRY POINT =============

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prepare_dataset.py <input_path> [<output_path>]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    prepare_dataset(input_path, output_path)
