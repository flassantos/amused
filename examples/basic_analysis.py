"""
Basic analysis examples for AMUSED dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_all_participants(dataset_path, network='SN_1'):
    """
    Load behavioral data for all participants in a network
    """
    dataset_path = Path(dataset_path)
    network_path = dataset_path / network
    
    all_data = []
    
    for participant_dir in network_path.iterdir():
        if participant_dir.is_dir():
            csv_path = participant_dir / "data.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, sep='\t', index_col=0)
                df['participant'] = participant_dir.name
                df['network'] = network
                all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Loaded data for {len(all_data)} participants from {network}")
        return combined_df
    else:
        print(f"No data found in {network_path}")
        return pd.DataFrame()


def analyze_event_patterns(df):
    """
    Analyze user interaction patterns
    """
    # Event type distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    event_counts = df['event'].value_counts()
    plt.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%')
    plt.title('Event Type Distribution')
    
    # Events per episode
    plt.subplot(1, 3, 2)
    events_per_episode = df.groupby(['participant', 'pid']).size()
    plt.hist(events_per_episode, bins=20, alpha=0.7)
    plt.xlabel('Events per Episode')
    plt.ylabel('Frequency')
    plt.title('Events per Episode Distribution')
    
    # Episode duration
    plt.subplot(1, 3, 3)
    episode_durations = df.groupby(['participant', 'pid'])['event_duration'].sum()
    plt.hist(episode_durations / 60, bins=20, alpha=0.7)  # Convert to minutes
    plt.xlabel('Episode Duration (minutes)')
    plt.ylabel('Frequency')
    plt.title('Episode Duration Distribution')
    
    plt.tight_layout()
    plt.show()


def analyze_usability_issues(df):
    """
    Analyze usability issues from expert annotations
    """
    if 'labels_0' not in df.columns:
        print("No usability annotations found")
        return
    
    # Extract labels
    def extract_labels(label_str):
        try:
            return eval(label_str) if isinstance(label_str, str) else []
        except:
            return []
    
    df['labels_parsed'] = df['labels_0'].apply(extract_labels)
    
    # Count label occurrences
    all_labels = []
    for labels in df['labels_parsed']:
        all_labels.extend(labels)
    
    if all_labels:
        label_counts = pd.Series(all_labels).value_counts()
        
        plt.figure(figsize=(10, 6))
        label_counts.plot(kind='bar')
        plt.title('Usability Issues Frequency')
        plt.xlabel('Usability Smell')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        return label_counts
    else:
        print("No usability labels found in the data")
        return pd.Series()


def compare_networks(dataset_path):
    """
    Compare interaction patterns across different social networks
    """
    networks = ['SN_1', 'SN_2', 'SN_3']
    network_data = {}
    
    for network in networks:
        df = load_all_participants(dataset_path, network)
        if not df.empty:
            network_data[network] = df
    
    if not network_data:
        print("No data loaded for comparison")
        return
    
    # Compare event types across networks
    plt.figure(figsize=(15, 5))
    
    for i, (network, df) in enumerate(network_data.items(), 1):
        plt.subplot(1, len(network_data), i)
        event_counts = df['event'].value_counts()
        plt.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%')
        plt.title(f'{network} Event Distribution')
    
    plt.tight_layout()
    plt.show()
    
    # Compare session statistics
    stats_data = []
    for network, df in network_data.items():
        participant_stats = df.groupby('participant').agg({
            'pid': 'nunique',  # Number of episodes
            'event_duration': 'sum',  # Total session time
            'eid': 'count'  # Total events
        }).rename(columns={'pid': 'episodes', 'event_duration': 'total_time', 'eid': 'total_events'})
        
        participant_stats['network'] = network
        stats_data.append(participant_stats)
    
    if stats_data:
        combined_stats = pd.concat(stats_data)
        
        # Box plots for comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        sns.boxplot(data=combined_stats, x='network', y='episodes', ax=axes[0])
        axes[0].set_title('Episodes per Participant')
        
        sns.boxplot(data=combined_stats, x='network', y='total_time', ax=axes[1])
        axes[1].set_title('Total Session Time (seconds)')
        
        sns.boxplot(data=combined_stats, x='network', y='total_events', ax=axes[2])
        axes[2].set_title('Total Events per Participant')
        
        plt.tight_layout()
        plt.show()
        
        return combined_stats


if __name__ == "__main__":
    # Example usage
    dataset_path = "UsabilitySmellsDataset"
    
    # Load data for one network
    df = load_all_participants(dataset_path, 'SN_1')
    
    if not df.empty:
        print("=== Event Pattern Analysis ===")
        analyze_event_patterns(df)
        
        print("\n=== Usability Issues Analysis ===")
        label_counts = analyze_usability_issues(df)
        
        print("\n=== Network Comparison ===")
        network_stats = compare_networks(dataset_path)
