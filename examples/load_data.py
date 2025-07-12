"""
Basic example of loading and exploring AMUSED dataset
"""

import pandas as pd
import h5py
import numpy as np
from pathlib import Path

def load_participant_data(participant_path):
    """
    Load all available data for a single participant
    
    Args:
        participant_path: Path to participant directory (e.g., "UsabilitySmellsDataset/SN_1/P01")
    
    Returns:
        dict: Dictionary containing available data modalities
    """
    participant_path = Path(participant_path)
    data = {}
    
    # Load behavioral data (always available)
    csv_path = participant_path / "data.csv"
    if csv_path.exists():
        data['behavioral'] = pd.read_csv(csv_path, sep='\t', index_col=0)
        print(f"Loaded behavioral data: {len(data['behavioral'])} events")
    
    # Load physiological data (if available)
    bit_path = participant_path / "bit.h5"
    if bit_path.exists():
        data['physiological'] = {}
        with h5py.File(bit_path, 'r') as f:
            for key in f.keys():
                item_data = {
                    'interval': f[key].attrs['interval'],
                    'level': f[key].attrs['level'],
                    'eeg_signal': np.array(f[key]['eeg_signal']),
                    'bvp_signal': np.array(f[key]['bvp_signal'])
                }
                data['physiological'][key] = item_data
        print(f"Loaded physiological data: {len(data['physiological'])} intervals")
    
    # Load facial emotion data (if available)
    face_path = participant_path / "face.h5"
    if face_path.exists():
        data['facial'] = {}
        with h5py.File(face_path, 'r') as f:
            for key in f.keys():
                item_data = {
                    'interval': f[key].attrs['interval'],
                    'level': f[key].attrs['level'],
                    'emotions': np.array(f[key]['emotions'])
                }
                data['facial'][key] = item_data
        print(f"Loaded facial emotion data: {len(data['facial'])} intervals")
    
    return data


def explore_behavioral_data(df):
    """
    Basic exploration of behavioral data
    """
    print("=== Behavioral Data Overview ===")
    print(f"Total events: {len(df)}")
    print(f"Number of episodes: {df['pid'].nunique()}")
    print(f"Event types: {df['event'].value_counts().to_dict()}")
    
    # Task completion analysis
    if 'finished_task' in df.columns:
        completion_rate = df.groupby('pid')['finished_task'].any().mean()
        print(f"Task completion rate: {completion_rate:.2%}")
    
    # Duration statistics
    if 'event_duration' in df.columns:
        print(f"Average event duration: {df['event_duration'].mean():.2f} seconds")
        print(f"Total session duration: {df['event_duration'].sum():.2f} seconds")


def analyze_physiological_patterns(physio_data):
    """
    Basic analysis of physiological data
    """
    if not physio_data:
        print("No physiological data available")
        return
    
    print("=== Physiological Data Analysis ===")
    
    # Analyze EEG patterns
    all_eeg_lengths = []
    all_bvp_lengths = []
    
    for key, item in physio_data.items():
        all_eeg_lengths.append(len(item['eeg_signal']))
        all_bvp_lengths.append(len(item['bvp_signal']))
    
    print(f"Average EEG signal length: {np.mean(all_eeg_lengths):.0f} samples")
    print(f"Average BVP signal length: {np.mean(all_bvp_lengths):.0f} samples")
    
    # Calculate basic statistics for first interval
    first_item = list(physio_data.values())[0]
    eeg_signal = first_item['eeg_signal']
    bvp_signal = first_item['bvp_signal']
    
    print(f"Sample EEG stats - Mean: {np.mean(eeg_signal):.3f}, Std: {np.std(eeg_signal):.3f}")
    print(f"Sample BVP stats - Mean: {np.mean(bvp_signal):.3f}, Std: {np.std(bvp_signal):.3f}")


def analyze_emotional_patterns(facial_data):
    """
    Basic analysis of facial emotion data
    """
    if not facial_data:
        print("No facial emotion data available")
        return
    
    print("=== Facial Emotion Analysis ===")
    
    emotion_names = ['anger', 'disgust', 'fear', 'enjoyment', 'contempt', 'sadness', 'surprise']
    all_emotions = []
    
    for key, item in facial_data.items():
        if len(item['emotions']) > 0:
            all_emotions.append(item['emotions'])
    
    if all_emotions:
        # Concatenate all emotion data
        combined_emotions = np.vstack(all_emotions)
        
        # Calculate average emotion probabilities
        avg_emotions = np.mean(combined_emotions, axis=0)
        
        print("Average emotion probabilities:")
        for i, emotion in enumerate(emotion_names):
            print(f"  {emotion}: {avg_emotions[i]:.3f}")
        
        # Most common emotion
        dominant_emotion = emotion_names[np.argmax(avg_emotions)]
        print(f"Dominant emotion: {dominant_emotion}")


if __name__ == "__main__":
    # Example usage
    participant_path = "UsabilitySmellsDataset/SN_1/P11"  # Participant with all modalities
    
    # Load data
    data = load_participant_data(participant_path)
    
    # Analyze each modality
    if 'behavioral' in data:
        explore_behavioral_data(data['behavioral'])
        print()
    
    if 'physiological' in data:
        analyze_physiological_patterns(data['physiological'])
        print()
    
    if 'facial' in data:
        analyze_emotional_patterns(data['facial'])
