# AMUSED: Affective Metrics & Usability Smell Evaluation Dataset

AMUSED (Affective Metrics & Usability Smell Evaluation Dataset) is a comprehensive multimodal dataset designed to advance research in Human-Computer Interaction (HCI), usability evaluation, and emotion recognition. This dataset combines user interaction data with physiological signals and emotional responses to provide insights into how users experience usability issues.


Download AMUSED at Zenodo: https://zenodo.org/records/15870704


**Dataset Highlights**
- **70 participants** with equal gender distribution, ages 18-61
- **24 hours 46 minutes** of total recorded interaction time
- **20,050 user interaction events** (clicks, scrolls, URL changes, input changes, keypress)
- **Multiple data modalities** synchronized in time
- **Expert annotations** of 11 types of usability smells by 20 HCI experts
- **Three social networking platforms** with usability issues (prototypes)


### Dataset Structure

```
AMUSED.tar.gz
└── UsabilitySmellsDataset/
    ├── SN_1/          # Social Network 1 (32 participants: P01-P32)
    ├── SN_2/          # Social Network 2 (30 participants: L01-L30) 
    └── SN_3/          # Social Network 3 (8 participants: S01-S08)
```

Each participant directory contains:
- **`data.csv`**: Processed log and annotation features
- **`bit.h5`**: EEG and heart rate features (when available)
- **`face.h5`**: Facial emotion features (when available)


### Data Availability by Modality

| Modality | SN_1 | SN_2 | SN_3 | Total |
|----------|------|------|------|-------|
| Logs | 32 | 30 | 8 | **70** |
| EEG/BVP | 23 | 30 | 3 | **56** |
| Facial Emotions | 22 | 28 | 0 | **50** |
| SAM | 1 | 28 | 0 | **29** |


## Quick Start

### Using the Dataset

The easiest way to start working with AMUSED is through our example notebooks:

```bash
# Clone the repository
git clone https://github.com/your-username/amused.git
cd amused

# Install dependencies
pip install -r requirements.txt

# Explore the dataset
jupyter-lab notebooks/
```



### Loading the Data

```python
import pandas as pd
import h5py
import numpy as np
from pathlib import Path

# Load behavioral data for a participant
df = pd.read_csv('UsabilitySmellsDataset/SN_1/P01/data.csv', sep='\t', index_col=0)
print(f"Loaded {len(df)} events for participant P01")

# Load all participants from a network
def load_all_participants(dataset_path, network='SN_1'):
    network_path = Path(dataset_path) / network
    all_data = []
    
    for participant_dir in network_path.iterdir():
        if participant_dir.is_dir():
            csv_path = participant_dir / "data.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, sep='\t', index_col=0)
                df['participant'] = participant_dir.name
                df['network'] = network
                all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Load all SN_1 participants
df_all = load_all_participants('UsabilitySmellsDataset', 'SN_1')
print(f"Loaded data for {df_all['participant'].nunique()} participants")

# Load physiological data (if available)
def load_physiological_data(participant_path):
    bit_path = Path(participant_path) / "bit.h5"
    if not bit_path.exists():
        return None
    
    physiological_data = {}
    with h5py.File(bit_path, 'r') as f:
        for key in f.keys():
            physiological_data[key] = {
                'interval': f[key].attrs['interval'],
                'level': f[key].attrs['level'],  # 'event' or 'tabchange'
                'eeg_signal': np.array(f[key]['eeg_signal']),
                'bvp_signal': np.array(f[key]['bvp_signal'])
            }
    return physiological_data

# Load facial emotion data (if available)
def load_facial_data(participant_path):
    face_path = Path(participant_path) / "face.h5"
    if not face_path.exists():
        return None
    
    facial_data = {}
    with h5py.File(face_path, 'r') as f:
        for key in f.keys():
            facial_data[key] = {
                'interval': f[key].attrs['interval'],
                'level': f[key].attrs['level'],
                'emotions': np.array(f[key]['emotions'])  # Shape: (n_frames, 7_emotions)
            }
    return facial_data

# Example: Load multimodal data for a participant with all modalities
participant_path = 'UsabilitySmellsDataset/SN_1/P11'
behavioral_data = pd.read_csv(f'{participant_path}/data.csv', sep='\t', index_col=0)
physiological_data = load_physiological_data(participant_path)
facial_data = load_facial_data(participant_path)

print(f"Behavioral events: {len(behavioral_data)}")
print(f"Physiological intervals: {len(physiological_data) if physiological_data else 0}")
print(f"Facial emotion intervals: {len(facial_data) if facial_data else 0}")
```


## Dataset Features

See [DOCUMENTATION.md](/DOCUMENTATION.md).


## Examples and Notebooks


See [examples](/examples) for python script examples on how to load the dataset and analyze it.

We also provide [jupyter notebooks](/notebooks) for:

- Log (behavior) analysis: [analyze_final_dataset.ipynb](/notebooks/analyze_final_dataset.ipynb)
- EEG/Face analysis: [explore_eeg_and_face_features.ipynb](/notebooks/explore_eeg_and_face_features.ipynb)



## Data Collection Methodology


### Experimental Setup

Participants completed realistic web interaction tasks on three different social network platforms while their behavior and physiological responses were recorded. The three selected social networks are:

- SN_1: https://github.com/flassantos/perspective
- SN_2: https://github.com/flassantos/react-social-network
- SN_3: https://github.com/flassantos/social-network


### Instructions

For more informations and instructions on the data collection and annotation, check this [wiki page](https://github.com/flassantos/annotation-usability-smells/wiki) of the annotation tool.



## Data Processing Scripts

This repository contains the complete pipeline for processing the AMUSED dataset:

### Core Scripts

#### `decrypt_wildfire_log.py`
Decrypts interaction logs from the [Wildfire browser extension](https://chromewebstore.google.com/detail/wildfire/djhgeeodemlfdpmcccdekfalbhllcoim).

```bash
python decrypt_wildfire_log.py input.wfire output.json
```

#### `generate_labeled_dataset.py` 
Combines interaction logs with expert annotations to create labeled datasets.

```bash
python generate_labeled_dataset.py \
    --input-json log.json \
    --input-tasks annotations.json \
    --diff-seconds -24 \
    --output dataset.csv
```

Can be run without annotations for unlabeled data:
```bash
python generate_labeled_dataset.py \
    --input-json log.json \
    --no-annotations \
    --output dataset.csv
```

#### `generate_eeg_features.py`
Extracts EEG and heart rate features from BITalino sensor data.

```bash
python generate_eeg_features.py \
    --input-dir participant_directory \
    --video-start-time "10:30:45" \
    --diff-log -24
```

#### `generate_face_features.py`
Processes facial emotion recognition data and aligns with behavioral events.

```bash
python generate_face_features.py \
    --input-dir participant_directory \
    --diff-log -24 \
    --diff-face -10
```

#### `prepare_log_dataset.py`
Creates the final analysis-ready dataset by combining all the log features from all participants.

```bash
python prepare_log_dataset.py participant_directory [output_directory]
```



## Citation

If you use this repository or the dataset in your research, please cite us:

```bibtex
@article{amused2025,
  title={AMUSED: A Multi-Modal Dataset for Usability Smell Identification},
  author={Flavia de S. Santos, Marcos V. Treviso, Kamila R. H. Rodrigues, Renata P. M. Fortes, Sandra P. Gama},
  journal={IEEE Transactions on Affective Computing},
  year={2025},
  note={Dataset available at: https://github.com/flassantos/amused},
  doi={10.XXXX/abc123}
}
```


## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

