# A Multi-Stream Fusion Approach with One-Class Learning for Audio-Visual Deepfake Detection
---
This repository contains the official implementation of our paper, "A Multi-Stream Fusion Approach with One-Class Learning for Audio-Visual Deepfake Detection"


# Requirement 
---
python==3.9 \\
pytorch-lightning=1.7.7

# Data Preparation
---
Need to create a new csv file for our train, validation, and test datasets by running `add_meta_data.ipynb`
Need to run `create_pairs.ipynb` and `voice_converion.py` to create voice conversion fake videos which is part of our test set.
Need to update path variable before running `create_pairs.ipynb`.

# Run the training code 
---
Need to change DATA_ROOT and OUTPUT before running the `scripts/run.sh`


# Evaluation 
--- 
Need to change DATA_ROOT and OUTPUT before running the `scripts/eval_all.sh`
