# SPECIAL
## Running the Code
`python main.py --config_file config.yml`

## Data Index
Train
Contact: 1-200 (200)
Containment:201-400 (200)
Stability:401-600 (200)

Val
Contact:601-666 (66)
Containment:667-733 (67)
Stability: 734-800 (67)

Test
Contact: 801-866 (66)
Containment: 867-933 (67)
Stability: 934-1000 (67)

## Data Processing
`python process_data.py --config_file config.yml`

## Tasks

- [X] Vary teacher forcing
- [X] Add first few teacher forced frames for dynamics learning
- [X] Add train-val-test splits
- [X] Fix loss calculations
- [X] Add mask for object identification/tracking
- [X] Add accuracy
- [ ] Inference code
- [X] Save model according to validation loss