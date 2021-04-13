# SPECIAL
## Running the Code
`python main.py --config_file config.yml`

## Data format
`[X, Y, Z, RotX, RotY, RotZ]`

## Data Processing
`python process_data.py --config_file config.yml`

## Tasks
- [X] Data processing
- [X] Build basic model
- [X] Add loss for generated images
- [X] Add skip connections
- [X] Vary teacher forcing
- [X] Add first few teacher forced frames for dynamics learning
- [X] Add train-val-test splits
- [X] Add temporal discriminator
- [X] Fix loss calculations
- [X] Add coordinates for object identification/tracking
- [X] Add accuracy
- [ ] Inference code