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
`python process_data_<tasks>.py --config_file config.yml` to generate the .json

Run `ColorSegmentation.py` to generate the masks.

Run `VideoToFrames.py` to extract the frames.



