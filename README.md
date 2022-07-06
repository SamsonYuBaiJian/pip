# PIP: Physical Interaction Prediction via Mental Simulation with Span Selection

## Configurations
- **task_type**: Task to run experiments for, can be "contact", "contain", "stability" or "combined".
- **frame_path**: Path to frames.
- **mask_path**: Path to masks.
- **train_label_path**: Path to train label .json file.
- **val_label_path**: Path to validation label .json file.
- **test_label_path**: Path to test label .json file.
- **video_path**: Path to videos, used only for data processing.
- **data_path**: Path to original data, used only for data processing.
- **train_val_test_splits**: List of [train, validation, test] split, should sum up to 1, used only for data processing.
- **save_path**: Path to save information from experiment runs.
- **save_spans**: Set as True to save all spans selections across train/validation or test runs, for visualization of span distribution.
- **load_model_path**: Path to load model for testing or fine-tuning.
- **experiment_type**: Experiment run type, can be "train" or "test.
- **device**: Set device to "cpu" or "cuda".
- **model_type**: Set to "pip" for PIP, "ablation" for PIP w/o SS, "phydnet" for the modified PhyDNet model and "baseline" for baseline model.
- **num_epoch**: Number of epochs.
- **batch_size**: Batch size.
- **teacher_forcing_prob**: Teacher forcing probability, can be from 0 - 1.
- **first_n_frame_dynamics**: Number of initial frames to use, can be from 0 to maximum length of frame sequence.
- **frame_interval**: Frame interval during data loading.
- **learning_rate**: Learning rate.
- **max_seq_len**: Maximum sequence length to generate with PIP for the frames, set after frames have been divided by frame interval.
- **span_num**: Number of spans to extract.
- **seed**: Random seed.

## SPACE+ Dataset
### Data Processing
1. Set configurations in `config.yml`.
2. Run `python utils/process_data_<task>.py --config_file config.yml` to generate the .json files.
3. Run `utils/get_frames_from_video.py` to extract the frames.
4. Run `utils/generate_masks.py` to generate the masks.

### Combined Task Data Setup
Train
- Contact: 1 - 200 (200)
- Containment: 201 - 400 (200)
- Stability: 401 - 600 (200)

Validation
- Contact: 601 - 666 (66)
- Containment: 667 - 733 (67)
- Stability: 734 - 800 (67)

Test
- Contact: 801 - 866 (66)
- Containment: 867 - 933 (67)
- Stability: 934 - 1000 (67)

## Running the Code
1. Download pretrained 3D ResNet34 weights `r3d34_K_200ep.pth` from https://github.com/kenshohara/3D-ResNets-PyTorch and put the file in the root directory.
2. Set configurations in `config.yml`.
3. Run `python main.py --config_file config.yml`.


## PIP Visualizations
### Generation
A generated sequence will be automatically saved at the end of every epoch during training for both training and validation, and saved at the end during testing, inside the experiment run directory.

### Span Selection Distribution 
1. Set `save_spans` in `config.yml` to True.
2. Train or test PIP, and a file named `all_spans.json` will be produced in the experiment run directory.
3. Replace the path to the `all_spans.json` in `utils/visualize_span_distribution.py`.
4. Run `python utils/visualize_span_distribution.py`, and the plot will be saved as `span_distribution.png`.


## Acknowledgments
Parts of our code, specifically the files `constrain_moments.py`, `phydnet_main.py` and `phydnet_model.py`, are adapted from the code from the paper "Disentangling Physical Dynamics from Unknown Factors for UnsupervisedVideo Prediction" at https://github.com/vincent-leguen/PhyDNet. We would like to thank the authors for their well-organised code.