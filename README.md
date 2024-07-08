# Whole-slide CNN Training Pipeline

### Hardware Requirements

Make sure the system contains adequate amount of main memory space (minimal: 256 GB, recommended: 512 GB) to prevent out-of-memory error.
For ones who would like to have a try with less concern about model accuracy, setting a lower resizing ratio and image size in configuration can drastically reduce memory consumption, friendly for limited computing resources.

### Packages

The codes are tested on the environment with Ubuntu 18.04 / CentOS 7.5, Python 3.7.3, cuda 10.0, cudnn 7.6 and Open MPI 4.0.1.
Some Python packages should be installed before running the scripts, including

- Tensorflow v1.x (tensorflow-gpu==1.15.3)
- Horovod (horovod==0.19.0)
- MPI for Python (mpi4py==3.0.3)
- OpenSlide 3.4.1 (https://github.com/openslide/openslide/releases/tag/v3.4.1)
- OpenSlide Python (openslide-python=1.1.1)
- Tensorflow Huge Model Support (our package)

Refer to requirements.txt for the full list.
The installation of these packages should take few minutes.

## Usage

### 1. Define Datasets

To initiate a training task, several CSV files, e.g. train.csv, val.csv and test.csv, should be prepared to define training, validation and testing datasets.

These CSV files should follow the format:
```
[slide_name_1],[class_id_1]
[slide_name_2],[class_id_2]
...
```
, where [slide_name_\*] specify the filename **without extension** of a slide image and [class_id_\*] is an integer indicating a slide-level label (e.g. 0 for normal, 1 for cancerous). 

The configuration files for our experiments are placed at data_configs/.

### 2. Set Up Training Configurations

Model hyper-parameters are set up in a YAML file. 


The following table describes each field in a train_config.
| Field                      | Description
| -------------------------- | ---------------------------------------------------------------------------------------------
| RESULT_DIR                 | Directory to store output stuffs, including model weights, testing results, etc.
| TRAIN_CSV_PATH             | CSV file defining the training dataset.
| VAL_CSV_PATH               | CSV file defining the validation dataset.
| TEST_CSV_PATH              | CSV file defining the testing dataset.
| SLIDE_DIR                  | Directory containing all the slide image files (can be soft links).
| SLIDE_FILE_EXTENSION       | File extension. (e.g. ".svs")
| SLIDE_READER               | Library to read slides. (default: `openslide`)
| RESIZE_RATIO               | Resize ratio for downsampling slide images.
| INPUT_SIZE                 | Size of model inputs in [height, width, channels]. Resized images are padded or cropped to the size. Try decreasing this field when main memory are limited.
| NUM_CLASSES                | Number of classes.
| BATCH_SIZE                 | Number of slides processed in each training iteration for each MPI worker. (default: 1)

The following fields are valid only when `USE_MIL`.
| Field                      | Description
| -------------------------- | ---------------------------------------------------------------------------------------------
| MIL_PATCH_SIZE             | Patch size of the MIL model in [height, width].
| MIL_INFER_BATCH_SIZE       | Batch size for MIL finding representative patches.
| MIL_USE_EM                 | Whether to use EM-MIL.
| MIL_K                      | Number of representative patches. (default: 1)
| MIL_SKIP_WHITE             | Whether to skip white patches. (default: `True`)
