import sys
from pathlib import Path
import joblib
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

from src.utils.dataloader_utils import WholeSlideDataloader
from src.utils.dataset_utils import Dataset
from src.utils.load_params import load_params

def data_load(params):
    csv_path=Path(params.data_load.TRAIN_CSV_PATH)
    slide_dir=Path(params.data_load.SLIDE_DIR)
    slide_dir.mkdir(exist_ok=True)
    slide_file_extension=params.data_load.SLIDE_FILE_EXTENSION
    target_size=params.data_load.INPUT_SIZE[: 2]
    resize_ratio=params.data_load.RESIZE_RATIO
    slide_reader=params.data_load.SLIDE_READER
    snapshot_path=params.data_load.DEBUG_PATH
    num_classes=params.data_load.NUM_CLASSES
    batch_size=params.data_load.BATCH_SIZE
    
    train_dataset = Dataset(
            csv_path=csv_path,
            slide_dir=slide_dir,
            slide_file_extension=slide_file_extension,
            target_size=target_size,
            resize_ratio=resize_ratio,
            slide_reader=slide_reader,
            snapshot_path=snapshot_path
        )
    dataloader = WholeSlideDataloader(
                dataset=train_dataset,
                augment=True,
                shuffle=True,
                num_classes=num_classes,
                batch_size=batch_size,
                snapshot_path=snapshot_path
            )
    return dataloader


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    dataload = data_load(params)
    