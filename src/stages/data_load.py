import sys
from pathlib import Path
import joblib
src_path = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(src_path))

import argparse

from src.utils.model_utils import build_resnet_model
from src.utils.dataloader_utils import WholeSlideDataloader, MILDataloader
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
    mil_infer_batch_size=params.data_load.MIL_INFER_BATCH_SIZE
    mil_patch_size=params.data_load.MIL_PATCH_SIZE
    mil_use_em=params.data_load.MIL_USE_EM
    mil_k=params.data_load.MIL_K
    mil_skip_white=params.data_load.MIL_SKIP_WHITE
    
    input_shape=mil_patch_size + [params.data_load.INPUT_SIZE[2]]
    train_dataset = Dataset(
            csv_path=csv_path,
            slide_dir=slide_dir,
            slide_file_extension=slide_file_extension,
            target_size=target_size,
            resize_ratio=resize_ratio,
            slide_reader=slide_reader,
            snapshot_path=snapshot_path
        )
    
    model=build_resnet_model(input_shape, num_classes)
    
    train_dataloader = MILDataloader(
            dataset=train_dataset,
            augment=True,
            shuffle=True,
            num_classes=num_classes,
            batch_size=batch_size,
            snapshot_path=None,
            mil_model=model,
            mil_patch_size=mil_patch_size,
            mil_infer_batch_size=mil_infer_batch_size,
            mil_use_em=mil_use_em,
            mil_k=mil_k,
            mil_skip_white=mil_skip_white
        )
    return train_dataloader
def getitem__(dataloader: MILDataloader):
    x_y_batch_path=Path(params.data_load.X_Y_Batch_Path)
    x_y_batch_path.mkdir(exist_ok=True)
    
    x_batch, y_batch = dataloader[0]

    joblib.dump(x_batch,x_y_batch_path/'x_batch.joblib')
    joblib.dump(y_batch,x_y_batch_path/'y_batch.joblib')

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    params_path = args.config
    params = load_params(params_path)
    dataload = data_load(params)
    getitem__(dataloader=dataload)

    