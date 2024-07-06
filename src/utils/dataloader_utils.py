import cv2
from imgaug import augmenters as iaa
import numpy as np
import os
from PIL import Image, ImageOps
import tensorflow as tf
import time
import traceback
import joblib

def preprocess_input(img):
    return (np.array(img).astype(np.float32) - 127.5) / 127.5

def inverse_preprocess_input(img):
    x = np.array(img) * 127.5 + 127.5
    x = np.minimum(255.0, np.maximum(0.0, x))
    x = x.astype(np.uint8)
    return x

class WholeSlideDataloader(tf.keras.utils.Sequence):
    def __init__(
        self, 
        dataset,
        augment,
        shuffle,
        num_classes,
        batch_size=1,
        snapshot_path=None,
        x_y_batch_path=None,
    ):
        self.dataset = dataset
        self.augment = augment
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.snapshot_path = snapshot_path
        self.x_y_batch_path = x_y_batch_path

        self._reinit_shuffle_list()

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        begin_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size

        x_batch = []
        y_batch = []
        for shuffle_list_idx in range(begin_idx, end_idx):
            dataset_idx = self.shuffle_list[shuffle_list_idx]
            loaded = False
            while not loaded:
                try:
                    img, label = self.dataset[dataset_idx]
                    loaded = True
                except Exception as e:
                    print(traceback.format_exc())
                    print(
                        "Error occurs while loading {} . Retry after 5 seconds.".format(
                            self.dataset.get_slide_path(dataset_idx)
                        )
                    )
                    time.sleep(5)

            if self.augment:
                if img.size < 4 * 1024 * 1024 * 1024:
                    img = _get_augmentor().augment_image(img)
                else:
                    # If the image is too large, imgaug will fail on affine transformation. Use PIL instead.
                    img = _get_augmentor_wo_affine().augment_image(img)
                    img = Image.fromarray(img)
                    angle = np.random.uniform(0.0, 360.0)
                    translate = tuple(np.random.randint(-220, 220, size=[2]))
                    img = img.rotate(angle, resample=Image.NEAREST, translate=translate, fillcolor=(255, 255, 255))
                    img = np.array(img)

            if self.snapshot_path != None:
                os.makedirs(self.snapshot_path, exist_ok=True)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    os.path.join(self.snapshot_path, "dataloader_snapshot.tiff"),
                    img_bgr,
                )

            img = preprocess_input(img)
            y = np.zeros(shape=[self.num_classes])
            y[label] = 1.0
            
            x_batch.append(img)
            y_batch.append(y)
            joblib.dump(np.array(x_batch),self.x_y_batch_path/'x_batch.joblib')
            joblib.dump(np.array(y_batch), self.x_y_batch_path/'y_batch.joblib')

        return np.array(x_batch), np.array(y_batch)

    def on_epoch_end(self):
        self._reinit_shuffle_list()

    def get_slide_path(self, idx):
        dataset_idx = self.shuffle_list[idx]
        return self.dataset.get_slide_path(dataset_idx)

    def get_y_true(self, idx):
        dataset_idx = self.shuffle_list[idx]
        return self.dataset.get_y_true(dataset_idx)

    def get_dataset_idx(self, idx):
        return self.shuffle_list[idx]

    def _reinit_shuffle_list(self):
        self.shuffle_list = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.shuffle_list)
        