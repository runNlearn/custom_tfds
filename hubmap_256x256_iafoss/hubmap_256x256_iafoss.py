import os
import io
from glob import glob

import numpy as np
import tensorflow_datasets as tfds

# TODO(hubmap_256x256_iafoss): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
# TFDS for HuBMAP dataset made by `iafoss` and `joshi98kishan`.
Train: 256x256
Test: 256x256
"""

# TODO(hubmap_256x256_iafoss): BibTeX citation
_CITATION = """
"""

_KAGGLE_TRAIN_DATA_NAME = 'iafoss/hubmap-256x256'
_KAGGLE_TEST_DATA_NAME = 'joshi98kishan/hubmap-256x256-test-data'


class Hubmap256x256Iafoss(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hubmap_256x256_iafoss dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(hubmap_256x256_iafoss): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          "image": tfds.features.Image(shape=(256, 256, 3)),
#          "mask": tfds.features.Image(shape=(256, 256, 1))
            # These are the features of your dataset like images, labels ...
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=("image", "mask"),  # e.g. ('image', 'label')
        homepage='https://www.kaggle.com/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    train_path = dl_manager.download_kaggle_data(_KAGGLE_TRAIN_DATA_NAME)
    test_path = dl_manager.download_kaggle_data(_KAGGLE_TEST_DATA_NAME)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
              'path': train_path,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
              'path': test_path,
            },
        ),
    ]

  def _get_fname(self, path):
    basename = os.path.basename(path)
    fname = basename.split(os.path.sep)[0]
    return fname

  def _generate_examples(self, path):
    """Yields examples."""
    cv2 = tfds.core.lazy_imports.cv2
    def _decode(path):
      img = cv2.imread(path, cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      return img

    train_base = _KAGGLE_TRAIN_DATA_NAME.split('/')[0]
    test_base = _KAGGLE_TEST_DATA_NAME.split('/')[0]
    basename = os.path.basename(path)
    base = basename.split('_')[0]
    assert base in [train_base, test_base]
    if base == train_base:
      images_path = os.path.join(path, 'train', '*.png')
      images_path = glob(images_path)
      masks_path = [path.replace('train', 'masks') for path in images_path]
    else:
      images_path = os.path.join(path, '*.png')
      images_path = glob(images_path)
      masks_path = None

    n = len(images_path)
    for i in range(n):
      record = dict()

      img = _decode(images_path[i])
      record['image'] = img

#      if masks_path:
#        mask = _decode(masks_path[i])
#        record['mask'] = mask
#      else:
#        record['mask'] = np.zeros((256, 256, 1), dtype=np.uint8)
      fname = self._get_fname(images_path[i])

      yield fname, record

