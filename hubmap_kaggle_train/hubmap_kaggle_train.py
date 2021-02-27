"""hubmap_kaggle_train dataset."""
import os
from glob import glob

import tensorflow_datasets as tfds

_DESCRIPTION = """
This dataset is built on the data processed by `iafoss`.
Available shapes of images are 256, 512, and 1024.
"""

# TODO(hubmap_kaggle_train): BibTeX citation
_CITATION = """
"""

_TRAIN_DATA_REF_PREFIX = 'iafoss/hubmap'


class HubmapKaggleTrainConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Hubmap Kaggle."""

  def __init__(self, size, **kwargs):
    super(HubmapKaggleTrainConfig, self).__init__(
      version=tfds.core.Version('1.0.0'), **kwargs)
    self.size = size


def _make_builder_configs():
  """Return BuilderConfigs."""
  configs = []
  for size in [256, 512, 1024]:
    configs.append(
      HubmapKaggleTrainConfig(
        name='%dx%d' % (size, size),
        size=size,
        description=f"Training images cropped to {size}x{size}"
      ),
    )
  return configs


class HubmapKaggleTrain(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hubmap_kaggle_train dataset."""

  VERSION = tfds.core.Version('1.0.0')
  BUILDER_CONFIGS = _make_builder_configs()
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    size = self.builder_config.size
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(
                      shape=(size, size, 3), encoding_format='png'),
            'mask': tfds.features.Image(
                      shape=(size, size, 1), encoding_format='png'),
            'id': tfds.features.Text(),
        }),
        supervised_keys=('image', 'mask'),  # Set to `None` to disable
        homepage='https://www.kaggle.com/iafoss/256x256-images',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    size = self.builder_config.size
    train_path = dl_manager.download_kaggle_data(
                    _TRAIN_DATA_REF_PREFIX + f'-{size}x{size}')

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
              'path': train_path,
            },
        ),
    ]


  def _generate_examples(self, path):
    """Yields examples."""
    def _get_fname(path):
      basename = os.path.basename(path)
      fname = basename.split('.')[0]
      return fname

    image_paths = os.path.join(path, 'train', '*.png')
    image_paths = glob(image_paths)
    mask_paths = [path.replace('train', 'masks') for path in image_paths]

    n = len(image_paths)
    for i in range(n):
      image = open(image_paths[i], 'rb')
      mask = open(mask_paths[i], 'rb') 
      fname = _get_fname(image_paths[i])
      record = {
        'id': fname,
        'image': image,
        'mask': mask,
      }
      yield fname, record
