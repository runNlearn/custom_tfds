"""kaggle_hubmap_test dataset."""

import os
from glob import glob
import tensorflow_datasets as tfds

# TODO(kaggle_hubmap_train): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Kaggle Competition: HuBMAP Test Data.
Images are so large, user need to split them to small patches.
Available sizes of patches: 256
"""

# TODO(kaggle_hubmap_train): BibTeX citation
_CITATION = """
"""

DATASET_ID_PREFIX = 'dartwin/hubmap-test'


class KaggleHubmapTestConfig(tfds.core.BuilderConfig):
  """BuilderConfig for KaggleHubmapTest."""
  def __init__(self, size, **kwargs):
    super(KaggleHubmapTestConfig, self).__init__(
        version=tfds.core.Version('1.0.0'), **kwargs)
    self.size = size

def _make_builder_configs():
  """Return BuilderConfigs."""
  configs = []
  for size in [256]:
    configs.append(
      KaggleHubmapTestConfig(
        name=f'{size:d}x{size:d}',
        size=size,
        description=f'Splitted test images to {size}x{size} size'
      ),
    )
  return configs


class KaggleHubmapTest(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for kaggle_hubmap_test dataset."""

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
            'name': tfds.features.Text(),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://www.kaggle.com/dartwin/hubmap-test-256x256',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    size = self.builder_config.size
    test_path = dl_manager.download_kaggle_data(
            DATASET_ID_PREFIX + f'-{size:d}x{size:d}')
    return {
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'path': test_path,
            },
        ),
    }
  def _generate_examples(self, path):
    """Yields examples."""
    def get_fname(path):
      basename = os.path.basename(path)
      fname = basename.split('.')[0]
      return fname

    image_paths = os.path.join(path, 'image', '*.png')
    image_paths = glob(image_paths)

    for img_path in image_paths:
      image = open(img_path, 'rb')
      fname = get_fname(img_path)
      records = {
        'image': image,
        'name': fname,
      }
      yield fname, record
