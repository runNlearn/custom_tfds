"""kaggle_hubmap_test dataset."""

import os
from tensorflow.io.gfile import glob
from tensorflow.io.gfile import GFile
import tensorflow_datasets as tfds

# TODO(kaggle_hubmap_test): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Kaggle Competition: HuBMAP Test Data.
Images are so large, user need to split them to small patches.
Available sizes of patches: 256
"""

# TODO(kaggle_hubmap_test): BibTeX citation
_CITATION = """
"""

_DATASET_ID_PREFIX = 'dartwin/hubmap-test'

_SIZE_LIST = [256]
_TYPE_LIST = [
    '1_raw',
]
_VERSION = '1.0.0'


class KaggleHubmapTestConfig(tfds.core.BuilderConfig):
  """BuilderConfig for KaggleHubmapTest."""
  def __init__(self, size, kaggle_data_version, **kwargs):
    super(KaggleHubmapTestConfig, self).__init__(**kwargs)
    self.size = size
    self.kaggle_data_version = kaggle_data_version


class KaggleHubmapTest(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for kaggle_hubmap_test dataset."""

  BUILDER_CONFIGS = [
    KaggleHubmapTestConfig(
        size=size,
        kaggle_data_version=int(type_.split('_')[0]),
        name=f'{size}x{size}_{type_[2:]}',
        version=_VERSION,
        description=type_[2:],
    ) for size in _SIZE_LIST
    for type_ in _TYPE_LIST
  ]

  RELEASE_NOTES = {
    '1.0.0': 'initial version'
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
        supervised_keys=('image', 'mask'),  # Set to `None` to disable
        homepage='https://www.kaggle.com/dartwin/hubmap-test-256x256',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    size = self.builder_config.size
    version = self.builder_config.kaggle_data_version
    kaggle_data_path = dl_manager.download_kaggle_data(
            _DATASET_ID_PREFIX + f'-{size:d}x{size:d}/version/{version}')
    return  [ 
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'path': kaggle_data_path,
            },
        ),
    ] 

  def _generate_examples(self, path):
    """Yields examples."""
    def get_fname(path):
      basename = os.path.basename(path)
      fname = basename.split('.')[0]
      return fname
    
    image_paths = os.path.join(path, '*.png')
    image_paths = glob(image_paths)

    for img_path in image_paths:
      image = GFile(img_path, 'rb')
      fname = get_fname(img_path)
      record = {
        'image': image,
        'name': fname,
      }
      yield fname, record
