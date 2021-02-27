"""hubmap_kaggle_test dataset."""

import os
from glob import glob 

import tensorflow_datasets as tfds

_DESCRIPTION = """
# TFDS for test set of HuBMAP dataset in `kaggle`.
This dataset is built on the data processed by `joshi98kishan`.
"""

# TODO(hubmap_kaggle_test): BibTeX citation
_CITATION = """
"""

_TEST_DATA_REF = 'joshi98kishan/hubmap-256x256-test-data'


class HubmapKaggleTest(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for hubmap_kaggle_test dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(
                shape=(256, 256, 3), encoding_format='png'),
            'id': tfds.features.Text(),
        }),
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://www.kaggle.com/joshi98kishan/hubmap-256x256-test-data',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    test_path = dl_manager.download_kaggle_data(_TEST_DATA_REF)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'path': test_path,
            }
        )
    ]

  def _generate_examples(self, path):
    """Yields examples."""
    def _get_fname(path):
      basename = os.path.basename(path)
      fname = basename.split('.')[0]
      return fname

    image_paths = os.path.join(path, '*.png')
    image_paths = glob(image_paths)

    n = len(image_paths)
    for i in range(n):
      image = open(image_paths[i], 'rb')
      fname = _get_fname(image_paths[i])
      record = {
        'id': fname,
        'image': image,
      }
      yield fname, record
