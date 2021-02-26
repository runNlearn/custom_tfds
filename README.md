# Custom Dataset on TFDS 
These codes are used to generate *custom* tfds datasets.

## Prerequisites
- [tfds-nightly](https://github.com/tensorflow/datasets)
- [kaggle API](https://www.kaggle.com/docs/api)

## Usage
First, make your own new token for kaggle API, check the details on [here](https://www.kaggle.com/docs/api),
and add a file of your token to `~/.kaggle/kaggle.json`.

To download and generate tfrecords, go to the corresponding directory of dataset,
and use `TFDS CLI`, for example:

```console
  $ cd custom_tfds/hubmap_256x256_iafoss
  $ tfds build
```

