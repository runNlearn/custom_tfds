# Custom Dataset on TFDS 
These codes are used to generate *custom* tfds datasets.

## Prerequisites
- [tensorflow](https://www.tensorflow.org) (just `tensorflow-cpu` is okay)
- [tfds-nightly](https://github.com/tensorflow/datasets)
- [kaggle API](https://www.kaggle.com/docs/api)

## Usage
If you already have generated custom data on GCS or your local path, just use 
`tfds.load(NAME_OF_DATA, data_dir=YOUR_PATH)`.  
  
#### Build from the ground
First, make your own token for kaggle API, check the details on [here](https://www.kaggle.com/docs/api),
and add a file of your token to `~/.kaggle/kaggle.json`.

To download and generate tfrecords, go to the corresponding directory of the 
dataset, and use `TFDS CLI`, for example:

```console
  $ cd custom_tfds/kaggle_hubmap_train
  $ tfds build
```

```python

  train_ds = tfds.load('kaggle_hubmap_train/256x256',
                       split='train')  # you can use `as_supervised=True`
  train_ds = (train_ds.map(...).batch(...)
              ...)
  for data in train_ds:
    ...
```
