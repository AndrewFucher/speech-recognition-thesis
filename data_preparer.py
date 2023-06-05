import tarfile
from urllib import request

import os.path
import pathlib

import os
import copy
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import tensorflow as tf

from transformers import Transformer, LabelIndexer, LabelPadding, MfccPadding
from preprocessors import AudioReader

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")

LIBRISPEECH_DATASET_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
ARCHIVE_PATH = "./data/dev-clean.tar.gz"
EXTRACTED_DATA_PATH = "./data/speech"
VOCAB = "abcdefghijklmnopqrstuvwxyz' "
DATA_PROVIDER_PICKLE_PATH = "./data/pickle/data_provider.pkl"

class DataProvider(tf.keras.utils.Sequence):
    def __init__(
        self, 
        dataset: typing.Union[str, list, pd.DataFrame],
        data_preprocessors: typing.List[typing.Callable] = None,
        batch_size: int = 4,
        shuffle: bool = True,
        initial_epoch: int = 1,
        transformers: typing.List[Transformer] = None,
        skip_validation: bool = True,
        limit: int = None,
        use_cache: bool = False,
        log_level: int = logging.INFO,
        ) -> None:
        """ Standardised object for providing data to a model while training.
        Attributes:
            dataset (str, list, pd.DataFrame): Path to dataset, list of data or pandas dataframe of data.
            data_preprocessors (list): List of data preprocessors. (e.g. [read image, read audio, etc.])
            batch_size (int): The number of samples to include in each batch. Defaults to 4.
            shuffle (bool): Whether to shuffle the data. Defaults to True.
            initial_epoch (int): The initial epoch. Defaults to 1.
            transformers (list, optional): List of transformer functions. Defaults to None.
            skip_validation (bool, optional): Whether to skip validation. Defaults to True.
            limit (int, optional): Limit the number of samples in the dataset. Defaults to None.
            use_cache (bool, optional): Whether to cache the dataset. Defaults to False.
            log_level (int, optional): The log level. Defaults to logging.INFO.
        """
        self._dataset = dataset
        self._data_preprocessors = [] if data_preprocessors is None else data_preprocessors
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._epoch = initial_epoch
        self._transformers = [] if transformers is None else transformers
        self._skip_validation = skip_validation
        self._limit = limit
        self._use_cache = use_cache
        self._step = 0
        self._cache = {}
        self._on_epoch_end_remove = []

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Validate dataset
        if not skip_validation:
            self._dataset = self.validate(dataset, skip_validation, limit)
        else:
            self.logger.info("Skipping Dataset validation...")

        if limit:
            self.logger.info(f"Limiting dataset to {limit} samples.")
            self._dataset = self._dataset[:limit]

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.ceil(len(self._dataset) / self._batch_size))

    @property
    def transformers(self) -> typing.List[Transformer]:
        """ Return transformers """
        return self._transformers

    @transformers.setter
    def transformers(self, transformers: typing.List[Transformer]):
        """ Decorator for adding transformers to the DataProvider """
        for transformer in transformers:
            if isinstance(transformer, Transformer):
                if self._transformers is not None:
                    self._transformers.append(transformer)
                else:
                    self._transformers = [transformer]

            else:
                self.logger.warning(f"Transformer {transformer} is not an instance of Transformer.")

        return self._transformers
    
    @property
    def data_preprocessors(self) -> typing.List[typing.Callable]:
        """ Return data preprocessors """
        return self._data_preprocessors

    @data_preprocessors.setter
    def data_preprocessors(self, data_preprocessors: typing.List[typing.Callable]):
        """ Decorator for adding data preprocessors to the DataProvider """
        for data_preprocessor in data_preprocessors:
            if isinstance(data_preprocessor, typing.Callable):
                if self._data_preprocessors is not None:
                    self._data_preprocessors.append(data_preprocessor)
                else:
                    self._data_preprocessors = [data_preprocessor]

            else:
                self.logger.warning(f"Transformer {data_preprocessor} is not an instance of Transformer.")

        return self._data_preprocessors

    @property
    def epoch(self) -> int:
        """ Return Current Epoch"""
        return self._epoch

    @property
    def step(self) -> int:
        """ Return Current Step"""
        return self._step

    def on_epoch_end(self):
        """ Shuffle training dataset and increment epoch counter at the end of each epoch. """
        self._epoch += 1
        if self._shuffle:
            np.random.shuffle(self._dataset)

        # Remove any samples that were marked for removal
        for remove in self._on_epoch_end_remove:
            self.logger.warn(f"Removing {remove} from dataset.")
            self._dataset.remove(remove)
        self._on_epoch_end_remove = []

    def validate_list_dataset(self, dataset: list, skip_validation: bool = False) -> list:
        """ Validate a list dataset """
        validated_data = [data for data in tqdm(dataset, desc="Validating Dataset") if os.path.exists(data[0])]
        if not validated_data:
            raise FileNotFoundError("No valid data found in dataset.")

        return validated_data

    def validate(self, dataset: typing.Union[str, list, pd.DataFrame], skip_validation: bool) -> list:
        """ Validate the dataset and return the dataset """

        if isinstance(dataset, str):
            if os.path.exists(dataset):
                return dataset
        elif isinstance(dataset, list):
            return self.validate_list_dataset(dataset, skip_validation)
        elif isinstance(dataset, pd.DataFrame):
            return self.validate_list_dataset(dataset.values.tolist(), skip_validation)
        else:
            raise TypeError("Dataset must be a path, list or pandas dataframe.")

    def split(self, split: float = 0.9, shuffle: bool = True) -> typing.Tuple[typing.Any, typing.Any]:
        """ Split current data provider into training and validation data providers. 
        
        Args:
            split (float, optional): The split ratio. Defaults to 0.9.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        Returns:
            train_data_provider (tf.keras.utils.Sequence): The training data provider.
            val_data_provider (tf.keras.utils.Sequence): The validation data provider.
        """
        if shuffle:
            np.random.shuffle(self._dataset)
            
        train_data_provider, val_data_provider = copy.deepcopy(self), copy.deepcopy(self)
        train_data_provider._dataset = self._dataset[:int(len(self._dataset) * split)]
        val_data_provider._dataset = self._dataset[int(len(self._dataset) * split):]

        return train_data_provider, val_data_provider

    def to_csv(self, path: str, index: bool=False) -> None:
        """ Save the dataset to a csv file 
        Args:
            path (str): The path to save the csv file.
            index (bool, optional): Whether to save the index. Defaults to False.
        """
        df = pd.DataFrame(self._dataset)
        df.to_csv(path, index=index)

    def get_batch_annotations(self, index: int) -> typing.List:
        """ Returns a batch of annotations by batch index in the dataset
        Args:
            index (int): The index of the batch in 
        Returns:
            batch_annotations (list): A list of batch annotations
        """
        self._step = index
        start_index = index * self._batch_size

        # Get batch indexes
        batch_indexes = [i for i in range(start_index, start_index + self._batch_size) if i < len(self._dataset)]

        # Read batch data
        batch_annotations = [self._dataset[index] for index in batch_indexes]

        return batch_annotations
    
    def __iter__(self):
        """ Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def process_data(self, batch_data):
        """ Process data batch of data """
        if self._use_cache and batch_data[0] in self._cache:
            data, annotation = copy.deepcopy(self._cache[batch_data[0]])
        else:
            data, annotation = batch_data
            for preprocessor in self._data_preprocessors:
                data, annotation = preprocessor(data, annotation)
            
            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, marking for removal on epoch end.")
                self._on_epoch_end_remove.append(batch_data)
                return None, None
            
            if self._use_cache and batch_data[0] not in self._cache:
                self._cache[batch_data[0]] = (copy.deepcopy(data), copy.deepcopy(annotation))

        # Then transform and postprocess the batch data
        for objects in [self._transformers]:
            for obj in objects:
                data, annotation = obj(data, annotation)

        # Convert to numpy array if not already
        if not isinstance(data, np.ndarray):
            data = data.numpy()

        # Convert to numpy array if not already
        # TODO: This is a hack, need to fix this
        if not isinstance(annotation, (np.ndarray, int, float, str, np.uint8, float)):
            annotation = annotation.numpy()

        return data, annotation

    def __getitem__(self, index: int):
        """ Returns a batch of data by batch index"""
        dataset_batch = self.get_batch_annotations(index)
        
        # First read and preprocess the batch data
        batch_data, batch_annotations = [], []
        for index, batch in enumerate(dataset_batch):

            data, annotation = self.process_data(batch)

            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, skipping.")
                continue

            batch_data.append(data.astype(object))
            batch_annotations.append(annotation.astype(object))

        # from IPython import embed
        # embed()
        return [np.array(batch_data, dtype=np.float32), np.array(batch_annotations, dtype=np.int_)], np.array(batch_annotations, dtype=np.int_)

    
def getDataProvider(n_mfcc: int = 13, load_from_pickle: bool = True) -> typing.Tuple[DataProvider, typing.Tuple]:
    if load_from_pickle:
        if not os.path.isfile(DATA_PROVIDER_PICKLE_PATH):
            print("Pickle does not exist")
        else:
            with open(DATA_PROVIDER_PICKLE_PATH, "rb") as pckl:
                data_provider, input_shape = pickle.load(pckl)
                return data_provider, input_shape
    dataset = getDataset()
    max_mfcc_length, max_text_length, max_mfcc_length = 0, 0, 0
    input_shape = [None]
    for file_path, label in tqdm(dataset):
        spectrogram = AudioReader.get_mfcc(file_path, n_mfcc=n_mfcc)
        valid_label = [c for c in label.lower() if c in VOCAB]
        max_text_length = max(max_text_length, len(valid_label))
        max_mfcc_length = max(max_mfcc_length, spectrogram.shape[1])
    input_shape = [n_mfcc, max_mfcc_length]
    # from IPython import embed
    # embed()
    data_provider = DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=8,
        data_preprocessors=[
            AudioReader(),
            ],
        transformers=[
            MfccPadding(max_mfcc_length=max_mfcc_length, padding_value=0),
            LabelIndexer(VOCAB),
            LabelPadding(max_word_length=max_text_length, padding_value=len(VOCAB)),
            ],
    )
    filePath = pathlib.Path(DATA_PROVIDER_PICKLE_PATH)
    filePath.parent.mkdir(exist_ok=True, parents=True)
    with open(filePath, "wb") as pckl:
        pickle.dump((data_provider, input_shape), pckl)
    return data_provider, input_shape

def getDataset() -> typing.List[typing.List[str]]:
    raw_dataset = getRawDataset()
    
    for i in range(len(raw_dataset)):
        raw_dataset[i] = [raw_dataset[i][0], raw_dataset[i][1].lower()]
        
    return raw_dataset
    
def getRawDataset() -> typing.List[typing.List[str]]:
    transcription_paths = list(pathlib.Path(".").rglob("*.trans.txt"))
    audios_paths = list(pathlib.Path(".").rglob("*.flac"))
    name_to_path_audios = {i.stem: str(i.absolute()) for i in audios_paths}

    all_transcriptions = ""
    for transcription_path in transcription_paths:
        with open(str(transcription_path.absolute()), 'r') as file:
            text = file.read()
            all_transcriptions += text

    dataset = []
    for record in list(filter(None, all_transcriptions.split('\n'))):
        audio_file_name, label = record.split(' ', 1)
        dataset.append([name_to_path_audios[audio_file_name], label])
    
    return dataset

def tryLoadData() -> None:
    if not os.path.isfile(ARCHIVE_PATH):
        request.urlretrieve(LIBRISPEECH_DATASET_URL, ARCHIVE_PATH)

    if not os.path.isdir(EXTRACTED_DATA_PATH):
        file = tarfile.open(ARCHIVE_PATH)
        file.extractall(EXTRACTED_DATA_PATH)
        file.close()