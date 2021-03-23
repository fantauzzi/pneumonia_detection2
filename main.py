import re
from pathlib import Path
from collections import Counter
import pickle
import logging
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from kerastuner.tuners import Hyperband
import kerastuner as kt
from tqdm import tqdm

dataset_path = '/mnt/storage/datasets/chestxray'
dataset2_path = '/mnt/storage/datasets/chest_xray'
augmented_root = dataset_path + '/images_aug'
metadata_fpath = dataset_path + '/Data_Entry_2017.csv'
logs_root = 'logs'
seed = 42
p_test = .2
p_val = p_test / (1 - p_test)
AUTOTUNE = tf.data.AUTOTUNE
image_size = (224, 224)
input_shape = image_size + (3,)
batch_size = 16
val_batch_size = 256
n_classes = 1
patience = 10
epochs_base = 2
epochs_ft = 2
augmentation = 4
pos_to_neg_rate_on_test = 1.
n_folds = 5


def split_dataset2(metadata):
    split1_metadata, split2_metadata = train_test_split(metadata,
                                                        test_size=.5,
                                                        random_state=seed,
                                                        stratify=metadata['Pneumonia'])

    return split1_metadata, split2_metadata


def shuffle_DataFrame(df, seed=None):
    df = df.sample(n=len(df), replace=False, random_state=seed)
    return df


def keep_n_negative_samples(dataframe, n_wanted):
    ''' Remove enough negative samples, choosen at random, from the dataset, in order to obtain a dataset with the
    requested number of negative samples'''
    assert int(n_wanted) == n_wanted
    # Compile a dataframe with all the negative samples (of the train set)
    non_pneumonia = dataframe[dataframe['Pneumonia'] == 0].copy()  # Make a copy, don't take a copy of a slice
    # Choose enough of them, stratifying over the fact that the samples have other findings, different from pneumonia
    non_pneumonia['Other Findings'] = non_pneumonia['Finding Labels'] != 'No Finding'
    non_pneumonia_sampled, _ = train_test_split(non_pneumonia,
                                                train_size=int(n_wanted),
                                                random_state=seed,
                                                stratify=non_pneumonia['Other Findings'])
    pneumonia = dataframe[dataframe['Pneumonia'] == 1]
    result = pd.concat([pneumonia, non_pneumonia_sampled], axis=0, ignore_index=True)
    # The 'Other Findings' column is not needed anymore
    result.drop('Other Findings', axis=1, inplace=True)
    # Shuffle the rows of the result
    result = shuffle_DataFrame(result, seed)

    assert sum(result['Pneumonia'] == 0) == n_wanted
    assert sum(result['Pneumonia']) == sum(dataframe['Pneumonia'])

    return result


def split_dataset(metadata, neg_to_pos_ratio):
    ''' Partition the metadata between training, validation and test set, ensuring that all samples for the same
        patient belong to the same set. Samples relative to the same patient must not be split across different sets '''
    metadata['Pneumonia'] = metadata['Finding Labels'].apply(lambda labels: 1 if 'Pneumonia' in labels else 0)
    ''' Find which patients have no x-ray with pneumonia, which have some (but not all) x-rays with pneumonia, and which 
    have only x-rays with pneumonia, and stratify the split based on that information '''
    patients_pneumonia = metadata.groupby(by='Patient ID').mean()['Pneumonia']  # The index is the patient ID
    ''' Assign 0 to patients that have no x-rays with pneumonia, 1 to those that have some x-rays with pneumonia, 2 to
    those that have only x-rays with pneumonia '''
    patients_pneumonia = patients_pneumonia.apply(lambda the_mean: 0 if the_mean == 0 else (2 if the_mean == 1 else 1))
    # Do the stratified splits
    dev_patients, test_patients = train_test_split(patients_pneumonia,
                                                   test_size=p_test,
                                                   random_state=seed,
                                                   stratify=patients_pneumonia)
    train_patients, val_patients = train_test_split(dev_patients,
                                                    test_size=p_val,
                                                    random_state=seed,
                                                    stratify=dev_patients)
    # The patient IDs are still in the indices of the respective series
    train_metadata = metadata[metadata['Patient ID'].isin(train_patients.index)]
    val_metadata = metadata[metadata['Patient ID'].isin(val_patients.index)]
    test_metadata = metadata[metadata['Patient ID'].isin(test_patients.index)]
    assert len(train_metadata) + len(val_metadata) + len(test_metadata) == len(metadata)
    assert sum(train_metadata['Pneumonia']) + sum(val_metadata['Pneumonia']) + sum(test_metadata['Pneumonia']) == sum(
        metadata['Pneumonia'])
    assert set(train_metadata['Patient ID']).isdisjoint(set(val_metadata['Patient ID']))
    assert set(train_metadata['Patient ID']).isdisjoint(set(test_metadata['Patient ID']))
    assert set(val_metadata['Patient ID']).isdisjoint(set(test_metadata['Patient ID']))
    # The number of positive samples currently in the train set
    train_n_pos = np.sum(train_metadata['Pneumonia'])
    # The number of negative samples wanted for the train set
    wanted_neg = min(train_n_pos * neg_to_pos_ratio, len(train_metadata) - train_n_pos)

    train_metadata = keep_n_negative_samples(train_metadata, wanted_neg)
    val_metadata = keep_n_negative_samples(val_metadata, sum(val_metadata['Pneumonia']) / pos_to_neg_rate_on_test)
    test_metadata = keep_n_negative_samples(test_metadata, sum(test_metadata['Pneumonia']) / pos_to_neg_rate_on_test)

    return train_metadata, val_metadata, test_metadata


def load_image(filepath, y):
    image = tf.io.read_file(filepath)
    image = tf.io.decode_png(image, channels=3)
    return image, y


def load_and_resize_image(filepath, y):
    image, y = load_image(filepath, y)
    image = tf.image.resize(image, size=image_size)
    image = tf.math.round(image)
    image = tf.cast(image, dtype=tf.uint8)
    return image, y


def make_pipeline(file_paths, y, batch_size, shuffle, seed=None):
    assert shuffle or seed is None
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=seed)
    dataset = dataset.map(load_and_resize_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
    if not shuffle:
        dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def compile_model(model, learning_rate):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])


def make_model_DenseNet121(hp):
    learning_rate = hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
    base_model = tf.keras.applications.DenseNet121(include_top=False,
                                                   weights='imagenet',
                                                   input_shape=input_shape,
                                                   pooling=None,
                                                   classes=n_classes)
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    compile_model(model, learning_rate)
    return model


def make_model_Resnet50(hp):
    learning_rate = hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                                  weights='imagenet',
                                                  input_shape=input_shape,
                                                  pooling=None,
                                                  classes=n_classes)
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    compile_model(model, learning_rate)
    return model


class ModelMaker:
    def __init__(self, tuner):
        self.tuner = tuner

    def make_model_Resnet50(self, hp):
        learning_rate = hp.Float('learning_rate', 1e-5, 1e-4, sampling='log')

        model = self.tuner.get_best_models(num_models=1)[0]

        base_model = model.layers[3]
        assert base_model.name == 'resnet50v2'
        base_model.trainable = True

        pattern = re.compile('conv(\d)_block(\d)_out')
        block_ends = []
        block_end_names = []
        for i, layer in enumerate(base_model.layers):
            match = pattern.match(layer.name)
            if match is not None:
                block_ends.append(i)
                block_end_names.append(layer.name)
        last_frozen_layer = hp.Int('last_frozen_layer', 0, len(block_ends) - 1)

        for i in range(block_ends[last_frozen_layer] + 1):
            base_model.layers[i].trainable = False

        compile_model(model, learning_rate)
        return model

    def make_model_DenseNet121(self, hp):
        learning_rate = hp['learning_rate'] if isinstance(hp, dict) else hp.Float('learning_rate',
                                                                                  1e-5,
                                                                                  1e-4,
                                                                                  sampling='log')

        model = self.tuner.get_best_models(num_models=1)[0]

        base_model = model.layers[4]
        assert base_model.name == 'densenet121'
        base_model.trainable = True
        pattern = re.compile('conv(\d)_block(\d)_concat')
        block_ends = []
        block_end_names = []
        for i, layer in enumerate(base_model.layers):
            match = pattern.match(layer.name)
            if match is not None:
                block_ends.append(i)
                block_end_names.append(layer.name)
        last_frozen_layer = hp['last_frozen_layer'] if isinstance(hp, dict) else hp.Int('last_frozen_layer',
                                                                                        0,
                                                                                        len(block_ends) - 1)

        for i in range(block_ends[last_frozen_layer]):
            base_model.layers[i].trainable = False

        compile_model(model, learning_rate)
        return model


def show_samples(dataset):
    # Show a couple images from a pipeline, along with their GT, as a sanity check
    n_cols = 4
    # n_rows = int(ceil(16 / n_cols))
    n_rows = 2
    samples_iter = iter(dataset)
    samples = next(samples_iter)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=109.28)
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            axs[row, col].imshow(samples[0][idx])
            x_label = samples[1][idx].numpy()
            y_label = ''
            idx += 1
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_xlabel(x_label)
            axs[row, col].set_ylabel(y_label)
    # plt.show()
    plt.draw()
    plt.pause(.01)


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=.1, fill_mode='nearest', seed=seed),
        tf.keras.layers.experimental.preprocessing.RandomRotation(1 / 60, fill_mode='nearest', seed=seed),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=.1,
                                                                     width_factor=.05,
                                                                     fill_mode='nearest',
                                                                     seed=None)
    ]
)


def load_augment_save(filepath, aug_filepath):
    image_png = tf.io.read_file(filepath)
    image = tf.io.decode_png(image_png, channels=3)
    image = tf.expand_dims(image, axis=0)
    augmented = data_augmentation(image)
    augmented = tf.squeeze(augmented, axis=0)
    augmented_png = tf.io.encode_png(augmented)
    tf.io.write_file(aug_filepath, augmented_png)
    return filepath, aug_filepath


def make_aug_filepath(filepath, i):
    stem = Path(str(filepath)).stem
    aug_filepath = '{}/{}_{:02d}.png'.format(augmented_root, stem, i)
    return aug_filepath


def augment(metadata, rate, batch_size=16, seed=None):
    aug_metadata_list = []
    for i in range(rate):
        aug_metadata = metadata.copy()
        aug_metadata['Augmented File Path'] = metadata['File Path'].apply(
            lambda filepath: make_aug_filepath(filepath, i))
        aug_metadata_list.append(aug_metadata)
    aug_metadata = pd.concat(aug_metadata_list, ignore_index=True)
    found = aug_metadata['Augmented File Path'].apply(lambda filepath: Path(filepath).is_file())
    if not np.alltrue(found):
        dataset = tf.data.Dataset.from_tensor_slices((aug_metadata['File Path'], aug_metadata['Augmented File Path']))
        dataset = dataset.map(load_augment_save, num_parallel_calls=AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

        count_expected = int(np.ceil(len(metadata) * rate / batch_size))

        ds_iter = iter(dataset)
        for _ in tqdm(ds_iter, total=count_expected):
            pass

    aug_metadata.drop('File Path', inplace=True, axis=1)
    aug_metadata.rename({'Augmented File Path': 'File Path'}, inplace=True, axis=1)

    return aug_metadata


def load_train_test(path):
    filepaths_neg = [str(filepath) for filepath in Path(path + '/NORMAL').glob('**/*.jpeg')]
    filepaths_pos = [str(filepath) for filepath in Path(path + '/PNEUMONIA').glob('**/*.jpeg')]
    filepaths = pd.DataFrame({'File Path': filepaths_pos, 'Pneumonia': [1] * len(filepaths_pos)})
    filepaths = pd.concat([filepaths,
                           pd.DataFrame(
                               {'File Path': filepaths_neg,
                                'Pneumonia': [0] * len(filepaths_neg)})],
                          ignore_index=True)
    filepaths['File Name'] = filepaths['File Path'].apply(lambda filepath: Path(filepath).name)
    return filepaths


def load_metadata2():
    train_filepaths = load_train_test(dataset2_path + '/train')
    test_filepaths = load_train_test(dataset2_path + '/test')
    val_filepaths = load_train_test(dataset2_path + '/val')
    test_filepaths = pd.concat([val_filepaths, test_filepaths], ignore_index=True)
    return train_filepaths, test_filepaths


def load_metadata():
    metadata = pd.read_csv(metadata_fpath)
    renaming = {'Image Index': 'File Name',
                'Follow-up #': 'Follow-up',
                'OriginalImage[Width': 'Original Width',
                'Height]': 'Original Height',
                'OriginalImagePixelSpacing[x': 'Original Pixel Spacing X',
                'y]': 'Original Pixel Spacing Y'}
    metadata.rename(renaming, axis=1, inplace=True)
    metadata.drop(['Unnamed: 11'], axis=1, inplace=True)

    # Check that patients (patient IDs) are numbered starting from 1, with no gaps in the numbering
    patient_ids = sorted(metadata['Patient ID'].unique())
    for i, id in enumerate(patient_ids):
        assert id == i + 1

    # For every sample, fill in the complete file path (starting from root) to the image file
    filepaths = [str(filepath) for filepath in list(Path(dataset_path).glob('**/*.png'))]
    filename_to_path = {filepath[-16:]: filepath for filepath in filepaths}
    metadata['File Path'] = metadata['File Name'].map(filename_to_path)
    assert not np.any(metadata['File Path'].isna())
    for filepath in metadata['File Path']:
        assert Path(filepath).is_file()

    return metadata


def make_logger(name, log_level):
    """
    Initializes and return a logger. See https://docs.python.org/3/library/logging.html
    :param name: the name for the logger, a string.
    :param log_level: the requested log level, as documented in https://docs.python.org/3/library/logging.html#levels
    :return: an instance of logging.Logger
    """
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s: %(levelname)s %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def make_k_fold_file_name(stem):
    """
    Returns the pickle file name for a given stem.
    :param stem: the given stem, a string.
    :return: the pickle file name.
    """
    fname = stem + '_kfold.pickle'
    return fname


def keep_last_two_files(file_name):
    if Path(file_name).is_file():
        Path(file_name).replace(file_name + '.prev')
    Path(file_name + '.tmp').replace(file_name)


def k_fold_resumable_fit(model_maker, hp, comp_dir, project, train_datasets, val_datasets, log_dir,
                         log_level=logging.INFO,
                         **kwargs):
    """
    Performs k-fold cross validation of a Keras model, training and validating the model k times. The dataset
    partitioning in k folds is the responsibility of the client code, to be performed in a callback passed to the
    function, see parameter `make_datasets_cb`.

    :param model: the Keras model. In doesn't need to be compiled, as the function will take care of it. If a previous
    training and validation process on the same model has been interrupted, but its state was saved in files, then the
    function will load the model from files and resume training and validation, and this parameter will be ignored,
     it can be set to None.
    :param comp_dir: path to the directory where the training state will be saved, a string.
    :param stem: a stem that will be used to make file names to save the computation state and its results, a string.
    :param compile_cb: a function that will be called to build the Keras model as necessary. It must take one
    argument, which is the model to be compiled; anything it returns is ignored.
    :param make_datasets_cb: a function or method that instantiates the training and validation pipelines.
    It takes three parameters: the fold number, a non-negative integer, the total number of folds, a positive
    integer, and **kwargs, as passed to this function. It must return two instances of td.data.Dataset, with the
    training and validation datasets respectively for the given fold number.
    :param n_folds: number of folds (k) required for the k-fold cross-validation.
    :param log_dir: base name for the directory where to save logs for Tensorboard. Logs for fold XX are saved in
    <log_dir>-fold<nn>, where <nn> is the fold number. Logs for the overall validation, that is the average of the k
    validations, are saved in <log_dir>-xval.
    :param log_level: logging level for this funciton, as defined in package `logging`.
    :param kwargs: parameters to be passed to resumable_fit() for the training and validation of each fold.
    :return: a 2-tuple; the first element of the tuple is a list of k History objects, each with a record of the
    computation on fold k, as returned by tf.Keras.Model.fit(). The second element is a pd.DataFrame with the averages
    of validation metrics per epoch, averaged across folds.
    """
    n_folds = len(train_datasets)
    assert n_folds < 100
    assert len(val_datasets) == n_folds
    # assert kwargs.get('x') is None
    # assert kwargs.get('validation_data') is None

    logger = make_logger(name='k_fold_resumable_fit', log_level=log_level)

    state_file_path = f'{comp_dir}/{project}/comp_state.pickle'
    current_fold = 0
    histories = []

    # Restore the state of the k-fold cross validation from file, if the file is available
    if Path(state_file_path).is_file():
        with open(state_file_path, 'br') as pickle_f:
            pickled = pickle.load(pickle_f)
        current_fold = pickled['fold'] + 1
        histories = pickled['histories']
        logger.info(
            f"Reloaded the state of previous k-fold cross validation from {state_file_path} - {pickled['fold']} folds already computed")
    else:
        logger.info(f"State of k-fold cross validation will be saved in {state_file_path}")

    # saved_model_fname = f'{comp_dir}/{stem}_orig.h5'

    for fold in range(current_fold, n_folds):
        model = model_maker(hp)
        if fold == 0:
            logger.info(f'Starting cross-validation on fold 0 for model {model.name}')
        else:
            logger.info(f'Resuming cross-validation on fold {fold} for model {model.name}')

        log_dir = '{}/{}/fold-{:02d}'.format(comp_dir, project, fold)
        tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                        histogram_freq=1,
                                                        profile_batch=0)
        callbacks = kwargs.get('callbacks', [])
        callbacks.append(tensorboard_cb)
        kwargs['callbacks'] = callbacks
        history = model.fit(x=train_datasets[fold], validation_data=val_datasets[fold], **kwargs)
        histories.append(history.history)
        # Update the state of the k-fold x-validation as saved in the pickle
        with open(state_file_path + '.tmp', 'bw') as pickle_f:
            pickle_this = {'fold': fold, 'histories': histories}
            pickle.dump(obj=pickle_this, file=pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
            keep_last_two_files(state_file_path)

    logger.info(f"All {n_folds} folds of the cross-validation have been processed")
    histories_df = None
    for i, history in enumerate(histories):
        if histories_df is None:
            histories_df = pd.DataFrame(history)
            histories_df['fold'] = i
            histories_df['epoch'] = histories_df.index
        else:
            history_df = pd.DataFrame(history)
            history_df['fold'] = i
            history_df['epoch'] = history_df.index
            histories_df = pd.concat([histories_df, history_df], ignore_index=True)

    means = histories_df.groupby(['epoch']).mean()
    to_be_dropped = ['fold']
    """for column in means.columns:
        if str(column)[:4] != 'val_':
            to_be_dropped.append(column)"""
    means.drop(labels=to_be_dropped, axis=1, inplace=True)
    means['epoch'] = means.index

    file_writer = tf.summary.create_file_writer(log_dir + '-xval')
    file_writer.set_as_default()  # Note: if you don't set this deafault, nothing will be logged
    for column in means.columns:
        for epoch, data in zip(means['epoch'], means[column]):
            tf.summary.scalar(str(column),
                              data=data,
                              step=epoch,
                              description='Average of validation metrics across folds during k-fold cross validation.')

    means.to_csv(f'{comp_dir}/{project}/xval_report.csv', index=False)

    return histories, means


def main():
    metadata = load_metadata()
    # train_metadata2, test_metadata2 = load_metadata2()

    ''' Fraction of positive samples wanted in both test and validation set, the same as it is in the test set of 
    the Kaggle Chest X-Ray Images (Pneumonia) dataset'''

    train_metadata, val_metadata, test_metadata = split_dataset(metadata, augmentation + 1)
    # train_metadata = train_metadata.head(8)
    # val_metadata = val_metadata.head(256)

    if augmentation > 0:
        pos_train_metadata = train_metadata[train_metadata['Pneumonia'] == 1]
        neg_train_metadata = train_metadata[train_metadata['Pneumonia'] == 0]
        pos_train_metadata_augmented = augment(metadata=pos_train_metadata, rate=augmentation, batch_size=16, seed=seed)
        train_metadata = pd.concat([pos_train_metadata, pos_train_metadata_augmented, neg_train_metadata],
                                   ignore_index=True)
    # train_metadata = pd.concat(([train_metadata, train_metadata2]), ignore_index=True)
    train_metadata = shuffle_DataFrame(train_metadata, seed)

    """val_metadata2, test_metadata2 = split_dataset2(test_metadata2)
    val_metadata = pd.concat(([val_metadata, val_metadata2]), ignore_index=True)
    test_metadata = pd.concat(([test_metadata, test_metadata2]), ignore_index=True)"""

    print('Train:')
    print(Counter(train_metadata['Pneumonia']))
    print('Validation:')
    print(Counter(val_metadata['Pneumonia']))
    print('Test:')
    print(Counter(test_metadata['Pneumonia']))

    train_ds = make_pipeline(file_paths=train_metadata['File Path'].to_numpy(),
                             y=train_metadata['Pneumonia'].to_numpy(),
                             shuffle=True,
                             batch_size=batch_size,
                             seed=seed)
    val_ds = make_pipeline(file_paths=val_metadata['File Path'].to_numpy(),
                           y=val_metadata['Pneumonia'].to_numpy(),
                           shuffle=False,
                           batch_size=val_batch_size)

    samples_ds = make_pipeline(file_paths=train_metadata['File Path'].to_numpy(),
                               y=train_metadata['Pneumonia'].to_numpy(),
                               shuffle=True,
                               batch_size=16,
                               seed=seed)
    show_samples(samples_ds)
    del samples_ds  # Not needed anymore

    """str_time = datetime.datetime.now().strftime("%m%d-%H%M%S")
    log_dir = f'{logs_root}/{str_time}'

    logs_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=0)
    history = model.fit(x=train_ds, validation_data=val_ds, epochs=300, shuffle=False, callbacks=[logs_cb])"""

    # TODO what if I use val_loss instead?
    tuner = Hyperband(make_model_DenseNet121,
                      objective=kt.Objective("val_auc", direction="max"),  # Careful to keep the direction updated
                      max_epochs=epochs_base,
                      hyperband_iterations=2,
                      directory='computations',
                      project_name='base-1dataset-densenet121-auc-auc')

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=patience)

    tuner.search(x=train_ds,
                 validation_data=val_ds,
                 epochs=epochs_base,
                 shuffle=False,
                 callbacks=[early_stopping_cb])

    model_maker = ModelMaker(tuner)

    tuner_ft = Hyperband(model_maker.make_model_DenseNet121,
                         objective=kt.Objective("val_auc", direction="max"),  # Careful to keep the direction updated
                         max_epochs=epochs_ft,
                         hyperband_iterations=2,
                         directory='computations',
                         project_name='fine-1dataset-densenet121-auc-auc')

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=patience)

    tuner_ft.search(x=train_ds,
                    validation_data=val_ds,
                    epochs=epochs_ft,
                    shuffle=False,
                    callbacks=[early_stopping_cb])

    best_ft_model = tuner_ft.get_best_models()[0]

    test_ds = make_pipeline(file_paths=test_metadata['File Path'].to_numpy(),
                            y=test_metadata['Pneumonia'].to_numpy(),
                            shuffle=False,
                            batch_size=val_batch_size)

    test_results = best_ft_model.evaluate(x=test_ds, return_dict=True, verbose=1)
    print('\nHyper-parameters for the fine-tuned model:')
    print(tuner_ft.get_best_hyperparameters()[0].values)
    print('\nTest results on the fine-tuned model:')
    print(test_results)

    hps = tuner_ft.get_best_hyperparameters()[0].values

    dev_metadata = pd.concat([train_metadata, val_metadata], ignore_index=True)
    dev_metadata = shuffle_DataFrame(dev_metadata, seed=seed)
    dev_metadata.reset_index(inplace=True, drop=True)
    fold_size = int(np.ceil(len(dev_metadata) / n_folds))
    train_datasets, val_datasets = [], []
    for fold in range(n_folds):
        fold_val_metadata = dev_metadata.iloc[fold * fold_size:fold * fold_size + fold_size, :]
        val_datasets.append(make_pipeline(file_paths=fold_val_metadata['File Path'].to_numpy(),
                                          y=fold_val_metadata['Pneumonia'].to_numpy(),
                                          batch_size=val_batch_size,
                                          shuffle=False))
        fold_train_metadata1 = dev_metadata.iloc[:fold * fold_size, :]
        fold_train_metadata2 = dev_metadata.iloc[fold * fold_size + fold_size:, :]
        fold_train_metadata = pd.concat([fold_train_metadata1, fold_train_metadata2], ignore_index=False)
        train_datasets.append(make_pipeline(file_paths=fold_train_metadata['File Path'].to_numpy(),
                                            y=fold_train_metadata['Pneumonia'].to_numpy(),
                                            batch_size=batch_size,
                                            shuffle=True,
                                            seed=seed))

    histories, means = k_fold_resumable_fit(model_maker=model_maker.make_model_DenseNet121,
                                            hp=hps,
                                            comp_dir='computations',
                                            project='xval',
                                            train_datasets=train_datasets,
                                            val_datasets=val_datasets,
                                            log_dir='computations/xval/logs',
                                            epochs=epochs_ft)

    print(histories)
    print(means)
    pass


if __name__ == '__main__':
    main()

""" TODO
can you have TB logs for the tuner trials?
CHeck if TF warnings "Unresolved object in checkpoint" can be ignored
Maximise dynamic range during pre-processing
run overnight
Modulate the classification threshold to choose an F1 or precision/recall tradeoff (do it with inference only, in a notebook)
Check "Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection using Chest X-ray"
https://arxiv.org/abs/2004.06578
"""
