import re
from pathlib import Path
from collections import Counter
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from kerastuner.tuners import Hyperband
import kerastuner as kt

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
epochs = 20
augmentation = 3


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
    train_metadata = metadata.iloc[train_patients.index]
    val_metadata = metadata.iloc[val_patients.index]
    test_metadata = metadata.iloc[test_patients.index]

    # The number of positive samples currently in the train set
    train_n_pos = np.sum(train_metadata['Pneumonia'])
    # The number of negative samples wanted for the train set
    wanted_neg = min(train_n_pos * neg_to_pos_ratio, len(train_metadata) - train_n_pos)

    ''' Sample and remove enough negative samples from the train set to obtain a train set balanced between positive and
    negative samples '''
    # TODO Should I do this at a patient (not sample) level?
    # Compile a dataframe with all the negative samples (of the train set)
    non_pneumonia = train_metadata[train_metadata['Pneumonia'] == 0].copy()  # Make a copy, don't take a copy of a slice
    # Choose enough of them, stratifying over the fact that the samples have other findings, different from pneumonia
    non_pneumonia['Other Findings'] = non_pneumonia['Finding Labels'] != 'No Finding'
    non_pneumonia_sampled, _ = train_test_split(non_pneumonia,
                                                train_size=wanted_neg,
                                                random_state=seed,
                                                stratify=non_pneumonia['Other Findings'])
    pneumonia = train_metadata[train_metadata['Pneumonia'] == 1]
    train_metadata = pd.concat([pneumonia, non_pneumonia_sampled], axis=0, ignore_index=True)
    # The 'Other Findings' column is not needed anymore
    train_metadata.drop('Other Findings', axis=1, inplace=True)
    # Shuffle the rows of the result
    train_metadata = train_metadata.sample(n=len(train_metadata), replace=False, random_state=seed)
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


def make_model(hp):
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
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])
    return model


class ModelMaker:
    def __init__(self, tuner):
        self.tuner = tuner

    def make_model(self, hp):
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

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
                               tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.AUC(name='auc')])
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

        ds_iter = iter(dataset)
        for _ in ds_iter:
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


def main():
    metadata = load_metadata()
    metadata2_train, metadata2_test = load_metadata2()

    train_metadata, val_metadata, test_metadata = split_dataset(metadata, augmentation + 1)
    # train_metadata = train_metadata.head(8)
    # val_metadata = val_metadata.head(256)

    metadata2 = load_metadata2()

    pos_train_metadata = train_metadata[train_metadata['Pneumonia'] == 1]
    neg_train_metadata = train_metadata[train_metadata['Pneumonia'] == 0]
    pos_train_metadata_augmented = augment(metadata=pos_train_metadata, rate=augmentation, batch_size=16, seed=seed)
    train_metadata = pd.concat([pos_train_metadata, pos_train_metadata_augmented, neg_train_metadata],
                               ignore_index=True)
    train_metadata = train_metadata.sample(n=len(train_metadata), replace=False, random_state=seed)

    train_ds = make_pipeline(file_paths=train_metadata['File Path'].to_numpy(),
                             y=train_metadata['Pneumonia'].to_numpy(),
                             shuffle=True,
                             batch_size=batch_size,
                             seed=seed)
    val_ds = make_pipeline(file_paths=val_metadata['File Path'].to_numpy(),
                           y=val_metadata['Pneumonia'].to_numpy(),
                           shuffle=False,
                           batch_size=val_batch_size)

    # model = make_model()
    # model.summary()

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
    tuner = Hyperband(make_model,
                      objective=kt.Objective("val_auc", direction="max"),  # Careful to keep the direction updated
                      max_epochs=epochs,
                      hyperband_iterations=2,
                      directory='computations',
                      project_name='base_model-aug3-auc')

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=patience)

    """tuner.search(x=train_ds,
                 validation_data=val_ds,
                 epochs=epochs,
                 shuffle=False,
                 callbacks=[early_stopping_cb])"""

    model_maker = ModelMaker(tuner)

    tuner_ft = Hyperband(model_maker.make_model,
                         objective=kt.Objective("val_auc", direction="max"),  # Careful to keep the direction updated
                         max_epochs=epochs,
                         hyperband_iterations=2,
                         directory='computations',
                         project_name='fine_tuning-aug3-auc')

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=patience)

    tuner_ft.search(x=train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    shuffle=False,
                    callbacks=[early_stopping_cb])


if __name__ == '__main__':
    main()

""" TODO
Maximise dynamic range during pre-processing
Check "Transfer Learning with Deep Convolutional Neural Network (CNN) for Pneumonia Detection using Chest X-ray"
https://arxiv.org/abs/2004.06578
"""
