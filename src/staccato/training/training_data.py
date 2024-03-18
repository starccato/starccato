"""This loads the signal data from the raw simulation outputs from Richers et al (20XX) ."""
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple

# TODO: @tarin-e, save these on the zenodo as well.
SIGNALS_CSV = "https://raw.githubusercontent.com/tarin-e/gw-generative-models/main/data/input/richers_1764.csv"
PARAMETERS_CSV = "https://raw.githubusercontent.com/tarin-e/gw-generative-models/main/data/input/richers_1764_parameters.csv"
TIME_CSV = "https://raw.githubusercontent.com/tarin-e/gw-generative-models/main/data/input/richers_1764_times.csv"


class TrainingData(Dataset):
    def __init__(self, signals_csv=SIGNALS_CSV, parameters_csv=PARAMETERS_CSV):
        ### read data from csv files
        parameters = pd.read_csv(parameters_csv)
        self.signals = pd.read_csv(signals_csv).astype('float32')
        # remove unusual parameters
        keep_signals_idx = np.array(parameters[parameters['beta1_IC_b'] > 0].index)
        parameters = parameters.iloc[:, :]
        ###

        ### process beta_ic_b parameter
        # ranges = [0, 0.06, 0.17, 1]
        # labels = [0, 1, 2]
        # num_classes = len(labels)
        # y = y['beta1_IC_b']
        # y = pd.cut(y, bins=ranges, labels=labels).astype('int')
        # y = y.values
        # y = np.eye(num_classes)[y]
        # y = np.reshape(y, (y.shape[0], y.shape[1], 1)).astype('float32')
        self.parameters = parameters
        ###

        # drop corresponding signals which have erroneous parameter values
        self.signals = self.signals.iloc[:, keep_signals_idx]
        self.signals = self.signals.values
        self.augmented_signals = np.empty(shape=(256, 0)).astype('float32')

        ### flatten signals and take last 256 timestamps
        temp_data = np.empty(shape=(256, 0)).astype('float32')

        for i in range(0, self.signals.shape[1]):
            signal = self.signals[:, i]
            signal = signal.reshape(1, -1)

            cut_signal = signal[:, int(len(signal[0]) - 256):len(signal[0])]
            temp_data = np.insert(temp_data, temp_data.shape[1], cut_signal, axis=1)

        self.signals = temp_data
        self.mean = self.signals.mean()
        self.std = np.std(self.signals, axis=None)
        self.scaling_factor = 5
        self.ylim_signal = (self.signals[:, :].min(), self.signals[:, :].max())

    ### augmentation methods ###
    def jittering_augmentation(self, signal):
        # todo: add noise only after time of core bounce
        # noise_start_time = 203
        noise = np.random.normal(0, 1, signal.shape[1])
        jittered_signal = signal + noise

        return jittered_signal

    def shift_augmentation(self, signal):
        shift = np.random.normal(0, 50, 1)
        shifted_signal = np.roll(signal, int(shift))

        return shifted_signal

    def scale_augmentation(self, signal):
        scale_factor = np.random.normal(1, 0.5, 1)
        scale_factor = np.maximum(scale_factor, 0)
        scaled_signal = scale_factor * signal
        return scaled_signal

    def mixture_augmentation(self, signal_1, signal_2):
        distance_multiplier = np.random.normal(0.5, 0.2, 1)
        # clip signal to range [0,1] as this is the multiplier by the normalised difference in signals
        distance_multiplier = np.clip(distance_multiplier, 0, 1)
        mixture_signal = signal_1 + distance_multiplier * (signal_2 - signal_1)

        return mixture_signal

    def window_warping_augmentation(self, signal):
        # take window size of 10% of the signal with a warping factor of 2 or 0.5 (from literature)
        warping_factor = np.random.choice([0.5, 2])
        # warping_factor = 0.5

        window_size = math.floor(signal.shape[1] / 10)
        scaled_window_size = warping_factor * window_size

        # don't warp anything a little bit before the core-bounce - preserves core-bounce position
        window_min_idx = 53

        # find random reference position for start of window warping
        window_start_idx = np.random.randint(window_min_idx, signal.shape[1] - scaled_window_size * 2)
        window_end_idx = window_start_idx + window_size

        # select between warping by factor 1/2 or 2
        if (warping_factor == 2):
            # extract values before, at and after the window
            # clip end of signal to make up for extra size due to window warping
            signal_before_window = signal[0][:window_start_idx]
            signal_window = signal[0][window_start_idx:window_end_idx]
            signal_after_window = signal[0][window_end_idx:int(signal.shape[1] - (window_size))]

            # time points
            t = np.arange(len(signal_window))
            warped_t = np.arange(0, len(signal_window), 0.5)

            # interpolation for window warping
            signal_window_warped = np.interp(warped_t, t, signal_window)

            # combine signals
            warped_signal = np.concatenate((signal_before_window, signal_window_warped, signal_after_window), axis=0)
        elif (warping_factor == 0.5):
            # extract values before, at and after the window
            # clip end of signal to make up for extra size due to window warping
            signal_before_window = signal[0][:window_start_idx]
            signal_window = signal[0][window_start_idx:window_end_idx]
            signal_after_window = signal[0][window_end_idx:]
            # add values to end of signal to make up for downsampled window
            signal_after_window = np.pad(signal_after_window, (0, int(window_size - scaled_window_size)), mode='edge')

            signal_window_warped = signal_window[::int(1 / warping_factor)]

            warped_signal = np.concatenate((signal_before_window, signal_window_warped, signal_after_window), axis=0)
        else:
            warped_signal = signal

        return warped_signal

    def summary(self):
        """Display summary stats about the data"""
        str = f"Signal Dataset mean: {self.mean:.3f} +/- {self.std:.3f}\n"
        str += f"Signal Dataset scaling factor (to match noise in generator): {self.scaling_factor}\n"
        str += f"Signal Dataset shape: {self.signals.shape}\n"
        print(str)


    def standardize(self, signal):
        standardized_signal = (signal - self.mean) / self.std
        standardized_signal = standardized_signal / self.scaling_factor
        return standardized_signal

    def augmentation(self, desired_augmented_data_count):
        while self.signals.shape[1] < desired_augmented_data_count:
            idx_1 = np.random.randint(0, self.signals.shape[1])
            signal_1 = self.signals[:, idx_1]
            signal_1 = signal_1.reshape(1, -1)

            ### mixture augmentation only ###
            # find the class of signal_1 (assuming class is a column in self.parameters)
            beta_class_of_signal_1 = np.argmax(self.parameters[idx_1, :])
            # sample only from the same class for signal_2 and make sure it's not the same as signal_1
            candidate_indices = [x for x in range(0, 1764) if
                                 x != idx_1 and np.argmax(self.parameters[x, :]) == beta_class_of_signal_1]
            idx_2 = np.random.choice(candidate_indices)
            signal_2 = self.signals[:, idx_2]
            signal_2 = signal_2.reshape(1, -1)

            # call selected augmentation function here
            # augmented_signal = self.window_warping_augmentation(signal_1)
            augmented_signal = self.mixture_augmentation(signal_1, signal_2)

            self.augmented_data = np.insert(self.augmented_data, self.augmented_data.shape[1], augmented_signal, axis=1)
            self.signals = np.insert(self.signals, self.signals.shape[1], augmented_signal, axis=1)

            # just copy parameters for now
            augmented_parameter = self.parameters[idx_1, :]

            self.augmented_parameters = np.insert(self.augmented_parameters, self.augmented_parameters.shape[0],
                                                  augmented_parameter, axis=0)
            self.parameters = np.insert(self.parameters, self.parameters.shape[0], augmented_parameter, axis=0)

        print("Signal Dataset Size after Data Augmentation: ", self.signals.shape)
        print("Parameter Dataset Size after Data Augmentation: ", self.parameters.shape)

    ### overloads ###
    def __len__(self):
        return self.signals.shape[1]

    def __getitem__(self, idx):
        signal = self.signals[:, idx]
        signal = signal.reshape(1, -1)

        signal_standardized = self.standardize(signal)

        return signal_standardized

    def get_loader(self, batch_size: int = 32) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=True, num_workers=0)

    def get_signals_iterator(self):
        return next(iter(self.get_loader()))

    def plot_waveforms(self, fname=None, standardised=False) -> Tuple[plt.Figure, plt.Axes]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes = axes.flatten()
        signal_iterator = self.get_signals_iterator()

        # Plot each signal on a separate subplot
        for i, ax in enumerate(axes):
            x = np.arange(signal_iterator.size(dim=2))
            y = signal_iterator[i, :, :].flatten()

            if standardised:
                y = y * self.scaling_factor
                y = y * self.std + self.mean

            ax.plot(x, y, color='blue')

            ax.axvline(x=53, color='black', linestyle='--', alpha=0.5)
            ax.grid(True)
            ax.set_ylim((-4, 2))
            if standardised:
                ax.set_ylim(self.ylim_signal)

            # Add axis titles
            ax.set_ylabel('distance * strain (cm^2)')
            ax.set_xlabel('n (timestamps)')
            ax.set_xlim(min(x), max(x))


            # parameters = signal_iterator[i, :].numpy()[0]
            # parameters_with_names = f'{parameter_names[0]}: {parameters[0]:.6f}\n{parameter_names[1]}: {parameters[1]:.2f}, {parameter_names[2]}: {parameters[2]:.2f}'
            # ax.set_xlabel(f'Parameters:\n{parameters_with_names}')


        fig.suptitle('Waveforms')
        if standardised:
            fig.suptitle('Standardised Waveforms')

        for i in range(407, 8 * 4):
            fig.delaxes(axes[i])

        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname)
        return fig, axes
