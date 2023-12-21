# EEG data handling
import mne
from mne.datasets import eegbci
import numpy as np

# Data processing
from sklearn.preprocessing import LabelEncoder


class BrainComputerInterface:
    """
    Class for handling EEG data and extracting features from it.

    Attributes
    ----------
    subject : int
        Subject number.
    runs_motor : list[int]
        List of motor runs.
    runs_imaginary : list[int]
        List of imaginary runs.
    tasks_dict : dict
        Dictionary mapping run number to task.
    classifiers : list[Classifier]
        List of classifiers used for prediction.
    pipeline : Pipeline
        Pipeline used for prediction.

    Methods
    -------
    _validate_input(subject, runs_motor, runs_imaginary)
        Validates input parameters.
    _init_tasks_dict()
        Initializes tasks_dict.
    _get_tasks_number()
        Returns a string of concatenated runs number.
    _load_eeg_data(subject, runs, event_mapping)
        Loads EEG data from files.
    _load_motor_data(subject, runs_motor)
        Loads motor data.
    _load_imaginary_data(subject, runs_imaginary)
        Loads imaginary data.
    _load_raw_data(subject, runs_motor, runs_imaginary)
        Loads raw data from files and concatenates it.
    _extract_features(raw_files, tmin, tmax, fft)
        Extracts features from raw data.
    _load_and_extract_features(subject, runs_motor, runs_imaginary, tmin, tmax, fft)
        Loads raw data and extracts features from it.
    _apply_fft(data)
        Applies FFT on data.
    """

    def __init__(
        self,
        subject: int = None,
        runs_motor: list[int] = None,
        runs_imaginary: list[int] = None,
    ):
        self._validate_input(subject, runs_motor, runs_imaginary)
        self.subject = subject
        self.runs_motor = runs_motor
        self.runs_imaginary = runs_imaginary
        self.tasks_dict = {}
        self._init_tasks_dict()
        self.classifiers = None
        self.pipeline = None
        mne.set_log_level("WARNING")
        mne.set_config(
            "MNE_DATASETS_EEGBCI_PATH", "/mnt/nfs/homes/jsemel/sgoinfre/mne_data"
        )

    @staticmethod
    def _validate_input(
        subject: int, runs_motor: list[int], runs_imaginary: list[int]
    ) -> None:
        """
        Validates input parameters.

        Parameters
        ----------
        subject : int
            Subject number.
        runs_motor : list[int]
            List of motor runs.
        runs_imaginary : list[int]
            List of imaginary runs.

        Raises
        ------
        TypeError
            If subject is not an integer.
            If runs_motor is not a list of integers.
            If runs_imaginary is not a list of integers.
        ValueError
            If subject is not between 1 and 109.
            If runs_motor is not between 1 and 14.
            If runs_imaginary is not between 1 and 14.

        Returns
        -------
        None
        """
        if not isinstance(subject, int):
            raise TypeError("Subject must be an integer.")
        if not all(isinstance(x, int) for x in runs_motor):
            raise TypeError("Runs motor must be a list of integers.")
        if not all(isinstance(x, int) for x in runs_imaginary):
            raise TypeError("Runs imaginary must be a list of integers.")
        if not (1 <= subject <= 109):
            raise ValueError("Subject must be between 1 and 109.")
        if not all(1 <= x <= 14 for x in runs_motor):
            raise ValueError("Runs motor must be between 1 and 14.")
        if not all(1 <= x <= 14 for x in runs_imaginary):
            raise ValueError("Runs imaginary must be between 1 and 14.")

    def _init_tasks_dict(self) -> None:
        """
        Initializes tasks_dict.

        Returns
        -------
        None
        """
        physio_tasks = {
            1: "rest",
            2: "rest",
            3: "motor/fists",
            4: "imagine/fists",
            5: "motor/feet",
            6: "imagine/feet",
            7: "motor/fists",
            8: "imagine/fists",
            9: "motor/feet",
            10: "imagine/feet",
            11: "motor/fists",
            12: "imagine/fists",
            13: "motor/feet",
            14: "imagine/feet",
        }
        for run in self.runs_motor:
            self.tasks_dict[run] = physio_tasks[run]
        for run in self.runs_imaginary:
            self.tasks_dict[run] = physio_tasks[run]

    def _get_tasks_number(self) -> str:
        """
        Returns a string of concatenated runs number.

        Returns
        -------
        str
            String of concatenated runs number.
        """
        return "_".join([str(x) for x in self.runs_motor + self.runs_imaginary])

    @staticmethod
    def _load_eeg_data(subject: int, runs: list[int], event_mapping: dict) -> list:
        """
        Loads EEG data from files.

        Parameters
        ----------
        subject : int
            Subject number.
        runs : list[int]
            List of runs.
        event_mapping : dict
            Dictionary mapping event number to event description.

        Returns
        -------
        list
            List of raw EEG files.
        """
        raw_files = []
        for run in runs:
            raw_list = [
                mne.io.read_raw_edf(f, preload=True, stim_channel="auto")
                for f in eegbci.load_data(subject, run)
            ]
            raw_concatenated = mne.concatenate_raws(raw_list)
            events, _ = mne.events_from_annotations(
                raw_concatenated, event_id=dict(T0=1, T1=2, T2=3)
            )
            annot_from_events = mne.annotations_from_events(
                events=events,
                event_desc=event_mapping,
                sfreq=raw_concatenated.info["sfreq"],
                orig_time=raw_concatenated.info["meas_date"],
            )
            raw_concatenated.set_annotations(annot_from_events)
            raw_files.append(raw_concatenated)
        return raw_files

    def _load_motor_data(self, subject: int, runs_motor: list[int]) -> list:
        """
        Loads motor data.

        Parameters
        ----------
        subject : int
            Subject number.
        runs_motor : list[int]
            List of motor runs.

        Returns
        -------
        list
            List of raw EEG files.
        """
        mapping = {1: "rest"}
        for run in runs_motor:
            task = self.tasks_dict.get(run)
            if task == "motor/feet":
                mapping[2] = "motor/feet"
            elif task == "motor/fists":
                mapping[3] = "motor/fists"

        return self._load_eeg_data(subject, runs_motor, mapping)

    def _load_imaginary_data(self, subject: int, runs_imaginary: list[int]) -> list:
        """
        Loads imaginary data.

        Parameters
        ----------
        subject : int
            Subject number.
        runs_imaginary : list[int]
            List of imaginary runs.

        Returns
        -------
        list
            List of raw EEG files.
        """
        mapping = {1: "rest"}
        for run in runs_imaginary:
            task = self.tasks_dict.get(run)
            if task == "imagine/feet":
                mapping[2] = "imagine/feet"
            elif task == "imagine/fists":
                mapping[3] = "imagine/fists"

        return self._load_eeg_data(subject, runs_imaginary, mapping)

    def _load_raw_data(
        self, subject: int, runs_motor: list[int], runs_imaginary: list[int]
    ) -> list:
        """
        Loads raw data from files and concatenates it.

        Parameters
        ----------
        subject : int
            Subject number.
        runs_motor : list[int]
            List of motor runs.
        runs_imaginary : list[int]
            List of imaginary runs.

        Returns
        -------
        list
            List of raw EEG files.
        """
        raw_files = []
        raw_files += self._load_motor_data(subject, runs_motor)
        raw_files += self._load_imaginary_data(subject, runs_imaginary)
        return raw_files

    def _extract_features(
        self, raw_files: list, tmin: float, tmax: float, fft: bool
    ) -> tuple[list, np.ndarray]:
        """
        Extracts features from raw data.

        Parameters
        ----------
        raw_files : list
            List of raw EEG files.
        tmin : float
            Start time before event.
        tmax : float
            End time after event.
        fft : bool
            Whether to apply FFT on data.

        Returns
        -------
        tuple[list, np.ndarray]
            Tuple of transformed data and labels.
        """
        raw = mne.concatenate_raws(raw_files)
        picks = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )

        raw.filter(8.0, 40.0, fir_design="firwin", skip_by_annotation="edge")

        event_id = {
            "motor/feet": 1,
            "motor/fists": 2,
            "imagine/feet": 3,
            "imagine/fists": 4,
            "rest": 5,
        }
        event_id = {
            k: v for k, v in event_id.items() if k in raw.annotations.description
        }
        events, event_dict = mne.events_from_annotations(raw=raw, event_id=event_id)
        epochs = mne.Epochs(
            raw=raw,
            events=events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            proj=True,
            picks=picks,
            baseline=None,
            preload=True,
        )

        label_encoder = LabelEncoder()
        epoch_data = epochs.get_data(copy=True).astype(np.float64)
        transformed_data = self._apply_fft(epoch_data) if fft else epoch_data
        labels = label_encoder.fit_transform(epochs.events[:, -1] - 1)

        return transformed_data, labels

    def _load_and_extract_features(
        self,
        subject: int,
        runs_motor: list[int],
        runs_imaginary: list[int],
        tmin: float,
        tmax: float,
        fft: bool,
    ) -> tuple[list, np.ndarray]:
        """
        Loads raw data and extracts features from it.

        Parameters
        ----------
        subject : int
            Subject number.
        runs_motor : list[int]
            List of motor runs.
        runs_imaginary : list[int]
            List of imaginary runs.
        tmin : float
            Start time before event.
        tmax : float
            End time after event.
        fft : bool
            Whether to apply FFT on data.

        Returns
        -------
        tuple[list, np.ndarray]
            Tuple of transformed data and labels.
        """
        raw_files = self._load_raw_data(subject, runs_motor, runs_imaginary)
        return self._extract_features(raw_files, tmin, tmax, fft)

    def _apply_fft(self, data: np.ndarray) -> np.ndarray:
        """
        Applies FFT on data.

        Parameters
        ----------
        data : np.ndarray
            Data to apply FFT on.

        Returns
        -------
        np.ndarray
            Data after FFT.
        """
        # Apply FFT and retain only the absolute values (power spectral density)
        fft_result = np.fft.rfft(data, axis=2)
        fft_abs = np.abs(fft_result)

        # Since FFT halves the last dimension, we take twice the values except for DC component
        fft_power = np.concatenate((fft_abs[:, :, 0:1], fft_abs[:, :, 1:] * 2), axis=2)
        return fft_power
