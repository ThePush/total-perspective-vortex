import numpy as np
from sklearn.model_selection import train_test_split
from BrainComputerInterface import BrainComputerInterface
import joblib
import time
from KafkaStreamer import KafkaStreamer
import threading


class BCIPredictor(BrainComputerInterface):
    """
    Class for predicting the labels of the test data using the trained model

    Attributes
    ----------
    subject : int
        Subject number.
    runs_motor : list[int]
        List of motor runs.
    runs_imaginary : list[int]
        List of imaginary runs.
    kafka_server : str
        Kafka server address.
    topic : str
        Kafka topic name.
    buffer : list[np.ndarray]
        Buffer for storing the data chunks.

    Methods
    -------
    _load_model(model_name: str)
        Load the model from the file.
    _init_testing_data()
        Initialize the testing data.
    _process_message(message)
        Process the message from the Kafka topic.
    stream_predictions()
        Stream the predictions from the Kafka topic.
    _get_model_score()
        Get the model score.
    _get_predictions_accuracy()
        Get the predictions accuracy.
    _get_best_accuracy()
        Get the best accuracy.
    print_accuracy()
        Print the accuracy.

    Raises
    ------
    FileNotFoundError
        If the model file is not found.
    """

    def __init__(self, subject, runs_motor, runs_imaginary, kafka_server, topic):
        super().__init__(subject, runs_motor, runs_imaginary)
        model_name = f"subject_{subject}_task_{self._get_tasks_number()}.pkl"
        try:
            self.model, self.best_interval, self.fft = self._load_model(model_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(e)
        self.kafka_server = kafka_server
        self.topic = topic
        self.buffer = []
        self.predictions_results = []
        self.X_test, self.y_test = self._init_testing_data()

    @staticmethod
    def _load_model(model_name: str) -> tuple:
        """
        Load the model from the file.

        Parameters
        ----------
        model_name : str
            Name of the model file.

        Raises
        ------
        FileNotFoundError
            If the model file is not found.

        Returns
        -------
        tuple
            Tuple of the model, best interval and fft.
        """
        try:
            with open(f"models/{model_name}", "rb") as file:
                model_data = joblib.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model {model_name} not found")
        return model_data["model"], model_data["best_interval"], model_data["fft"]

    def _init_testing_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialize the testing data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of the testing data and labels.
        """
        epochs_data, labels = self._load_and_extract_features(
            self.subject,
            self.runs_motor,
            self.runs_imaginary,
            *self.best_interval,
            self.fft,
        )
        X_train, X_test, _, y_test = train_test_split(
            epochs_data, labels, stratify=labels, random_state=42
        )
        # self.buffer.append(X_train)
        return X_test, y_test

    def _process_message(self, message):
        """
        Process the message from the Kafka topic.

        Parameters
        ----------
        message : dict
            Message from the Kafka topic.

        Returns
        -------
        None
        """
        epoch_number = message.value["epoch"]
        data_chunk = np.array(message.value["data"])
        label = message.value["label"]

        self.buffer.append(data_chunk)
        prediction = self.model.predict(np.vstack(self.buffer))

        # prediction = self.model.predict(data_chunk)

        is_correct = prediction[-1] == label[-1]
        self.predictions_results.append(is_correct)

        result_str = "True" if is_correct else "False"
        color_code = "\033[92m" if result_str == "True" else "\033[91m"

        print(
            f" | {epoch_number:^5} | {prediction[-1]:^10} | {label[-1]:^12} | {color_code}{result_str:^10}\033[0m |"
        )

    def stream_predictions(self):
        """
        Stream the predictions from the Kafka topic.

        Returns
        -------
        None
        """
        streamer = KafkaStreamer(self.kafka_server, self.topic)
        data = [
            {
                "epoch": i + 1,
                "data": self.X_test[i : i + 1].tolist(),
                "label": self.y_test[i : i + 1].tolist(),
            }
            for i in range(len(self.X_test))
        ]

        producer_thread = threading.Thread(target=streamer.send_data, args=(data,))
        consumer_thread = threading.Thread(
            target=streamer.receive_data, args=(self._process_message,)
        )

        print(f" Subject {self.subject:03} - Task {self._get_tasks_number()}")
        header = " | {:^5} | {:^10} | {:^12} | {:^10} |".format(
            "Epoch", "Prediction", "Ground truth", "Result"
        )
        separator = " " + "-" * (len(header) - 1)
        print(separator)
        print(header)
        print(separator)

        consumer_thread.start()
        time.sleep(1)
        producer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        print("Accuracy:", np.mean(self.predictions_results))
        print(separator)

    def _get_model_score(self) -> float:
        """
        Get the model score.

        Raises
        ------
        Exception
            If the model is not initialized.

        Returns
        -------
        float
            Model score.
        """
        if self.model is None:
            raise Exception("Model not initialized")
        return self.model.score(self.X_test, self.y_test)

    def _get_predictions_accuracy(self) -> float:
        """
        Get the predictions accuracy.

        Returns
        -------
        float
            Predictions accuracy.
        """
        for i in range(len(self.X_test)):
            self.buffer.append(self.X_test[i : i + 1])
            prediction = self.model.predict(np.vstack(self.buffer))
            is_correct = prediction[-1] == self.y_test[i : i + 1]
            self.predictions_results.append(is_correct)
        return np.mean(self.predictions_results)

    def _get_best_accuracy(self) -> float:
        """
        Get the best accuracy.

        Returns
        -------
        float
            Best accuracy.
        """
        model_score = self._get_model_score()
        predictions_accuracy = self._get_predictions_accuracy()
        return (
            model_score if model_score > predictions_accuracy else predictions_accuracy
        )

    def print_accuracy(self) -> float:
        """
        Print the accuracy.

        Returns
        -------
        float
            Accuracy.
        """
        run = self._get_tasks_number()
        if run == "3_7_11":
            task = 1
        elif run == "4_8_12":
            task = 2
        elif run == "5_9_13":
            task = 3
        elif run == "6_10_14":
            task = 4
        else:
            task = run
        accuracy = self._get_best_accuracy()
        print(
            f"experiment {task}: subject {self.subject:03}: accuracy = {accuracy:.4f}"
        )
        return accuracy
