from BCITrainer import BCITrainer
from BCIPredictor import BCIPredictor
import time

import argparse


def train_models_in_batch(tasks: dict, subjects: list[int]) -> None:
    """
    Train models in batch

    Parameters
    ----------
    tasks : dict
        Dictionary of tasks.
    subjects : list[int]
        List of subjects.

    Throws
    ------
    Exception
        If an error occurs while training a model.

    Returns
    -------
    None
    """
    for task in tasks:
        for subject in subjects:
            try:
                bci_trainer = None
                if task % 2 == 0:
                    bci_trainer = BCITrainer(subject, [], tasks[task])
                else:
                    bci_trainer = BCITrainer(subject, tasks[task], [])
                bci_trainer.train()
            except Exception as e:
                print(e)
                pass


def stream_predictions_in_batch(
    tasks: dict, subjects: list[int], kafka_server: str, topic: str
) -> None:
    """
    Stream predictions in batch

    Parameters
    ----------
    tasks : dict
        Dictionary of tasks.
    subjects : list[int]
        List of subjects.
    kafka_server : str
        Kafka server address.
    topic : str
        Kafka topic name.

    Throws
    ------
    Exception
        If an error occurs while streaming predictions.

    Returns
    -------
    None
    """
    for task in tasks:
        for subject in subjects:
            try:
                bci_predictor = None
                if task % 2 == 0:
                    bci_predictor = BCIPredictor(
                        subject, [], tasks[task], kafka_server, topic
                    )
                else:
                    bci_predictor = BCIPredictor(
                        subject, tasks[task], [], kafka_server, topic
                    )
                bci_predictor.stream_predictions()
                time.sleep(1)
            except Exception as e:
                print(e)
                pass


def print_stats(
    tasks: dict, subjects: list[int], kafka_server: str, topic: str
) -> None:
    """
    Print stats

    Parameters
    ----------
    tasks : dict
        Dictionary of tasks.
    subjects : list[int]
        List of subjects.
    kafka_server : str
        Kafka server address.
    topic : str
        Kafka topic name.

    Throws
    ------
    Exception
        If an error occurs while printing stats.

    Returns
    -------
    None
    """
    task_accuracy_sum = {task: 0 for task in tasks}
    subject_count = {task: 0 for task in tasks}
    for task in tasks:
        for subject in subjects:
            try:
                bci_predictor = None
                if task % 2 == 0:
                    bci_predictor = BCIPredictor(
                        subject, [], tasks[task], kafka_server, topic
                    )
                else:
                    bci_predictor = BCIPredictor(
                        subject, tasks[task], [], kafka_server, topic
                    )
                task_accuracy_sum[task] += bci_predictor.print_accuracy()
                subject_count[task] += 1
            except Exception as e:
                print(e)
                pass

    print(
        f"\nMean accuracy of the {len(tasks)} different experiments for all {len(subjects)} subjects:"
    )
    task_accuracy_avg = {
        task: task_accuracy_sum[task] / subject_count[task]
        if subject_count[task] > 0
        else 0
        for task in tasks
    }
    for task, avg_accuracy in task_accuracy_avg.items():
        print(f"experiment {task}: accuracy = {avg_accuracy:.4f}")

    print(
        f"\nMean accuracy of {len(tasks)} experiments: {sum(task_accuracy_avg.values()) / len(task_accuracy_avg):.4f}"
    )


def train_model(task: int, subject: int) -> None:
    """
    Train model

    Parameters
    ----------
    task : int
        Task number.
    subject : int
        Subject number.

    Throws
    ------
    Exception
        If an error occurs while training a model.

    Returns
    -------
    None
    """
    runs_motor = [3, 5, 7, 9, 11, 13]
    runs_imaginary = [4, 6, 8, 10, 12, 14]
    try:
        if task in runs_motor:
            bci_trainer = BCITrainer(subject, [task], [])
        elif task in runs_imaginary:
            bci_trainer = BCITrainer(subject, [], [task])
        else:
            raise ValueError("Task must be a motor or imaginary task. (3-14)")
        bci_trainer.train()
    except Exception as e:
        raise e


def stream_predictions(task: int, subject: int, kafka_server: str, topic: str) -> None:
    """
    Stream predictions

    Parameters
    ----------
    task : int
        Task number.
    subject : int
        Subject number.
    kafka_server : str
        Kafka server address.
    topic : str
        Kafka topic name.

    Raises
    ------
    Exception
        If an error occurs while streaming predictions.

    Returns
    -------
    None
    """
    runs_motor = [3, 5, 7, 9, 11, 13]
    runs_imaginary = [4, 6, 8, 10, 12, 14]
    try:
        if task in runs_motor:
            bci_predictor = BCIPredictor(subject, [task], [], kafka_server, topic)
        elif task in runs_imaginary:
            bci_predictor = BCIPredictor(subject, [], [task], kafka_server, topic)
        else:
            raise ValueError("Task must be a motor or imaginary task. (3-14)")
        bci_predictor.stream_predictions()
    except Exception as e:
        raise e


if __name__ == "__main__":
    """
    Main function

    Parameters
    ----------
    action : str
        Action to perform (train, predict, train_all, stats, predict_all)
    task : int
        Task number (3-14)
    subject : int
        Subject number (1-109)

    Returns
    -------
    None
    """
    kafka_server = "localhost:9093"
    topic = "eeg_data_topic"
    all_subjects = list(range(1, 110))
    all_tasks = {
        1: [3, 7, 11],  # motor/fists
        2: [4, 8, 12],  # imagine/fists
        3: [5, 9, 13],  # motor/feet
        4: [6, 10, 14],  # imagine/feet
    }

    parser = argparse.ArgumentParser(description="BCI Training and Prediction Script")
    parser.add_argument(
        "action",
        type=str,
        choices=["train", "predict", "train_all", "stats", "predict_all"],
        help="Action to perform (train, predict, train_all, stats, predict_all)",
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=range(3, 15),
        help="Task number (3-14)",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--subject",
        type=int,
        choices=range(1, 110),
        help="Subject number (1-109)",
        nargs="?",
        default=None,
    )
    args = parser.parse_args()

    if args.action in ["train", "predict"] and (
        args.task is None or args.subject is None
    ):
        parser.error(
            "The 'train' and 'predict' actions require both --task and --subject arguments."
        )
    elif args.action in ["predict_all"] and args.subject is None:
        parser.error("The 'predict_all' actions require the --subject argument.")

    if args.action == "train":
        try:
            train_model(args.task, args.subject)
        except Exception as e:
            print(f"Error while training model: {e}")
    elif args.action == "predict":
        try:
            stream_predictions(args.task, args.subject, kafka_server, topic)
        except Exception as e:
            print(f"Error while streaming prediction: {e}")
    elif args.action == "train_all":
        try:
            train_models_in_batch(all_tasks, all_subjects)
        except Exception as e:
            print(f"Error while training models: {e}")
    elif args.action == "stats":
        print_stats(all_tasks, all_subjects, kafka_server, topic)
    elif args.action == "predict_all":
        try:
            stream_predictions_in_batch(all_tasks, [args.subject], kafka_server, topic)
        except Exception as e:
            print(f"Error while streaming predictions: {e}")
