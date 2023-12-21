from kafka import KafkaProducer, KafkaConsumer
import json
import time


class KafkaStreamer:
    """
    KafkaStreamer is a class that handles sending and receiving data from Kafka.

    Attributes
    ----------
    kafka_server : str
        Kafka server address
    topic : str
        Kafka topic name
    producer : KafkaProducer
        Kafka producer object
    consumer : KafkaConsumer
        Kafka consumer object

    Methods
    -------
    send_data(data)
        Sends data to Kafka topic
    send_end_of_stream()
        Sends end of stream message to Kafka topic
    receive_data(callback)
        Receives data from Kafka topic
    """

    def __init__(self, kafka_server: str, topic: str):
        self.kafka_server = kafka_server
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_server],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[kafka_server],
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id=None,
        )

    def send_data(self, data: list) -> None:
        """
        Sends data to Kafka topic

        Parameters
        ----------
        data : list
            List of data to send

        Returns
        -------
        None
        """
        for message in data:
            self.producer.send(self.topic, value=message)
            time.sleep(0.5)
        print("Producer: No more data to send")
        self.send_end_of_stream()

    def send_end_of_stream(self) -> None:
        """
        Sends end of stream message to Kafka topic

        Returns
        -------
        None
        """
        end_of_stream_message = {"epoch": -1}
        self.producer.send(self.topic, value=end_of_stream_message)
        self.producer.flush()
        self.producer.close()
        print("Producer: End of stream")

    def receive_data(self, callback: callable) -> None:
        """
        Receives data from Kafka topic

        Parameters
        ----------
        callback : callable
            Callback function to process received data

        Returns
        -------
        None
        """
        for message in self.consumer:
            if message.value["epoch"] == -1:
                print("Consumer: End of stream message received")
                break
            callback(message)
