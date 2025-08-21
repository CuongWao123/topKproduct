docker exec -it kafka /opt/kafka/bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --topic topic1 --partitions 3 --replication-factor 1

docker exec -it kafka /opt/kafka/bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic topic1

docker exec -it kafka /opt/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic topic1 --from-beginning


