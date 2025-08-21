import csv
import json
import time
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic_name = "topic1"

csv_file = '../data/processed/test_1_2016.csv'

batch_size = 5
batch = []

with open(csv_file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f) 
    
    for row in reader:
        batch.append(row)

        if len(batch) == batch_size:
            for rec in batch:
                producer.send(topic_name, value=rec)
                print(f"Sent: {rec}")

            batch = []  
            time.sleep(10)

if batch:
    for rec in batch:
        producer.send(topic_name, value=rec)
        print(f"Sent (remaining): {rec}")

producer.flush()
producer.close()
