#!/usr/bin/env python3
"""Module for providing stats about Nginx logs stored in MongoDB"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx

    total = collection.find().count()
    print("{} logs".format(total))

    print("Methods:")
    for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
        count = collection.find({"method": method}).count()
        print("\tmethod {}: {}".format(method, count))

    status = collection.find({"method": "GET", "path": "/status"}).count()
    print("{} status check".format(status))
