#!/usr/bin/env python3
"""Module for providing stats about Nginx logs stored in MongoDB"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')
    collection = client.logs.nginx

    total = collection.count_documents({})
    print("{} logs".format(total))

    print("Methods:")
    for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    status = collection.count_documents({"method": "GET", "path": "/status"})
    print("{} status check".format(status))
