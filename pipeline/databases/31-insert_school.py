#!/usr/bin/env python3
"""Module for inserting a document in a MongoDB collection"""


def insert_school(mongo_collection, **kwargs):
    """Inserts a new document in a collection based on kwargs"""
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
