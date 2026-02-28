#!/usr/bin/env python3
"""Module for finding schools by topic in MongoDB"""


def schools_by_topic(mongo_collection, topic):
    """Returns the list of schools having a specific topic"""
    return list(mongo_collection.find({"topics": topic}))
