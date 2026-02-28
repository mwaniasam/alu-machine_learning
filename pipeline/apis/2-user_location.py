#!/usr/bin/env python3
"""Module for fetching a GitHub user's location"""
import requests
import sys
import time
import math


if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        reset_time = int(response.headers['X-Ratelimit-Reset'])
        minutes = math.ceil((reset_time - int(time.time())) / 60)
        print('Reset in {} min'.format(minutes))
    elif response.status_code == 200:
        data = response.json()
        location = data.get('location')
        if location:
            print(location)
        else:
            print('Not found')
    else:
        print('Not found')
