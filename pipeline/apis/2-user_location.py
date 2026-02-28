#!/usr/bin/env python3
"""Module for fetching a GitHub user's location"""
import requests
import sys
import time


if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 404:
        print('Not found')
    elif response.status_code == 403:
        reset_time = int(response.headers['X-Ratelimit-Reset'])
        minutes = (reset_time - int(time.time())) // 60
        print('Reset in {} min'.format(minutes))
    else:
        data = response.json()
        print(data.get('location'))
