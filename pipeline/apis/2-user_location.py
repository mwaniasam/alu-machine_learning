#!/usr/bin/env python3
"""Module for fetching a GitHub user's location"""
import requests
import sys
import time
import math


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)

    url = sys.argv[1]

    try:
        response = requests.get(url)

        if response.status_code == 403:
            reset_time = int(response.headers.get('X-Ratelimit-Reset', 0))
            minutes = math.ceil((reset_time - int(time.time())) / 60)
            print('Reset in {} min'.format(minutes))
        elif response.status_code == 404:
            print('Not found')
        elif response.status_code == 200:
            data = response.json()
            location = data.get('location')
            if location is None or location == '':
                print('Not found')
            else:
                print(location)
        else:
            print('Not found')
    except Exception:
        print('Not found')
