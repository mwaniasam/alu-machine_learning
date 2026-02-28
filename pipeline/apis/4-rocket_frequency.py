#!/usr/bin/env python3
"""Module for displaying the number of launches per rocket"""
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v4/launches'
    launches = requests.get(url).json()

    rocket_counts = {}
    for launch in launches:
        rocket_id = launch['rocket']
        rocket_counts[rocket_id] = rocket_counts.get(rocket_id, 0) + 1

    rocket_names = {}
    for rocket_id in rocket_counts:
        url = 'https://api.spacexdata.com/v4/rockets/{}'.format(rocket_id)
        rocket = requests.get(url).json()
        rocket_names[rocket_id] = rocket['name']

    sorted_rockets = sorted(
        rocket_counts.items(),
        key=lambda x: (-x[1], rocket_names[x[0]])
    )

    for rocket_id, count in sorted_rockets:
        print('{}: {}'.format(rocket_names[rocket_id], count))
