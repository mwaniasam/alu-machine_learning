#!/usr/bin/env python3
"""Module for fetching the upcoming SpaceX launch"""
import requests


if __name__ == '__main__':
    url = 'https://api.spacexdata.com/v5/launches/upcoming'
    response = requests.get(url)
    launches = response.json()

    upcoming = min(launches, key=lambda x: x['date_unix'])

    name = upcoming['name']
    date = upcoming['date_local']

    rocket_url = 'https://api.spacexdata.com/v4/rockets/{}'.format(
        upcoming['rocket']
    )
    rocket = requests.get(rocket_url).json()
    rocket_name = rocket['name']

    pad_url = 'https://api.spacexdata.com/v4/launchpads/{}'.format(
        upcoming['launchpad']
    )
    pad = requests.get(pad_url).json()
    pad_name = pad['name']
    pad_locality = pad['locality']

    print('{} ({}) {} - {} ({})'.format(
        name, date, rocket_name, pad_name, pad_locality
    ))
