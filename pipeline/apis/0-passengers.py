#!/usr/bin/env python3
"""Module for fetching available ships from the SWAPI API"""
import requests


def availableShips(passengerCount):
    """Returns list of ships that can hold a given number of passengers"""
    ships = []
    url = 'https://swapi-api.hbtn.io/api/starships/'

    while url:
        response = requests.get(url)
        data = response.json()
        for ship in data['results']:
            passengers = ship['passengers'].replace(',', '')
            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                pass
        url = data['next']

    return ships