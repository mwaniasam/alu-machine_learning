#!/usr/bin/env python3
"""Module for fetching home planets of sentient species from the SWAPI API"""
import requests


def sentientPlanets():
    """Returns list of names of home planets of all sentient species"""
    planets = []
    url = 'https://swapi-api.hbtn.io/api/species/'

    while url:
        response = requests.get(url)
        data = response.json()
        for species in data['results']:
            if (species['designation'] == 'sentient' or
                    species['classification'] == 'sentient'):
                homeworld = species['homeworld']
                if homeworld:
                    planet_response = requests.get(homeworld)
                    planet_data = planet_response.json()
                    planets.append(planet_data['name'])
        url = data['next']

    return planets

