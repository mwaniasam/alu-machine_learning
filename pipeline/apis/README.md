# Data Collection - APIs

## Description
Scripts for collecting and processing data from external APIs using the Python `requests` package as part of a Machine Learning pipeline.

## Learning Objectives
- Make HTTP GET requests using the `requests` package
- Handle rate limiting and pagination
- Fetch and manipulate JSON data from external services

## Requirements
- Python 3.5 (Ubuntu 16.04 LTS)
- `requests` package
- pycodestyle 2.4

## Installation
```bash
pip install requests
```

## Files
| File | Description |
|------|-------------|
| `0-passengers.py` | Fetches ships that can hold a given number of passengers |
| `1-sentience.py` | Fetches list of home planets of all sentient species |
| `2-schedule.py` | Displays the launch schedule of upcoming SpaceX rockets |

## Usage
```bash
./0-passengers.py <number_of_passengers>
./1-sentience.py
./2-schedule.py
```

## Author
Holberton School Machine Learning Pipeline Project