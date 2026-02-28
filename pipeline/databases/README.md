# Data Collection - Databases

## Description
Introduction to relational (MySQL) and non-relational (MongoDB) databases
for storing and processing Machine Learning pipeline data.

## Requirements

### MySQL
- Ubuntu 16.04/18.04
- MySQL 5.7.30
- All SQL keywords in UPPERCASE
- Comments before each query

### MongoDB
- Ubuntu 16.04/18.04
- MongoDB 4.2
- PyMongo 3.10
- First line of Mongo files: `// my comment`

### Python
- Python 3.5 or 3.7
- pycodestyle 2.5
- PyMongo 3.10

## Installation

### MySQL
```bash
sudo apt-get install mysql-server
service mysql start
# credentials: root/root
```

### MongoDB
```bash
wget -qO - https://www.mongodb.org/static/pgp/server-4.2.asc | apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu bionic/mongodb-org/4.2 multiverse" > /etc/apt/sources.list.d/mongodb-org-4.2.list
sudo apt-get update
sudo apt-get install -y mongodb-org
service mongod start
pip3 install pymongo
```

## Files
| File | Description |
|------|-------------|
| `0-create_database_if_missing.sql` | Creates database if it doesn't exist |
| `1-first_table.sql` | Creates first table |
| `2-list_values.sql` | Lists all rows of a table |
| ... | ... |