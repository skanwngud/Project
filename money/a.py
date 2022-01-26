import mariadb
import pandas as pd

port = 3306
user = 'root'
passwd = 'vhfoq'

conn = mariadb.connect(
    user=user, password=passwd,
    port=port
)
cur = conn.cursor()

