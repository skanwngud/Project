import pandas as pd
import mariadb
import sys

from db_conn import connect_db

host = connect_db.get('host')
port = connect_db.get('port')
user = connect_db.get('user')
passwd = connect_db.get('passwd')
database = connect_db.get('database')



try:
    conn = mariadb.connect(
        user=user, password=passwd,
        host=host, port=port, database=database,
        autocommit=False
    )
    print(f"DB connected! {host}:{port}, {user} - {database}")
except mariadb.Error as e:
    print(f"DB has Error : {e}")
    sys.exit(1)

cur = conn.cursor()


qurey = "select * from users;"

cur.execute(qurey)

row = cur.fetchall()
print(row)
# conn.commit()

cur.close()
conn.close()