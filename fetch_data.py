import pandas as pd
import mysql.connector

# MySQL connection details
address = "localhost"
port = 3306
database = "laveraluser"
table = "testdata2s"

# Establish a connection to the MySQL server



def fetch_datas(address, port, database, table , user, password):

    connection = mysql.connector.connect(
        host=address,
        port=port,
        database=database,
        user=user,
        password=password)
    # Fetch all data from the table
    query = f"SELECT id, boardID, data1 AS irms, data2 AS watt, data3 AS kwh, created_at AS Datetime FROM {table}"
    data = pd.read_sql(query, connection)
    # Close the MySQL connection
    connection.close()

    return data




