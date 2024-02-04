# import pandas as pd
# import mysql.connector

# # MySQL connection details
# address = "localhost"
# port = 3306
# database = "laveraluser"
# table = "testdata2s"

# # Establish a connection to the MySQL server



# def fetch_datas(address, port, database, table , user, password):

#     connection = mysql.connector.connect(
#         host=address,
#         port=port,
#         database=database,
#         user=user,
#         password=password)
#     # Fetch all data from the table
#     query = f"SELECT id, boardID, data1 AS irms, data2 AS watt, data3 AS kwh, created_at AS Datetime FROM {table}"
#     data = pd.read_sql(query, connection)
#     # Close the MySQL connection
#     connection.close()

#     return data


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

def fetch_datas(address, port, database, table , user, password):

    # Create an engine that connects to the MySQL database
    engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{address}:{port}/{database}')

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)

    # Create a Session
    session = Session()

    # Fetch all data from the table
    query = f"SELECT id, boardID, data1 AS irms, data2 AS watt, data3 AS kwh, created_at AS Datetime FROM {table} ORDER BY created_at DESC LIMIT 86400"
    data = pd.read_sql(query, session.bind)

    return data

def fetch_datas_last_24(address, port, database, table , user, password):

    # Create an engine that connects to the MySQL database
    engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{address}:{port}/{database}')

    # Create a configured "Session" class
    Session = sessionmaker(bind=engine)

    # Create a Session
    session = Session()

    # Fetch all data from the table
    query = f"SELECT id, boardID, data1 AS irms, data2 AS watt, data3 AS kwh, created_at AS Datetime FROM {table} WHERE created_at >= NOW() - INTERVAL 1 DAY"
    data = pd.read_sql(query, session.bind)

    return data