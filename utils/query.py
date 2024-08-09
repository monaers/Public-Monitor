import psycopg2
from psycopg2 import *
conn = psycopg2.connect(dbname='postgres',host='localhost',port=5432,user='postgres',password='zhizhuxia0505')
cursor = conn.cursor()
def query(sql, params, type='no_select'):
    try:
        cursor.execute(sql, params)
        if type == 'select_one':

            row = cursor.fetchone()
            if row:

                return dict(zip([col_name[0] for col_name in cursor.description], row))
            else:
                return None
        elif type != 'no_select':

            rows = cursor.fetchall()
            data_list = [dict(zip([col_name[0] for col_name in cursor.description], row)) for row in rows]
            return data_list
        else:

            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        print("finished ")
