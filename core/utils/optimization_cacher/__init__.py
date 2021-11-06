import psycopg2 as pg

from core.Config import OPTIMIZER_PG_CONFIG

connection = pg.connect(**OPTIMIZER_PG_CONFIG)

read_cursor = connection.cursor()