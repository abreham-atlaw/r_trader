import psycopg2 as pg

from .Config import DEFAULT_PG_CONFIG

try:
	pg_connection = pg.connect(**DEFAULT_PG_CONFIG)
except:
	print("Couldn't Connect to Postgres")
