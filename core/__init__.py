import psycopg2 as pg

from .Config import DEFAULT_PG_CONFIG


pg_connection = pg.connect(**DEFAULT_PG_CONFIG)
