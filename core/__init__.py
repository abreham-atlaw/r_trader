import psycopg2 as pg

from .Config import DEFAULT_PG_CONFIG
from lib.utils.logger import Logger

try:
	pg_connection = pg.connect(**DEFAULT_PG_CONFIG)
except:
	Logger.error("Couldn't Connect to Postgres")
