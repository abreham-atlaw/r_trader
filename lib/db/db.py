import psycopg2 as pg

connection = None

def initialize_connection(
		db_name,
		db_user,
		db_password,
		db_host,
		db_port
	):
	
	global connection
	print("[+]Initializing DB connection...")
	connection = pg.connect(	
		dbname=db_name,
		user=db_user,
		password=db_password,
		host=db_host,
		port=db_port
		)
	print(f"[+]DB Connection Initialized: {connection}")

	


