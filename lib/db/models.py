from typing import *

from psycopg2 import ProgrammingError

from lib.db.db import connection

print(f"Importing connection: {connection}")


class Model:

	__pk__ = None
	__tablename__ = None
	__columns__ = None

	def delete(self):
		cls = self.__class__
		cls.execute_query(
			f"DELETE FROM {cls.get_tablename()} WHERE {cls.get_pk()} = %s",
			(self.__get_pk(),),
		)

	def insert(self):
		cls = self.__class__
		involved_columns = [c for c in cls.get_columns() if c != cls.get_pk() or self.__get_pk() is not None]

		pk = cls.execute_query(
			f"INSERT INTO {cls.get_tablename()}"
			f"({','.join(involved_columns)}) "
			f"values({','.join(['%s']*len(involved_columns))}) "
			f"RETURNING {cls.get_pk()}",
			tuple([self.__get_property(column) for column in involved_columns]),
			single=True,
			deserialize=False
		)[0]
		self.__set_property(cls.get_pk(), pk)

	def update(self):
		cls = self.__class__
		cls.execute_query(
			f"UPDATE {cls.get_tablename()} "
			f"SET {','.join([f'{column} = %s' for column in cls.get_columns()])} "
			f"WHERE {cls.get_pk()}=%s",
			(
				*[self.__get_property(column) for column in cls.get_columns()],
				self.__get_pk()
			),
		)

	def save(self):
		if not self.__class__.get_with_pk(self.__get_pk()):
			self.insert()
		else:
			self.update()

	def __get_pk(self):
		return self.__get_property(self.__class__.get_pk())

	def __get_property(self, name):
		return self.__dict__.get(name)

	def __set_property(self, name, value):
		self.__dict__[name] = value

	@classmethod
	def get_initilization_map(cls):
		return {column: column for column in cls.get_columns()}

	@classmethod
	def get_initilization_key(cls, column):
		return cls.get_initization_map().get(columns)

	@classmethod
	def get_column_init_key(cls, init_key):
		for column, key in cls.get_initilization_map():
			if key == init_key:
				return column
	#TODO USE KWARGS FOR INITILIZATION USING THE COLUMN-INIT_KEY MAPPING ABOVE

	@classmethod
	def get_tablename(cls):
		return cls.__tablename__

	@classmethod
	def get_columns(cls):
		return cls.__columns__

	@classmethod
	def get_pk(cls):
		return cls.__pk__
	
	@classmethod
	def get_all(cls,):
		return cls.execute_query(
			query=f"SELECT {','.join(cls.get_columns())} FROM {cls.get_tablename()}",
			args=(),
			single=False
		)

	@classmethod
	def get_with_condition(cls, condition: str, args: Tuple, single=False):
		return cls.execute_query(
			f"SELECT {','.join(cls.get_columns())} FROM {cls.get_tablename()} WHERE {condition}",
			args,
			single=single
		)

	@classmethod
	def get_with_pk(cls, pk):
		return cls.get_with_condition(
			f"{cls.get_pk()} = %s",
			(pk,),
			single=True
		)

	@classmethod
	def execute_query(cls, query: str, args: Tuple, single=False, deserialize=True):
		print(f"[+]Executing query: {query} with args: {args}")
		cursor = connection.cursor()

		cursor.execute(
			query, args
		)

		connection.commit()

		try:
			if single:
				result = cursor.fetchone()
			else:
				result = cursor.fetchall()
		except ProgrammingError:
			return

		if deserialize:
			return cls._deserialize(result)
		return result

	@classmethod
	def _deserialize(cls, response):
		if response is None:
			return None
		if type(response) == list:
			return [cls(*value) for value in response]
		return cls(*response)
