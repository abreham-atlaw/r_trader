import os
import typing


class FileSyncer:

	def __init__(self):
		pass

	def __list_files(self, path):
		return [os.path.join(path, filename) for filename in os.listdir(path)]

	def __get_unmatched_files(self, paths: typing.List[str]) -> typing.List[str]:

		print(f"[+]Checking {len(paths)} Paths...")
		unmatched_files = []
		checked_paths = []
		for path1 in paths:
			for path2 in paths:
				if path1 == path2 or (path1, path2) in checked_paths or (path2, path1) in checked_paths:
					continue
				print(f"Comparing {path1} and {path2}...")
				files = self.__list_files(path1) + self.__list_files(path2)
				filenames = [os.path.basename(file) for file in files]
				for file in files:
					if filenames.count(os.path.basename(file)) < 2:
						unmatched_files.append(file)
				checked_paths.append((path1, path2))
		return unmatched_files

	def __sync_files(self, files: typing.List[str]):
		for file in files:
			print(f"[+]Removing {file}...")
			os.system(f"rm {os.path.abspath(file)}")

	def sync(self, paths: typing.List[str]):

		unmatched_files = self.__get_unmatched_files(paths)

		print(f"Found {len(unmatched_files)} Unmatched Files")
		print('\n'.join(unmatched_files))

		self.__sync_files(unmatched_files)
