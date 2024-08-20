import typing

import numpy as np

import os


class FileSizeCleaner:

	def __get_size(self, files: typing.List[str]) ->int:
		sizes = [np.load(file).shape[0] for file in files]
		return np.min(sizes)

	def start(self, files: typing.List[str], size=None, verbose=2):
		if verbose > 0:
			print(f"[+]Processing {len(files)} Files...")
		if size is None:
			size = self.__get_size(files)
		if verbose > 0:
			print(f"[+]Using Size {size}")
		removed_files = []
		updated_files = []
		clean_files = []
		for file in files:
			arr = np.load(file)
			file_size = arr.shape[0]
			if file_size == size:
				clean_files.append(file)
				continue
			if verbose > 1:
				print(f"\n\n[+]Irregular File Found: {file}(size: {file_size})")
			if file_size > size:
				if verbose > 1:
					print(f"Updating {file}")
					np.save(file, arr[:size])
				updated_files.append(file)
			else:
				if verbose > 1:
					print(f"Removing {file}")
				os.system(f"rm {os.path.abspath(file)}")
				removed_files.append(file)
		if verbose > 0:
			print(f"Removed Files: {len(removed_files)}")
			print(f"Updated Files: {len(updated_files)}")
			print(f"Clean Files: {len(clean_files)}")

		return removed_files, updated_files, clean_files
