import cattr
from datetime import datetime


TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

cattr.register_unstructure_hook(
    datetime, lambda dt: datetime.strftime(dt, TIME_FORMAT)
)
cattr.register_structure_hook(
    datetime, lambda dt_str, _: datetime.strptime(dt_str.split(".")[0], TIME_FORMAT)
)
