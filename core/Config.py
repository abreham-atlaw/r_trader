import json

import numpy as np
from pytz import timezone
import os
import random
from dataclasses import dataclass


@dataclass
class ModelConfig:
	id: str
	download: bool
	url: str
	path: str


BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

REMOTE_TRADER_URL = "http://localhost:8080/"
NETWORK_TRIES = 10

LOGGING = True
LOGGING_PID = True
LOGGING_CONSOLE = True
LOGGING_FILE_PATH = os.path.abspath("output.log")

MC_SERVER_PORT = 8000
MC_SERVER_URL = "http://127.0.0.1:%s" % (MC_SERVER_PORT,)

MIN_FREE_MEMORY = 10
MAX_PROCESSES = 6
RECURSION_DEPTH = 10000
NESTED_PROCESS = False
MAIN_PID = os.getpid()

MONGODB_URL = "mongodb+srv://abreham:zYUir15jnOcrPqg1@cluster0.vn0ngnn.mongodb.net/?retryWrites=true&w=majority"

OPTIMIZER_PG_CONFIG = {
	"dsn": "postgres://ontiwpwwgbtgwp:8702c0dec88af3c49473d464bf44e8ad17419facfce764c8684ed540839fb8cb@ec2-34-194-100-156.compute-1.amazonaws.com:5432/dcs4e3sfc908fi",
	"sslmode": "require"
}

DROPBOX_API_TOKEN = "sl.BrBqqSAAzJsdIo9x1NG-z51HwPzc6yw0KCrbQ7hn76DvI0vItMEhXBDiqW-W4yGzHzCHL1RVBEeFeKEEdj7EIIjorHfLqOeM9KEhanIYOynP3XGiFRi_4RK3MGATuFHz50iZUFezZ09Uwd6Uk4LC3Oc"
DROPBOX_FOLDER = "/RForexTrader"

# PCLOUD_API_TOKEN = "wiaJJZfKQB7ZFJGbF2LjLo0nFysSdWCN0mklLXYV"  # main
# PCLOUD_API_TOKEN = "NbYQ47Z6O9B7ZsjyeitXJhUpmug9Cg4qS8m40yXYk"  # 0
# PCLOUD_API_TOKEN = "1Qbjq7ZIO9B7ZzfX5wncB5G7ebGSYi95oiVmjFkky" # 1

PCLOUD_TOKENS = [
	# "XF5eu7ZfKQB7ZeXVhxX95vdV8zY123vs5gfnCUIbX",  # abrishatlaw@gmail.com +
	"7oUGTVZ6O9B7ZJB3ewVjpOnz8zSLT285MIV1ejtPk",  # abrishatlaw@yahoo.com +
	"KRVdTVZIO9B7Z1UeR69vE4XjjmPrlrElk3u8cKrby",  # abreham.atlaw@yahoo.com -
	"qSyuwZxDks7ZFndAH7ULFFjXkqoazz0r5BUlEd57",  # abrehamatlaw@outlook.com -
	# "lyJXAkZHDks7Z4w79whbTSVhssQ85JevC1QMEkoGk",  # abreham.atlaw@outlook.com +
	# "1xjpt7ZEWJs7ZcfiRorgfUDQMbJsY2QV1h0whI5ek",  # abreham-atlaw@outlook.com -
	# "CtEWXXZnvzs7Z8rc9rNJgHDQS6xh53cB8uy0hvhty",  # abreham_atlaw@outlook.com +
	# "51pub7ZkqQs7Z0HsMuiQ78i4HGbzAlNXIkJtNdvX0",  # abrehama@outlook.com +
	# "2V9aqXZiyRs7ZjXdUChjbQJkh9C7UjG76K73UbH1V",  # abreham.a@outlook.com +
	# "gEVq3kZPR4s7ZmdNgKxMPooQ9IKVwb8XgMyVYibuV",  # abreham_a@outlook.com +
	# "aQXg0kZkqQs7ZPjSXBAcaVeFixxH2SvvitBMCMnrk",  # abrehama@outlook.com +
	# "RC4By7ZPays7ZgYcLvQzFDPfjMNRnQzHGshbX040y",  # hiwotahab12@gmail.com +
	# "47ro2VZXe7s7ZwCNAS9a05du6xHUO9IHPrS8Jt1a7",  # abrehamatlaw321@gmail.com -
	# "LmwLc7ZktF97Zv62oq2MHiPJ4tL3VdgrcLj90O3DV"  # abrehamalemu@outlook.com
]

PCLOUD_API_TOKEN = "jfAYHkZfKQB7Zn0vw75zQgU82511XehVaVjc2zSRV"

PCLOUD_FOLDER = "/Apps/RTrader"
MODEL_PCLOUD_FOLDER = os.path.join(PCLOUD_FOLDER, "Models/10M/10MA")
CHECKPOINT_PCLOUD_FOLDER = os.path.join(PCLOUD_FOLDER, "Checkpoints/10M/10MA")

POLYGON_API_KEY = "1ijeQ0XUYNl1YMHy6Wl_5zEBtGbkipUP"

# ENVIRONMENT CONFIGS

# TRAINING ENVIRONMENT
HISTORICAL_TABLE_NAME = "test_currencies_history"
CURRENCIES_TABLE_NAME = "test_currencies"
TRADEABLE_PAIRS_TABLE_NAME = "test_tradeable_pairs"
DISTINCT_DATETIMES_TABLE_NAME = "test_distinct_datetimes"
"""
HISTORICAL_TABLE_NAME = "currencies_history"
CURRENCIES_TABLE_NAME = "currencies"
TRADEABLE_PAIRS_TABLE_NAME = "tradeable_pairs"
DISTINCT_DATETIMES_TABLE_NAME = "distinct_datetimes"
"""
DEFAULT_PG_CONFIG = {
	"host": "localhost",
	"database": "rtrader_db",
	"user": "rtrader_admin",
	"password": "4U7z7KJM"  # TODO
}

# LIVE ENVIRONMENT
OANDA_TOKEN = "4e3bc058fee3b2005e2a651081da881e-1cc2b5245cda5e61beb340aaf217c704"
OANDA_TRADING_URL = "https://api-fxpractice.oanda.com/v3"
OANDA_TRADING_ACCOUNT_ID = "101-001-19229086-002"
OANDA_TEST_ACCOUNT_ID = "101-001-19229086-002"

DEFAULT_TIME_IN_FORCE = "FOK"
TIMEZONE = timezone("Africa/Addis_Ababa")

# AGENT CONFIGS
UPDATE_AGENT = True
UPDATE_EXPORT_BATCH_SIZE = 2
UPDATE_SAVE_PATH = os.path.join(BASE_DIR, "temp/Data/drmca_export")
UPDATE_TRAIN = False
MARKET_STATE_MEMORY = 1033
MARKET_STATE_GRANULARITY = "M5"
DUMP_CANDLESTICKS_PATH = os.path.join(BASE_DIR, "temp/candlesticks/real")
TIME_PENALTY = 0
AGENT_TRADE_SIZE_GAP = 70
AGENT_DEPTH = 30  # TODO: DEPRECATED
AGENT_STATE_CHANGE_DELTA_MODEL_MODE = False
AGENT_MIN_PROBABILITY = 1e-6
AGENT_DISCOUNT_FACTOR = 1
AGENT_DISCOUNT_FUNCTION = None
AGENT_EXPLOIT_EXPLORE_TRADEOFF = 1
AGENT_UCT_EXPLORE_WEIGHT = 0.7
AGENT_LOGICAL_MCA = True
AGENT_FRICTION_TIME = 6
AGENT_STEP_TIME = (2 * 60) - AGENT_FRICTION_TIME
AGENT_MAX_INSTRUMENTS = 2
AGENT_USE_STATIC_INSTRUMENTS = True
AGENT_STATIC_INSTRUMENTS = [
	("AUD", "USD"),
]
AGENT_RANDOM_SEED = random.randint(0, 1000000)
AGENT_CURRENCY = "USD"
AGENT_CORE_PRICING = False
AGENT_COMMISSION_COST = 0.05  # IN AGENT_CURRENCY
AGENT_SPREAD_COST = 0.05  # IN AGENT_CURRENCY
AGENT_STM = False
AGENT_STM_THRESHOLD = 0.05
AGENT_STM_BALANCE_TOLERANCE = 5
AGENT_STM_SIZE = 5
AGENT_STM_AVERAGE_WINDOW_SIZE = 10
AGENT_STM_ATTENTION_MODE = False
AGENT_DEPTH_MODE = False
AGENT_PROBABILITY_CORRECTION = True
AGENT_ARBITRAGE_ZONE_SIZE = 0.02
AGENT_ARBITRAGE_ZONE_GUARANTEE_PERCENT = 0.00
AGENT_ARBITRAGE_BASE_MARGIN = 10
AGENT_MAX_OPEN_TRADES = 20
AGENT_NUM_ACTIONS = 20
AGENT_RECOMMENDATION_PERCENT = 0.5
AGENT_DEVICE = "cpu"
AGENT_USE_SOFTMAX = False
AGENT_MA_WINDOW_SIZE = 10
AGENT_CRA_SIZE = 5
AGENT_CRA_DISCOUNT = 0.7
AGENT_DRMCA_WP = 100
AGENT_TOP_K_NODES = 5
AGENT_DUMP_NODES = True
AGENT_DUMP_NODES_PATH = os.path.join(BASE_DIR, "temp/graph_dumps")
AGENT_DUMP_VISITED_ONLY = True
AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_EPSILON = 1e-5
with open(os.path.join(BASE_DIR, "res/bounds/01.json"), "r") as file:
	AGENT_STATE_CHANGE_DELTA_STATIC_BOUND = sorted(list(json.load(file)))
with open(os.path.join(BASE_DIR, "res/weights/01.json"), "r") as file:
	AGENT_STATE_CHANGE_DELTA_STATIC_BOUND_WEIGHTS = sorted(list(json.load(file)))
MODEL_SAVE_EXTENSION = "zip"

MC_WORKER_STEP_TIME = 1 * 60
MC_WORKERS = 8
CURRENCIES = [
	"AUD",
	"CAD",
	"CHF",
	"CZK",
	"DKK",
	"GBP",
	"HKD",
	"HUF",
	"JPY",
	"MXN",
	"NOK",
	"NZD",
	"PLN",
	"SEK",
	"SGD",
	"THB",
	"TRY",
	"USD",
	"ZAR"
]

CORE_MODEL_CONFIG = ModelConfig(
	id="core",
	url="https://www.dropbox.com/s/9nvcas994dpzq3a/model.h5?dl=0&raw=0",
	path="/home/abrehamatlaw/Downloads/Compressed/albertcamus0-rtrader-training-cnn-111-cum-0-it-0-tot.zip",
	download=False
)

DELTA_MODEL_CONFIG = ModelConfig(
	id="delta",
	url="https://www.dropbox.com/s/io0fbl7m44e6k8a/delta-bb_wrapped.h5?dl=0",
	path=os.path.join(BASE_DIR, "res/m10/combined_trained/delta_model_d.h5"),
	download=False
)

ARA_MODEL_CONFIG = ModelConfig(
	id="ara",
	url="",
	path=None,
	download=False
)

PREDICTION_MODELS = [
	CORE_MODEL_CONFIG,
	DELTA_MODEL_CONFIG,
	ARA_MODEL_CONFIG
]

WEIGHTED_MSE_ALPHA = 1e-3


class ResourceCategories:

	TEST_RESOURCE = "test"
	RUNNER_STAT = "runner_stats"
