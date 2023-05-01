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

MONGODB_URL ="mongodb+srv://abreham:zYUir15jnOcrPqg1@cluster0.vn0ngnn.mongodb.net/?retryWrites=true&w=majority"

OPTIMIZER_PG_CONFIG = {
	"dsn": "postgres://ontiwpwwgbtgwp:8702c0dec88af3c49473d464bf44e8ad17419facfce764c8684ed540839fb8cb@ec2-34-194-100-156.compute-1.amazonaws.com:5432/dcs4e3sfc908fi",
	"sslmode": "require"
}

DROPBOX_API_TOKEN = "QNheD1qJMugAAAAAAAAAAQfJuql5BxK2s2xo2MMZ8M2zGNRGZ9Xf2sWTyElklIPZ"
DROPBOX_FOLDER = "/RForexTrader"

# PCLOUD_API_TOKEN = "wiaJJZfKQB7ZFJGbF2LjLo0nFysSdWCN0mklLXYV"  # main
# PCLOUD_API_TOKEN = "NbYQ47Z6O9B7ZsjyeitXJhUpmug9Cg4qS8m40yXYk"  # 0
PCLOUD_API_TOKEN = "1Qbjq7ZIO9B7ZzfX5wncB5G7ebGSYi95oiVmjFkky" # 1
PCLOUD_FOLDER = "/Apps/RTrader"

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
UPDATE_AGENT = False
MARKET_STATE_MEMORY = 73
MARKET_STATE_GRANULARITY = "M1"
TIME_PENALTY = 0
AGENT_TRADE_SIZE_GAP = 70
AGENT_DEPTH = 30    # TODO: DEPRECATED
AGENT_STATE_CHANGE_DELTA_MODEL_MODE = True
AGENT_STATE_CHANGE_DELTA_STATIC_BOUND = (0.00001, 0.0001)
AGENT_DISCOUNT_FACTOR = 1
AGENT_DISCOUNT_FUNCTION = None
AGENT_EXPLOIT_EXPLORE_TRADEOFF = 1
AGENT_UCT_EXPLORE_WEIGHT = 0.1
AGENT_LOGICAL_MCA = True
AGENT_STEP_TIME = 1*60
AGENT_MAX_INSTRUMENTS = 2
AGENT_USE_STATIC_INSTRUMENTS = False
AGENT_STATIC_INSTRUMENTS = [
			("AUD", "USD"),
			("USD", "CHF"),
			("USD", "SEK")
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
AGENT_DEPTH_MODE = True
AGENT_PROBABILITY_CORRECTION = True
AGENT_ARBITRAGE_ZONE_SIZE = 0.02
AGENT_ARBITRAGE_ZONE_GUARANTEE_PERCENT = 0.00
AGENT_ARBITRAGE_BASE_MARGIN = 10

MC_WORKER_STEP_TIME = 0.05*60
MC_WORKERS = 4
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
	path=os.path.join(BASE_DIR, "res/m10/combined_trained/core_model_d.h5"),
	download=False
)

DELTA_MODEL_CONFIG = ModelConfig(
	id="delta",
	url="https://www.dropbox.com/s/io0fbl7m44e6k8a/delta-bb_wrapped.h5?dl=0",
	path=os.path.join(BASE_DIR, "res/m10/combined_trained/delta_model_d.h5"),
	download=False
)

PREDICTION_MODELS = [
	CORE_MODEL_CONFIG,
	DELTA_MODEL_CONFIG
]
