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
NETWORK_TRIES = 20

LOGGING = True
LOGGING_PID = False
LOGGING_CONSOLE = True
LOGGING_FILE_PATH = os.path.abspath("output.log")

MIN_FREE_MEMORY = 10
MAX_PROCESSES = 6
RECURSION_DEPTH = 10000
NESTED_PROCESS = False
MAIN_PID = os.getpid()

OPTIMIZER_PG_CONFIG = {
	"dsn": "postgres://ontiwpwwgbtgwp:8702c0dec88af3c49473d464bf44e8ad17419facfce764c8684ed540839fb8cb@ec2-34-194-100-156.compute-1.amazonaws.com:5432/dcs4e3sfc908fi",
	"sslmode": "require"
}

DROPBOX_API_TOKEN = "QNheD1qJMugAAAAAAAAAAQfJuql5BxK2s2xo2MMZ8M2zGNRGZ9Xf2sWTyElklIPZ"
DROPBOX_FOLDER = "/RForexTrader"

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
	"password": "4U7z7KJM",  # TODO
	"port": "5432"
}


# LIVE ENVIRONMENT
OANDA_TOKEN = "4e3bc058fee3b2005e2a651081da881e-1cc2b5245cda5e61beb340aaf217c704"
OANDA_TRADING_URL = "https://api-fxpractice.oanda.com/v3"
OANDA_TRADING_ACCOUNT_ID = "101-001-19229086-002"
OANDA_TEST_ACCOUNT_ID = "101-001-19229086-002"
DEFAULT_TIME_IN_FORCE = "FOK"
TIMEZONE = timezone("Africa/Addis_Ababa")


# AGENT CONFIGS
MARKET_STATE_MEMORY = 73
TIME_PENALTY = 0
AGENT_TRADE_SIZE_GAP = 60
AGENT_DEPTH = 30    # TODO: DEPRECATED
AGENT_STATE_CHANGE_DELTA_MODEL_MODE = True
AGENT_STATE_CHANGE_DELTA_STATIC_BOUND = (0.00001, 0.0001)
AGENT_DISCOUNT_FACTOR = 0.7
AGENT_EXPLOIT_EXPLORE_TRADEOFF = 1
AGENT_UCT_EXPLORE_WEIGHT = 0.05
AGENT_LOGICAL_MCA = True
AGENT_STEP_TIME = 1*60
AGENT_MAX_INSTRUMENTS = 5
AGENT_RANDOM_SEED = random.randint(0, 1000)
AGENT_CURRENCY = "USD"
AGENT_CORE_PRICING = True
AGENT_COMMISSION_COST = 0.02  # IN AGENT_CURRENCY
AGENT_SPREAD_COST = 0.125/100  # IN AGENT_CURRENCY
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
	path=os.path.join(BASE_DIR, "res/core_model_wrapped.h5"),
	download=False
)

DELTA_MODEL_CONFIG = ModelConfig(
	id="delta",
	url="https://www.dropbox.com/s/axr09n3xbbaqvpb/model.h5?dl=0",
	path=os.path.join(BASE_DIR, "res/delta_model_wrapped.h5"),
	download=False
)

PREDICTION_MODELS = [
	CORE_MODEL_CONFIG,
	DELTA_MODEL_CONFIG
]
