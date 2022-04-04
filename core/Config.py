from pytz import timezone
import os
import random

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

REMOTE_TRADER_URL = "http://localhost:8080/"
NETWORK_TRIES = 10

LOGGING = True
LOGGING_PID = True
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
MARKET_STATE_MEMORY = 73
TIME_PENALTY = 0
AGENT_TRADE_SIZE_GAP = 40
AGENT_DEPTH = 30
AGENT_STATE_CHANGE_DELTA = (0.00001, 0.000001)
AGENT_DISCOUNT_FACTOR = 0.9
AGENT_EXPLOIT_EXPLORE_TRADEOFF = 1
AGENT_STEP_TIME = 1*60
AGENT_MAX_INSTRUMENTS = 5
AGENT_RANDOM_SEED = random.randint(0, 1000)
AGENT_CURRENCY = "USD"
AGENT_CORE_PRICING = True
AGENT_COMMISSION_COST = 0.05
AGENT_SPREAD_COST = 0.05
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

REMOTE_TRANSITION_MODEL_ADDRESS = "http://localhost:8080"
REMOTE_MODEL = False

NEW_MODEL = False
MODEL_PATH = os.path.join(BASE_DIR, "res/model3_wrapped.h5")
MODEL_DOWNLOAD_URL = "https://www.dropbox.com/s/9nvcas994dpzq3a/model.h5?dl=0&raw=0"
MODEL_DOWNLOAD = False
