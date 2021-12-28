from pytz import timezone

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

DEFAULT_PG_CONFIG = {
	"host": "localhost",
	"database": "rtrader_db",
	"user": "rtrader_admin",
	"password": "4U7z7KJM"  # TODO
}

REMOTE_TRADER_URL = "http://localhost:8080/"

"""
HISTORICAL_TABLE_NAME = "currencies_history"
CURRENCIES_TABLE_NAME = "currencies"
TRADEABLE_PAIRS_TABLE_NAME = "tradeable_pairs"
DISTINCT_DATETIMES_TABLE_NAME = "distinct_datetimes"
"""

HISTORICAL_TABLE_NAME = "test_currencies_history"
CURRENCIES_TABLE_NAME = "test_currencies"
TRADEABLE_PAIRS_TABLE_NAME = "test_tradeable_pairs"
DISTINCT_DATETIMES_TABLE_NAME = "test_distinct_datetimes"


MARKET_STATE_MEMORY = 64
AGENT_DEPTH = 5

NETWORK_TRIES = 5
POLYGON_API_KEY = "1ijeQ0XUYNl1YMHy6Wl_5zEBtGbkipUP"

OANDA_TOKEN = "4e3bc058fee3b2005e2a651081da881e-1cc2b5245cda5e61beb340aaf217c704"
OANDA_TRADING_URL = "https://api-fxpractice.oanda.com/v3"
OANDA_TRADING_ACCOUNT_ID = "101-001-19229086-007"
OANDA_TEST_ACCOUNT_ID = "101-001-19229086-006"
DEFAULT_TIME_IN_FORCE = "FOK"
TIMEZONE = timezone("Africa/Addis_Ababa")

LOGGING = True


OPTIMIZER_PG_CONFIG = {
	"dsn": "postgres://ontiwpwwgbtgwp:8702c0dec88af3c49473d464bf44e8ad17419facfce764c8684ed540839fb8cb@ec2-34-194-100-156.compute-1.amazonaws.com:5432/dcs4e3sfc908fi",
	"sslmode": "require"
}