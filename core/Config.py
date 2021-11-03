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
	"database": "rtraderdb",
	"user": "rtrader_admin",
	"password": "4U7z7KJM"  # TODO
}

REMOTE_TRADER_URL = "http://localhost:8080/"

TABLE_NAME = "currencies_history"

MARKET_STATE_MEMORY = 64

NETWORK_TRIES = 5
POLYGON_API_KEY = "1ijeQ0XUYNl1YMHy6Wl_5zEBtGbkipUP"

OANDA_TOKEN = "4e3bc058fee3b2005e2a651081da881e-1cc2b5245cda5e61beb340aaf217c704"
OANDA_TRADING_URL = "https://api-fxpractice.oanda.com/v3"
OANDA_TRADING_ACCOUNT_ID = "101-001-19229086-007"
OANDA_TEST_ACCOUNT_ID = "101-001-19229086-006"
DEFAULT_TIME_IN_FORCE = "FOK"
TIMEZONE = timezone("Africa/Addis_Ababa")

LOGGING = True
