from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/{time:YYYY-MM-DD}.log", rotation="1 day", retention="7 days")