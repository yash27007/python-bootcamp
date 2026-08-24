import sys
import logging

logging.basicConfig(
    level=logging.WARNING,   # central switch: only WARNING and above pass through
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("demo")

print("--- with level=WARNING (production-like setting) ---")
logger.debug("connecting to db at host=10.0.0.5")   # suppressed, no code changed at the call site
logger.info("request served in 42ms")                # suppressed
logger.warning("cache miss rate above 30 percent")
logger.error("payment gateway timeout")

logging.getLogger("demo").setLevel(logging.DEBUG)
print("\n--- after raising the level to DEBUG (same call sites, zero edits) ---")
logger.debug("connecting to db at host=10.0.0.5")
logger.info("request served in 42ms")
logger.warning("cache miss rate above 30 percent")
logger.error("payment gateway timeout")
