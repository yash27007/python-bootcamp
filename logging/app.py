import logging

#logging setting
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("app1.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ArithmeticApp")

def add(a,b):
    logger.debug(f"Adding {a} and {b} = {a+b}")
    return a+b

def sub(a,b):
    logger.debug(f"Subracting {a} and {b} = {a-b}")
    return a-b

def mul(a,b):
    logger.debug(f"Multiplying {a} and {b} = {a*b}")
    return a*b

def divide(a,b):
    try:
        result = a/b
        logger.debug(f"Diving {a} by {b} = {result}")
        return result
    except ZeroDivisionError:
        logger.error("Division by Zero Error")
        return None

add(3,5)
sub(34,21)
mul(19,13)
divide(334,123)
divide(23423, 0)

