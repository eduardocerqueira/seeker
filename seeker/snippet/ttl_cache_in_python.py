#date: 2023-07-19T16:54:37Z
#url: https://api.github.com/gists/ef8c1ce1c8ec9da398ef382a0fcfb0dd
#owner: https://api.github.com/users/amanjaiswalofficial

from cachetools import TTLCache, cached
import time
from datetime import datetime, timedelta

# tutorial
cache_with_expiry = TTLCache(maxsize=10,
                             ttl=timedelta(seconds=8),
                             timer=datetime.now)

def time_this(func):
    """Measure time taken by any function in seconds, rounded to 3 decimals"""
    def outer_function(*func_args, **function_kwargs):
        import logging
        logging.basicConfig(format='%(asctime)s: %(levelname)-2s '
                                   '- %(filename)-2s [%(lineno)d]: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO)
        logger = logging.getLogger("root")
        start = time.time()
        output = func(*func_args, **function_kwargs)
        end = time.time()
        logger.info("Total time: {} seconds".format(str((round(end - start, 3)))))
        return output
    return outer_function


def this_runs_every_now_and_then():
    """To fetch/generate new values on cache expiry"""
    value_dict = {}
    for i in range(10):
        value = f"Value for {i} was generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        key = i
        value_dict[key] = value
      
    return value_dict

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********"_ "**********"f "**********"r "**********"o "**********"m "**********"_ "**********"e "**********"x "**********"p "**********"e "**********"n "**********"s "**********"i "**********"v "**********"e "**********"_ "**********"s "**********"e "**********"r "**********"v "**********"i "**********"c "**********"e "**********"( "**********"k "**********"e "**********"y "**********") "**********": "**********"
    """Imitating an expensive service which fetches data from an external service"""
    time.sleep(5)
    return this_runs_every_now_and_then()[key]

@cached(cache=cache_with_expiry)
 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********"_ "**********"f "**********"o "**********"r "**********"_ "**********"k "**********"e "**********"y "**********"( "**********"k "**********"e "**********"y "**********") "**********": "**********"
    """Make use of cache and fetch old values if already requested before"""
    auth = {}
    return get_secrets_from_expensive_service(key, **auth)
    

@time_this
def get_value(key):
    """User interface method to interact with"""
    authenticate = False
    return get_secrets_for_key(key)

this_runs_every_now_and_then()
print(get_value(1))
print(get_value(2))
print(get_value(3))
print(get_value(2))
time.sleep(5) # causing to exceed time than cache expiry, forcing it to fetch new values 
print(get_value(1))
print(get_value(1))
print(get_value(2))