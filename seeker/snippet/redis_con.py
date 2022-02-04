#date: 2022-02-04T16:44:58Z
#url: https://api.github.com/gists/44efdbcb9dbc7595fd6a5f58fe1bbf0f
#owner: https://api.github.com/users/roguh

from redis.sentinel import Sentinel
def connect_to_redis(address: str, port: int, redis_set: str):
    sentinel = Sentinel([(address, port)], socket_timeout=0.5)
    con = sentinel.master_for(
        redis_set,
        decode_responses=True,
        socket_timeout=0.5,
    )
    con.echo("test")
    return con