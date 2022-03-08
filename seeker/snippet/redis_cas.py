#date: 2022-03-08T17:11:12Z
#url: https://api.github.com/gists/d88ff9b49ad7b8e077f21493d3969b19
#owner: https://api.github.com/users/jac18281828

    def cas_increase_value(self, value: SingleInt) -> bool:
        """ Global atomic compare and set on value.  It will increase and return True
            or not and return False """
        with self.rcli.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(value.get_key())
                    last_count = pipe.get(value.get_key())
                    last_count = int(last_count) if last_count is not None else 0
                    pipe.multi()
                    if last_count <= value.get_value():
                        pipe.set(value.get_key(), value.get_value())
                        pipe.execute()
                        return True
                    value.set_value(last_count)
                    return False
                except WatchError:
                    continue
                finally:
                    pipe.reset()
