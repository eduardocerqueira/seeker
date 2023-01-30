#date: 2023-01-30T16:50:20Z
#url: https://api.github.com/gists/05ad3d6a38cc119fdbd2d5d2d5f60aa3
#owner: https://api.github.com/users/aidanjbailey

# requires the class to have a defined `_logger` Logger object

def log_outcome(opdesc: Optional[str] = None):
    def _log_outcome(func):
        def __log_outcome(self, *args, **kwargs):
            desc = opdesc if opdesc is not None else func.func_name
            self._logger.debug(
                f"trying to {desc} for args: {args} and kwargs: {kwargs}..."
            )
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                self._logger.error(
                    f"failed to {desc} for args: {args} and kwargs: {kwargs}: {e}"
                )
                raise
            self._logger.info(f"succeeded to {desc}")
            return result

        return __log_outcome

    return _log_outcome
