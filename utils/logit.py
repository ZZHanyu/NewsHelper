import logging
from functools import wraps

class log_decorator(object):
    def __init__(self) -> None:
        logging.basicConfig(filename=args.logging_path + f'{args.date_time}', level=logging.INFO)
        pass
    def __call__(self, a_func) -> logging.Any:
        @warps(a_func)
        def warpped_function():
            logging
        pass

