import logging


def config_log(stream_level="DEBUG", file_level="INFO", log_file=None):
    """Config logging with dictConfig.
    Parameters
    ----------
    log_file: log file
    stream_level: logging level for STDOUT
    file_level: logging level for log file
    """
    import tempfile
    from logging.config import dictConfig

    if log_file is None:
        _, log_file = tempfile.mkstemp()
        
    return log_file


##TODO: Clean this func
class Logger():
    def __init__(self, config):
        self.logfile = config_log(stream_level=config.log_level, log_file=config.log_file)
        self.log_level = config.log_level
        self.log_file = config.log_file
        print(config.log_file)
        print(self.logfile)
    
    def info(self, info):
        print(info)