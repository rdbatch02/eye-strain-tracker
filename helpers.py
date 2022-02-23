from distutils.log import debug


class Logger:
    def __init__(self, debug = False):
        self.debug = debug
        self.last_msg = ""
    def log(self, msg, force_write=False): # Shortcut to only print log messages if debug is enabled
        if self.debug and (self.last_msg != msg or force_write):
            self.last_msg = msg
            print(msg)        

def minutes(mins):
    return mins * 60