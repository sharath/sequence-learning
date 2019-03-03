import sys
from time import time

__start = str(int(time()))

class __LogPrint(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)    
    def flush(self):
        for f in self.files:
            f.flush()

def start_logging():
    f = open(__start + '.log', 'w')
    sys.stdout = __LogPrint(sys.stdout, f)