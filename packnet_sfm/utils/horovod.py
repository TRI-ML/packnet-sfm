
import horovod.torch as hvd

########################################################################################################################

def hvd_init():
    hvd.init()

def on_rank_0(func):
    def wrapper(*args, **kwargs):
        if rank() == 0:
            func(*args, **kwargs)
    return wrapper

def rank():
    return hvd.rank()

def world_size():
    return hvd.size()

@on_rank_0
def print0(string='\n'):
    print(string)

########################################################################################################################
