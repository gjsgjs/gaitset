from time import strftime, localtime

def print_log(message):
    print('[{0}] [INFO]: {1}'.format(strftime('%Y-%m-%d %H:%M:%S', localtime()), message))

def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))