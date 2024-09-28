from copy import deepcopy

from .utils import load_data
from .model import Model
from common import print_log

def initialize_model(config, data_source):
    model_config = config['model']
    model_param = deepcopy(model_config)
    model_param['data_source'] = data_source
    model_param['save_path'] = './work/checkpoints'
    m = Model(**model_param)
    print_log("Model initialization Finshed!")
    return m, model_param['save_path']


def initialization(config, training=False):
    data_source = load_data(**config['data'], cache=training)
    if training:
        data_source.load_all_data()
    return initialize_model(config, data_source)