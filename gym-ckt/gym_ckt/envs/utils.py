import framework
import numpy as np
from scipy import interpolate
import random as rnd
import statistics
import random

from collections import OrderedDict
from gym_ckt.spaces.dtup import DiscreteTuple
import yaml
import yaml.constructor

import os

## helper functions for working with files
def rel_path(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def load_array(fname):
    with open(rel_path(fname), "rb") as f:
        arr = np.load(f)
    return arr

## reward functions of interest
def norm_huber(x, d=0.05, g=1):
    if (abs(x) < d):
        return g * (0.5 * x**2)
    else:
        return g * (0.5 * x**2 + d * (abs(x) - d))

def norm_huber_lt(x, d=0.05, g=1):
    if (x < 0):
        return 0.0
    else:
        return norm_huber(x,d, g)

def norm_huber_gt(x, d=0.05, g=1):
    if (x > 0):
        return 0.0
    else:
        return norm_huber(x,d, g)

## working with min/max state variable regions
def box_move(x, xmin, xmax, dx):
    new_x = x + dx
    if (new_x < xmin):
        return xmin
    elif (new_x > xmax):
        return xmax
    else:
        return new_x

def rel_diff(curr, desired):
    diff = (curr-desired)/statistics.mean([curr,desired])
    return diff

class OrderedDictYAMLLoader(yaml.Loader):
    """
    A YAML loader that loads mappings into ordered dictionaries.
    """

    def __init__(self, *args, **kwargs):
        yaml.Loader.__init__(self, *args, **kwargs)

        self.add_constructor(u'tag:yaml.org,2002:map', type(self).construct_yaml_map)
        self.add_constructor(u'tag:yaml.org,2002:omap', type(self).construct_yaml_map)

    def construct_yaml_map(self, node):
        data = OrderedDict()
        yield data
        value = self.construct_mapping(node)
        data.update(value)

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.MappingNode):
            self.flatten_mapping(node)
        else:
            raise yaml.constructor.ConstructorError(None, None,
                'expected a mapping node, but found %s' % node.id, node.start_mark)

        mapping = OrderedDict()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping

