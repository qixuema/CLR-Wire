import yaml

class NestedDictToClass:
    def __init__(self, dictionary):
        self._convert(dictionary)
    
    def _convert(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = NestedDictToClass(value)
            elif isinstance(value, list):
                value = tuple(value)

            setattr(self, key.lower(), value)

# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def print_custom_attributes(obj, indent=''):
    attributes = vars(obj)
    for attr, value in attributes.items():
        if not attr.startswith("__"):
            if isinstance(value, (dict, list, tuple)):
                print(f"{indent}{attr}:")
                print_custom_attributes(value, indent + "  ")
            else:
                print(f"{indent}{attr}: {value}")

if __name__ == '__main__':
    cfg = load_config('train.yaml')
    args = NestedDictToClass(cfg)

    print_custom_attributes(args)