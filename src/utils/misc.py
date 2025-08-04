import torch


def extract_keys(d, parent_key=''):
    keys_list = []
    for k, v in d.items():
        # build new key path
        new_key = f"{parent_key}/{k}" if parent_key else k
        # if value is a dict, recursively call
        if isinstance(v, dict):
            keys_list.extend(extract_keys(v, new_key))
        else:
            keys_list.append(new_key)
    return keys_list


def save_static_dict_keys(static_dict, file_path='static_dict_keys.json'):
    # extract keys from static dict
    checkpoint_keys  = extract_keys(static_dict)

    # save keys to text file
    with open(file_path, 'w') as f:
        for key in checkpoint_keys:
            f.write(f"{key}\n")

    
def load_ema(model, load_path, device='cpu', strict=True):
    ema_ckpt = torch.load(load_path, map_location=device)
    model_state_dict = ema_ckpt['ema']
    # create new state dict with removed 'online_model.' prefix
    new_model_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('online_model.'):
            new_key = key.replace('online_model.', '')
            new_model_state_dict[new_key] = value


    # save_static_dict_keys(new_model_state_dict, file_path='static_dict_keys.txt')
    # save_static_dict_keys(model.state_dict(), file_path='model_state_dict_keys.txt')
        
    load_status = model.load_state_dict(new_model_state_dict, strict=strict)
    # print(device)
    print(f"model loaded from {load_path}")
    
    return model