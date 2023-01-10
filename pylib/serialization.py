import json
import os
import pickle

import pylib as py


def _check_ext(path, default_ext):
    name, ext = os.path.splitext(path)
    if ext == '':
        if default_ext[0] == '.':
            default_ext = default_ext[1:]
        path = name + '.' + default_ext
    return path


def save_json(path, obj, **kwargs):
    # default
    if 'indent' not in kwargs:
        kwargs['indent'] = 4
    if 'separators' not in kwargs:
        kwargs['separators'] = (',', ': ')

    path = _check_ext(path, 'json')

    # wrap json.dump
    with open(path, 'w') as f:
        json.dump(obj, f, **kwargs)


def load_json(path, **kwargs):
    # wrap json.load
    with open(path) as f:
        return json.load(f, **kwargs)


def save_yaml(path, data, **kwargs):
    import oyaml as yaml

    path = _check_ext(path, 'yml')

    with open(path, 'w') as f:
        yaml.dump(data, f, **kwargs)


def load_yaml(path, **kwargs):
    import oyaml as yaml
    with open(path) as f:
        yaml.add_multi_constructor('tag:yaml.org,2002:python/tuple', lambda loader, suffix, node: None,
                                   Loader=yaml.SafeLoader)
        return yaml.safe_load(f, **kwargs)


def save_pickle(path, obj, **kwargs):
    path = _check_ext(path, 'pkl')

    # wrap pickle.dump
    with open(path, 'wb') as f:
        pickle.dump(obj, f, **kwargs)


def load_pickle(path, **kwargs):
    # wrap pickle.load
    with open(path, 'rb') as f:
        return pickle.load(f, **kwargs)


def set_ganterfactual_repo_args(args):
    args.ganterfactual_repo = True
    args.crop_size = 512
    if args.dataset == "rsna":
        args.img_channels = 1  # Old Models with UNET and Alexnet -> 1 channel
        args.clf_name = "alexnet"
        args.clf_ckp_name = "2022-10-13--13.03"
    else:  # For MURA
        args.img_channels = 3  # Old Models with UNET and Alexnet -> 1 channel
        args.clf_name = "inception"
        args.clf_ckp_name = "2022-06-04--00.05"
        args.body_parts = "XR_WRIST"
    args.attention_type = "none"
    args.batch_size = 1
    args.img_shape = (args.crop_size, args.crop_size, args.img_channels)
    return args


def load_args(name, test_args, experiments_dir, training=False):
    # Load the args from the experiment directory
    try:
        if training:
            args = py.args_from_yaml(py.join(experiments_dir, name, 'settings.yml'))
        else:
            # Add args from testing script.
            args = py.args_from_yaml(py.join(experiments_dir, name, 'settings.yml'))
            args.__dict__.update(test_args.__dict__)  # Save test_args to loaded args
        try:
            args.img_shape = (args.crop_size, args.crop_size, args.img_channels)
        except AttributeError:
            args.img_shape = (args.crop_size, args.crop_size, 3)  # ABC-GANs images have always 3 channels.
    except FileNotFoundError:  # From GANterfacfual
        print("Couldn't find abc-gan settings.Loading for Ganterfactual.")
        args = set_ganterfactual_repo_args(test_args)
    return args
