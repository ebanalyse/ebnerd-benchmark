class set_args:
    def __init__(self, args_dict):
        _ = [setattr(set_args, key, val) for key, val in args_dict.items()]


def print_n_parameters(model) -> None:
    num_params = model.count_params()
    print("Number of parameters:", num_params)


def print_parameter_device(model) -> None:
    for variable in model.variables:
        print(f"Variable name: {variable.name}, Device: {variable.device}")
