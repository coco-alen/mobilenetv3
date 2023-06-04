import importlib

def import_variable(file_path, variable_name):
    spec = importlib.util.spec_from_file_location('quantize_config', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    variable = getattr(module, variable_name)
    return variable