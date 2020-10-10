import os
import json
import random as python_random

def save_for_evaluation(path_save, results, dataset_name):
    path_save_file = os.path.join(path_save, f'results_{dataset_name}.jsonl')
    with open(path_save_file, 'w', encoding='utf-8') as file:
        for index, line in enumerate(results):
            json_string = json.dumps(line, ensure_ascii=False, default=json_default_type_checker)
            json_string += '\n'

            file.writelines(json_string)

def json_default_type_checker(o):
    """
    From https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
    """
    if isinstance(o, int): return int(o)
    raise TypeError

def load_jsonl(path_file, seed=1):
    data = []
    with open(path_file, "r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            curr_line = json.loads(line.strip())
            data.append(curr_line)
    return data

