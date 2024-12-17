import numpy as np

"""
Get tokens, labels and predictions from predictions file.
"""
def parse_file(file_path):
    tokens, labels, predictions = [], [], []
    sep_lines = 0
    try:
        with open(file_path, encoding='utf-8') as nf:
            lines = nf.readlines()
            for num_line, line in enumerate(lines):
                if line.strip():
                    tk, lb, pr = line.split()
                    tokens.append(tk)
                    labels.append(lb)
                    predictions.append(pr)
                else:
                    sep_lines += 1
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError:
        print(f"Invalid file format: {file_path}")
        print(num_line, line, sep_lines)

    return tokens, labels, predictions

"""
Get tokens, labels and predictions from predictions file.
"""
def parse_file_for_arg_level(file_path):
    instances = []
    try:
        with open(file_path, encoding='utf-8') as nf:
            lines = nf.readlines()
            tokens, labels, predictions = [], [], []
            for line in lines:
                if line.strip():
                    tk, lb, pr = line.split()
                    tokens.append(tk)
                    labels.append(lb)
                    predictions.append(pr)
                else:
                    instances.append((tokens, labels, predictions))
                    tokens, labels, predictions = [], [], []
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError:
        print(f"Invalid file format: {file_path}")

    return instances


"""
Get argument indices from sequence of labels/predictions.
"""
def get_arguments_indices(sequence):
    indices = []
    start_index = None
    for i, label in enumerate(sequence):
        if label == '1':
            
            if start_index is None:
                start_index = i
            else:
                indices.append((start_index, i))
                start_index = i
        
        elif (label == '0') and (start_index is not None):
            indices.append((start_index, i))
            start_index = None

    if start_index is not None:
        indices.append((start_index, len(sequence)))

    return indices

"""
Compute matrix with R_values.
"""
def compute_r_matrix(gold_indices, predicted_indices):

    def get_R(gold_argument, predicted_argument):
        (gold_start, gold_end) = gold_argument
        (pred_start, pred_end) = predicted_argument

        intersection_start = max(gold_start, pred_start)
        intersection_end = min(gold_end, pred_end)

        len_intersection_interval = (intersection_end - intersection_start) if intersection_start <= intersection_end else 0
        len_longer_span = max(gold_end - gold_start, pred_end - pred_start)
        return round((len_intersection_interval / len_longer_span), 3)
    
    R_matrix = np.zeros((len(gold_indices), len(predicted_indices)), dtype=float)
    
    for i, gold_argument in enumerate(gold_indices):
        for j, predicted_argument in enumerate(predicted_indices):
            R_matrix[i][j] = get_R(gold_argument, predicted_argument)
    
    return R_matrix