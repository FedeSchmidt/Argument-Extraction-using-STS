from utils import *

def get_categorization(instances):
    categorization = []

    for (tokens, labels, predictions) in instances:
                
        ground_truth_indices = get_arguments_indices(labels)
        ground_truth_arguments = [" ".join(tokens[gr_start : gr_end]) for (gr_start, gr_end) in ground_truth_indices]
    
        predicted_args_indices = get_arguments_indices(predictions)
        
        R_matrix = compute_r_matrix(ground_truth_indices, predicted_args_indices)

        for i, (pr_start, pr_end) in enumerate(predicted_args_indices):
            column = R_matrix[:, i]
            positive_values = column[column > 0].tolist()
            predicted_argument = " ".join(tokens[pr_start:pr_end])
            
            if len(positive_values) == 0:
                categorization.append((f"p_{i}", predicted_argument, 'made-up', []))
            elif len(positive_values) == 1:
                category = 'Match' if positive_values[0] == 1 else 'partial-Match'
                categorization.append((f"p_{i}", predicted_argument, category, positive_values))
            elif len(positive_values) > 1:
                categorization.append((f"p_{i}", predicted_argument, 'partial-Match', positive_values))

    return categorization

# to check
def get_categorization_with_gold_arguments(instances):
    categorization = []

    for (tokens, labels, predictions) in instances:
                
        ground_truth_indices = get_arguments_indices(labels)
        ground_truth_arguments = [" ".join(tokens[gr_start : gr_end]) for (gr_start, gr_end) in ground_truth_indices]
    
        predicted_args_indices = get_arguments_indices(predictions)
        
        R_matrix = compute_r_matrix(ground_truth_indices, predicted_args_indices)

        for i, (pr_start, pr_end) in enumerate(predicted_args_indices):
            column = R_matrix[:, i]
            positive_values = column[column > 0].tolist()
            predicted_argument = " ".join(tokens[pr_start:pr_end])
            
            if len(positive_values) == 0:
                categorization.append((f"p_{i}", predicted_argument, 'made-up', 0, '-'))
            elif len(positive_values) == 1:
                gold_index = [i for i, x in enumerate(column) if x > 0][0]
                category = 'Match' if positive_values[0] == 1 else 'partial-Match'
                categorization.append((f"p_{i}", predicted_argument, category, positive_values[0], ground_truth_arguments[gold_index]))
            elif len(positive_values) > 1:
                gold_indices = [i for i, x in enumerate(column) if x > 0]
                matched_golds = [ground_truth_arguments[j] for j in gold_indices]
                # for j, gold_index in enumerate(gold_indices):
                categorization.append((f"p_{i}", predicted_argument, 'partial-Match', positive_values, matched_golds))

    return categorization

def get_number_predicted_arguments_per_category(categorization):
    number_match = 0
    number_partial_match = 0
    number_made_ups = 0

    for (y, z) in categorization:
        if len(z) == 0:
            number_made_ups += 1
        elif len(z) == 1:
            if z[0] == 1:
                number_match += 1
            else:
                number_partial_match += 1
        elif len(z) > 1:
            number_partial_match += 1

    return number_match, number_partial_match, number_made_ups

def get_values_for_predicted_arguments(instances):
    categorization = []

    for j, (tokens, labels, predictions) in enumerate(instances):
                
        ground_truth_indices = get_arguments_indices(labels)
        ground_truth_arguments = [" ".join(tokens[gr_start : gr_end]) for (gr_start, gr_end) in ground_truth_indices]
    
        predicted_args_indices = get_arguments_indices(predictions)
        
        R_matrix = compute_r_matrix(ground_truth_indices, predicted_args_indices)
    
        for i, (pr_start, pr_end) in enumerate(predicted_args_indices):
            column = R_matrix[:, i]
            positive_values = column[column > 0].tolist()
            predicted_argument = " ".join(tokens[pr_start:pr_end])
            categorization.append((f"p{j}.{i}", positive_values))

    return categorization

def get_semantic_values_for_predicted_arguments(instances, semModel):
    categorization = []

    for instance_nr, (tokens, labels, predictions) in enumerate(instances):
                
        ground_truth_indices = get_arguments_indices(labels)
        ground_truth_arguments = [" ".join(tokens[gr_start : gr_end]) for (gr_start, gr_end) in ground_truth_indices]
    
        predicted_args_indices = get_arguments_indices(predictions)
        
        R_matrix = compute_r_matrix(ground_truth_indices, predicted_args_indices)
    
        for i, (pr_start, pr_end) in enumerate(predicted_args_indices):
            column = R_matrix[:, i]
            positive_values = column[column > 0].tolist()
            predicted_argument = " ".join(tokens[pr_start:pr_end])

            if len(positive_values) == 0:
                categorization.append((f"p{instance_nr}.{i}", []))
            elif len(positive_values) == 1:
                if positive_values[0] == 1:
                    categorization.append((f"p{instance_nr}.{i}", [1]))
                else: # partial-match
                    semantic_simil_values = []
                    gold_indices = [gi for gi, x in enumerate(column) if x > 0]
                    matched_golds = [ground_truth_arguments[j] for j in gold_indices]
                    for matched_gold_argument in matched_golds:
                        embeddings = semModel.encode([predicted_argument, matched_gold_argument])
                        similarities = semModel.similarity(embeddings, embeddings)
                        semantic_simil_values.append(round(abs(similarities[0][1].item()), 3))
                    categorization.append((f"p{instance_nr}.{i}", semantic_simil_values))
            elif len(positive_values) > 1: # partial-match
                semantic_simil_values = []
                gold_indices = [gi for gi, x in enumerate(column) if x > 0]
                matched_golds = [ground_truth_arguments[j] for j in gold_indices]
                for matched_gold_argument in matched_golds:
                    embeddings = semModel.encode([predicted_argument, matched_gold_argument])
                    similarities = semModel.similarity(embeddings, embeddings)
                    semantic_simil_values.append(round(abs(similarities[0][1].item()), 3))
                categorization.append((f"p{instance_nr}.{i}", semantic_simil_values))

    return categorization

def get_partial_match_values(categorization):
    partial_match_values = []
    for (y, z) in categorization:
        if (len(z) == 1) and (0 < z[0] < 1):
            partial_match_values.append(z[0])
        elif len(z) > 1:
            partial_match_values.append(sum(t for t in z) / len(z))

    return partial_match_values

def get_gold_arguments(instances):
    nro_gold_args = 0
    for (tokens, labels, predictions) in instances:
        ground_truth_indices = get_arguments_indices(labels)
        nro_gold_args += len(ground_truth_indices)

    return nro_gold_args

def get_pred_arguments(instances):
    nro_pred_args = 0
    for (tokens, labels, predictions) in instances:
        predicted_args_indices = get_arguments_indices(predictions)
        nro_pred_args += len(predicted_args_indices)

    return nro_pred_args

def get_unrecognized_arguments(instances):
    nro_unrecognized = 0

    for (tokens, labels, predictions) in instances:
        ground_truth_indices = get_arguments_indices(labels)
        predicted_args_indices = get_arguments_indices(predictions)
        R_matrix = compute_r_matrix(ground_truth_indices, predicted_args_indices)
        for i, (gd_start, gd_end) in enumerate(ground_truth_indices):
            row = R_matrix[i, :]
            positive_values = row[row > 0].tolist()
            if len(positive_values) == 0:
                nro_unrecognized += 1
    
    return nro_unrecognized