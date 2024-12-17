from itertools import chain, combinations
import math

def get_number_of_equivalent_tokens(similarity_values_pred_orig, arg_pred, arg_orig, similarity_threshold):
    count = 0

    set_lower_tokens_arg_pred = set([tk.lower() for tk in arg_pred])
    set_lower_tokens_arg_orig = set([tk.lower() for tk in arg_orig])

    for i in range(len(set_lower_tokens_arg_pred)):
        for j in range(len(set_lower_tokens_arg_orig)):
            if (similarity_values_pred_orig[i,j].numpy() >= similarity_threshold):
                count += 1
                break
                # El break es para no contar repetidos (cuando una palabra de arg_pred tiene similitud >= threshold con más de una palabra de arg_orig)
                # VER! Queremos contarlos múltiples veces en ese caso? En principio creería que no, pero no estoy 100% segura.

    return count

def get_number_of_tokens(token_list):
    return len(set(token.lower() for token in token_list))


def sem_sim(tp, set_tokens_gold, tout, comparison_dict):
    # it can be replaced with model's similarity predictions.
    # here, we already computed the similarity values among tokens using SBERT
    # and saved that information with the following json format:
    # tk1 : {tk1: sim1, tk2: sim2, tk3: sim3, ..., tkn: simn},
    # tk2 : {tk1: sim2, tk2: simn...}

    token_entry = comparison_dict[tp]
    if token_entry:
        return len([tg for tg in set_tokens_gold if token_entry[tg] >= tout]) > 0
    else:
        return False


def power_set(iterable):
    # Generate all subsets (power set)
    items = list(iterable)
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))

def subset_meets_condition(subset_set, comparison_dict, threshold):
    # Check if all pairs in the subset meet the condition
    for tk1 in subset_set:
        for tk2 in subset_set:
            if tk1 != tk2 and comparison_dict.get(tk1, {}).get(tk2, 0) >= threshold:
                return False
    return True


def compute_secure_tokens(token_set, threshold, comparison_dict):
    # return tokens in token_set such that there is no other token with similarity >= threshold.
    secure_tokens = set()

    for tk1 in token_set:
        is_secure = True
        for tk2 in token_set:
            if tk1 != tk2 and comparison_dict.get(tk1, {}).get(tk2, 0) >= threshold:
                is_secure = False
                break
        if is_secure:
            secure_tokens.add(tk1)
    
    return secure_tokens


def reduce_tokens_to_maximal_set(token_set, threshold, comparison_dict):

    secure_tokens = compute_secure_tokens(token_set, threshold, comparison_dict)

    no_secure_tokens = token_set - secure_tokens
    
    # Start from the largest subsets and go down
    for N in range(len(no_secure_tokens), 0, -1):
        sets_of_len_N = list(combinations(no_secure_tokens, N))
        for subset in sets_of_len_N:
            subset_set = set(subset)
            if subset_meets_condition(subset_set, comparison_dict, threshold):
                return subset_set.union(secure_tokens)
    return secure_tokens



def get_similarity_tokens(tokens_pred, tokens_gold, tin, tout, comparison_dict):
    # it follows the definition in Section 6 of the paper.
    set_tokens_pred = set(token.lower() for token in tokens_pred)
    set_tokens_gold = set(token.lower() for token in tokens_gold)
    S = set([tp for tp in set_tokens_pred if sem_sim(tp, set_tokens_gold, tout, comparison_dict)]) # tokens of predicted argument with similarity > tout with tokens in gold argument.

    Sprime = reduce_tokens_to_maximal_set(S, tin, comparison_dict)

    
    return Sprime

# Adapted Jaccard Similarity Measure
def adapted_jaccard(number_tokens_pred, number_tokens_gold, number_similarity_tokens):
    arg_similarity = number_similarity_tokens / (number_tokens_gold + number_tokens_pred - number_similarity_tokens)

    return arg_similarity

# Adapted Dice Similarity Measure
def adapted_dice(number_tokens_pred, number_tokens_gold, number_similarity_tokens):

    arg_similarity = (2*number_similarity_tokens) / (number_tokens_gold + number_tokens_pred)

    return arg_similarity

# Adapted Sorensen Similarity Measure
def adapted_sorensen(number_tokens_pred, number_tokens_gold, number_similarity_tokens):

    arg_similarity = (4*number_similarity_tokens) / (number_tokens_gold + number_tokens_pred + 2*number_similarity_tokens)

    return arg_similarity

# Adapted Symmetric Anderberg Similarity Measure
def adapted_symmetric_anderberg(number_tokens_pred, number_tokens_gold, number_similarity_tokens):

    arg_similarity = (8*number_similarity_tokens) / (number_tokens_gold + number_tokens_pred + 6*number_similarity_tokens)

    return arg_similarity

# Adapted 'Sokal and Sneath 2' Similarity Measure
def adapted_sokal_sneath2(number_tokens_pred, number_tokens_gold, number_similarity_tokens):

    arg_similarity = (number_similarity_tokens) / (2*(number_tokens_gold + number_tokens_pred) - 3*number_similarity_tokens)

    return arg_similarity

# Adapted Ochiai Similarity Measure
def adapted_ochiai(number_tokens_pred, number_tokens_gold, number_similarity_tokens):

    arg_similarity = (number_similarity_tokens) / (math.sqrt(number_tokens_gold) * math.sqrt(number_tokens_pred))

    return arg_similarity

# Adapted 'Kulczynski 2' Similarity Measure
def adapted_kulczynski2(number_tokens_pred, number_tokens_gold, number_similarity_tokens):

    arg_similarity = ((number_similarity_tokens / number_tokens_gold) + (number_similarity_tokens / number_tokens_pred))/2

    return arg_similarity