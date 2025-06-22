import copy

def fed_avg(client_weights):
    total = sum(n for _, n in client_weights)
    base = copy.deepcopy(client_weights[0][0])

    for key in base.keys():
        base[key] *= client_weights[0][1] / total

    for state, count in client_weights[1:]:
        for key in base:
            base[key] += state[key] * (count / total)

    return base
