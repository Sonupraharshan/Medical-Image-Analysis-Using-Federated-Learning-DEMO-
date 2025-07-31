import copy

def fed_avg(client_weights):
    base = client_weights[0][0].copy()
    total = sum([cw[1] for cw in client_weights])
    for key in base.keys():
        base[key] = base[key].float() * (client_weights[0][1] / total)
        for cw in client_weights[1:]:
            base[key] += cw[0][key].float() * (cw[1] / total)
    return base
