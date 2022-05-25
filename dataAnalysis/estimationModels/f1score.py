

def precision(m, v):
    tp = m[v][v]
    fp = 0
    for i in range(len(m)):
        if i != v:
            fp += m[v][i]
    return tp / (tp+fp)

def recall(m, v):
    tp = m[v][v]
    fn = 0
    for i in range(len(m)):
        if i != v:
            fn += m[i][v]
    return tp / (tp + fn)

def f1score(m, v):
    p = precision(m, v)
    r = recall(m, v)
    return (2*p*r)/(p+r)


