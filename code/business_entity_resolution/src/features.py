"""Pair comparisons; IDs and country identities are never model features."""
from rapidfuzz.fuzz import ratio, token_sort_ratio, token_set_ratio
from normalize import expanded, numeric_tokens, accent_fold


def overlap(a, b):
    a,b=set(a),set(b)
    return len(a & b)/len(a | b) if a or b else 0.0


def pair_features(a,b):
    values=[]
    for i,kind in [(1,'name'),(2,'address')]:
        x,y=a[i] or '',b[i] or ''
        ex,ey=expanded(x,kind),expanded(y,kind)
        nx,ny=set(numeric_tokens(x)),set(numeric_tokens(y))
        values += [ratio(x,y)/100,token_sort_ratio(x,y)/100,
                   token_set_ratio(x,y)/100,ratio(ex,ey)/100,
                   ratio(ex.replace(' ',''),ey.replace(' ',''))/100,
                   ratio(accent_fold(x),accent_fold(y))/100,
                   overlap(x.split(),y.split()),overlap(nx,ny),
                   float(bool(nx and ny) and nx.isdisjoint(ny)),
                   float(bool(x) and x==y),float(not x),float(not y),
                   min(len(x),len(y))/max(len(x),len(y),1)]
    values += [float(b[0].startswith('S3-'))]
    return values
