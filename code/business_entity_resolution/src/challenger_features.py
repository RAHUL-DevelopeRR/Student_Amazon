"""Additional text-only evidence; no external dictionaries or identity lookup."""
import re
from functools import lru_cache
from rapidfuzz.fuzz import ratio, token_sort_ratio
from features import pair_features


@lru_cache(maxsize=16384)
def parts(text):
    tokens=tuple(text.split())
    nums=tuple(str(int(x)) for x in re.findall(r'\d+',text))
    return tokens,nums,''.join(tokens)


def enhanced(a,b):
    out=pair_features(a,b)
    for j in (1,2):
        x,nx,cx=parts(a[j] or '');y,ny,cy=parts(b[j] or '')
        sx,sy=set(x),set(y);shared=sx&sy
        out += [len(shared)/max(len(sx),1),len(shared)/max(len(sy),1),
                len(sx-sy),len(sy-sx),ratio(cx,cy)/100,
                float(bool(nx and ny) and nx[0]==ny[0]),
                float(bool(nx and ny) and nx[0]!=ny[0]),
                float(bool(nx and ny) and nx==ny),
                len(set(nx)-set(ny)),len(set(ny)-set(nx)),
                ratio(' '.join(nx),' '.join(ny))/100,
                token_sort_ratio(' '.join(nx),' '.join(ny))/100]
    return out


if __name__=='__main__':
    a=('S1-a','example','241 billington street apt 18','US')
    b=('S2-b','example','254 billington street apt 18','US')
    assert len(enhanced(a,b))==51
    assert enhanced(a,b)[45]==1
    assert parts('00403 street')[1]==('403',)
