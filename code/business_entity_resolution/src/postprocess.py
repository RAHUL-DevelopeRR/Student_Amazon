"""Frozen independent threshold decisions; retain empty and multiple matches."""
import math


def select_matches(ids,scores,threshold):
    if len(ids)!=len(scores) or len(ids)!=len(set(ids)): raise ValueError('Invalid scored candidate set')
    if not 0<=threshold<=1: raise ValueError('Invalid threshold')
    if any(not math.isfinite(float(s)) or not 0<=s<=1 for s in scores): raise ValueError('Invalid model score')
    return [i for i,s in zip(ids,scores) if s>=threshold]
