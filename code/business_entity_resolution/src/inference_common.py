"""Checksummed local inference artifacts; no training labels are read here."""
import hashlib
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path
from config import ROOT, save_json


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def code_hash(names):
    return {n:sha256(Path(__file__).with_name(n)) for n in names}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sqlite(path,readonly=False):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True) if readonly else sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-32768')
    return db


def seal(path,metadata):
    path=Path(path)
    save_json(path.with_suffix(path.suffix+'.json'),dict(metadata,sha256=sha256(path),bytes=path.stat().st_size))


def valid(path,expected):
    path=Path(path); manifest=path.with_suffix(path.suffix+'.json')
    if not path.exists() or not manifest.exists(): return False
    m=read_json(manifest)
    return all(m.get(k)==v for k,v in expected.items()) and m['sha256']==sha256(path)


def freeze(source,destination):
    source,destination=Path(source),Path(destination)
    names=['model.txt','model_config.json','metrics.json','splits.json']
    identity={'files':{n:sha256(source/n) for n in names},
              'code':code_hash(['features.py','normalize.py']),
              'requirements_sha256':sha256(ROOT/'code/business_entity_resolution/requirements.txt')}
    manifest=destination/'manifest.json'
    if manifest.exists():
        old=read_json(manifest)
        if old['identity']!=identity: raise ValueError('Frozen baseline differs; choose a new destination')
        for n,h in identity['files'].items():
            if sha256(destination/n)!=h: raise ValueError('Frozen baseline checksum mismatch')
        return old
    destination.mkdir(parents=True,exist_ok=True)
    for n in names: shutil.copy2(source/n,destination/n)
    m={'identity':identity,'created_unix':time.time(),'baseline':'baseline-v1'}
    save_json(manifest,m)
    return m


def sampled_rss():
    if os.name!='nt': return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in
            ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    c=Counters();c.cb=ctypes.sizeof(c)
    ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(c),c.cb):
        return int(c.WorkingSetSize)
    return None
