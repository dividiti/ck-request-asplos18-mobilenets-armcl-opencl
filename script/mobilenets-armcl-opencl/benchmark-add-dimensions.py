#! /usr/bin/python
import ck.kernel as ck
import os

def do(i):

    # List performance entries
    r=ck.access({'action':'search',
                 'module_uoa':'experiment',
                 'data_uoa':'mobilenets-performance-*',
                 'repo_uoa':'local'})
    if r['return']>0: return r
    lst=r['lst']

    for q in lst:
        duid=q['data_uid']
        duoa=q['data_uoa']
        path=q['path']

        ck.out(duoa)

        # Search matching accuracy entry
        aduoa=duoa.replace('-performance-','-accuracy-')

        r=ck.access({'action':'find',
                     'module_uoa':'experiment',
                     'data_uoa':aduoa,
                     'repo_uoa':'local'})
        if r['return']>0: return r
        apath=r['path']             

        # Checking points to aggregate
        dperf=os.listdir(path)
        for f in dperf:
            if f.endswith('.flat.json'):
               ck.out(' * '+f)

               # Load performance file 
               p1=os.path.join(path, f)

               r=ck.load_json_file({'json_file':p1})
               if r['return']>0: return r
               d=r['dict']


               # Save updated dict
               r=ck.save_json_to_file({'json_file':p1, 'dict':d, 'sort_keys':'yes'})
               if r['return']>0: return r

    return {'return':0}

r=do({})
if r['return']>0: ck.err(r)
