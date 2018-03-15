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

               p2=os.path.join(path, f[:-10]+'.features_flat.json') # Features

               r=ck.load_json_file({'json_file':p2})
               if r['return']>0: return r
               df=r['dict']

               # Remove batch
               del(df['##choices#env#CK_BATCH_COUNT'])
               
               # Find matching features file to merge
               dacc=os.listdir(apath)
               matched=False
               for af in dacc:
                   if af.endswith('.features_flat.json'):
                      r=ck.load_json_file({'json_file':os.path.join(apath,af)})
                      if r['return']>0: return r
                      adf=r['dict']

                      # Remove batch
                      del(adf['##choices#env#CK_BATCH_COUNT'])

                      # Compare dicts
                      r=ck.compare_dicts({'dict1':df, 'dict2':adf})
                      if r['return']>0: return r
                      if r['equal']=='yes':
                         matched=True

                         # Load accuracy data to merge
                         px=os.path.join(apath,af[:-19]+'.flat.json')
                         r=ck.load_json_file({'json_file':px})
                         if r['return']>0: return r
                         dd=r['dict']

                         # Merge keys
                         for k in dd:
                             if k.startswith('##characteristics#run#accuracy_top'):
                                d[k]=dd[k]

                         break
               
               if not matched:
                  return {'return':1, 'error':'no match - strange'}

               # Save updated dict
               r=ck.save_json_to_file({'json_file':p1, 'dict':d, 'sort_keys':'yes'})
               if r['return']>0: return r

    return {'return':0}

r=do({})
if r['return']>0: ck.err(r)
