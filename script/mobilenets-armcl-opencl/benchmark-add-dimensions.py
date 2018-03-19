#! /usr/bin/python
import ck.kernel as ck
import os

def do(i):

    # List performance entries
    r=ck.access({'action':'search',
                 'module_uoa':'experiment',
                 'data_uoa':'mobilenets-performance-*'})
    if r['return']>0: return r
    lst=r['lst']

    for q in lst:
        duid=q['data_uid']
        duoa=q['data_uoa']
        ruid=q['repo_uid']
        path=q['path']

        ck.out(duoa)

        # Search matching accuracy entry
        r=ck.access({'action':'load',
                     'module_uoa':'experiment',
                     'data_uoa':duid,
                     'repo_uoa':ruid})
        if r['return']>0: return r

        dd=r['dict']
        ruid=r['repo_uid']
        apath=r['path']             

        # Updating meta if needed
        dd['meta']['scenario_module_uoa']='a555738be4b65860' # module:request.asplos18

        dd['meta']['model_species']='07d4e7aa3750ddc6' # model.species:mobilenets

        dd['meta']['dataset_species']='ImageNet' # dataset species (free format)
        dd['meta']['dataset_size']=500 # number of images ...

        dd['meta']['platform_species']='embedded' # embedded vs server (maybe other classifications such as edge)

        dd['meta']['platform_peak_power']=4.5 #Watts
        dd['meta']['platform_price']=239 # $
        dd['meta']['platform_price_date']='20170425' # date

        dd['meta']['artifact']='08da9685582866a0' # artifact description

        # Updating entry
        r=ck.access({'action':'update',
                     'module_uoa':'experiment',
                     'data_uoa':duid,
                     'repo_uoa':ruid,
                     'dict':dd,
                     'substitute':'yes',
                     'ignore_update':'yes',
                     'sort_keys':'yes'
                    })
        if r['return']>0: return r

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

               mult=d.get('##choices#env#CK_ENV_MOBILENET_WIDTH_MULTIPLIER#min','')

               if mult==0.25: size=1990786
               elif mult==0.5: size=5459810
               elif mult==0.75: size=10498594
               elif mult==1.0: size=17106694
               else:
                  return {'return':1, 'error':'unknown width multiplier "'+str(mult)+'"'}

               d['##features#model_size#min']=size

               # Save updated dict
               r=ck.save_json_to_file({'json_file':p1, 'dict':d, 'sort_keys':'yes'})
               if r['return']>0: return r

    return {'return':0}

r=do({})
if r['return']>0: ck.err(r)
