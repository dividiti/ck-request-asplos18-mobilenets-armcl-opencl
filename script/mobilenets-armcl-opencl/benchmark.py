#! /usr/bin/python
import ck.kernel as ck
import copy
import re
import argparse,json
import os

# ReQuEST description
request_dict={
  'report_uid':'08da9685582866a0', # unique UID for a given ReQuEST submission generated manually by user (ck uid)
                                   # the same UID will be for the report (in the same repo)

  'repo_uoa':'ck-request-asplos18-mobilenets-armcl-opencl',
  'repo_uid':'7698eaf859b79f2b',

  'repo_cmd':'ck pull repo --url=https://github.com/dividiti/ck-request-asplos18-mobilenets-armcl-opencl',

  'algorithm_species':'4b8bbc192ec57f63'
}

# User vars
# Platform tag.
platform_tags='hikey-960'

# Batch size.
bs={
  'start':1,
  'stop':1,
  'step':1,
  'default':1
}

def do(i, arg):
#### Default values
    # Process vars
    num_repetitions=arg.repetitions
    experiment_type = 'performance'
    batch_count = 2
    if (arg.accuracy):
        experiment_type = 'accuracy'
    random_name=arg.random_name
    share_platform=arg.share_platform

    # Detect basic platform info.
    ii={'action':'detect',
        'module_uoa':'platform',
        'out':'con'}
    if share_platform: ii['exchange']='yes'
    r=ck.access(ii)
    if r['return']>0: return r

    # Keep to prepare ReQuEST meta
    platform_dict=copy.deepcopy(r)

    # Host and target OS params.
    hos=r['host_os_uoa']
    hosd=r['host_os_dict']

    tos=r['os_uoa']
    tosd=r['os_dict']
    tdid=r['device_id']

    program='mobilenets-armcl-opencl'

    ii={'action':'show',
        'module_uoa':'env',
        'tags':'dataset,imagenet,raw,val'}

    rx=ck.access(ii)
    if len(rx['lst']) == 0: return rx
    # FIXME: It's probably better to use CK_ENV_DATASET_IMAGE_DIR.
    img_dir_val = rx['lst'][0]['meta']['env']['CK_CAFFE_IMAGENET_VAL']

    if (arg.accuracy):
        batch_count = len([f for f in os.listdir(img_dir_val) 
           if f.endswith('.JPEG') and os.path.isfile(os.path.join(img_dir_val, f))])

    ii={'action':'show',
        'module_uoa':'env',
        'tags':'dataset,imagenet,aux'}
    rx=ck.access(ii)
    if len(rx['lst']) == 0: return rx
    img_dir_aux = rx['lst'][0]['meta']['env']['CK_ENV_DATASET_IMAGENET_AUX']
#    print img_dir_aux
    ii={'action':'load',
        'module_uoa':'program',
        'data_uoa':program}
    rx=ck.access(ii)
    if rx['return']>0: return rx
    mm=rx['dict']
    # Get compile-time and run-time deps.
    cdeps=mm.get('compile_deps',{})
    rdeps=mm.get('run_deps',{})

    # Merge rdeps with cdeps for setting up the pipeline (which uses
    # common deps), but tag them as "for_run_time".
    for k in rdeps:
        cdeps[k]=rdeps[k]
        cdeps[k]['for_run_time']='yes'
#    print cdeps
    depl=copy.deepcopy(cdeps['library'])
    if (arg.tos is not None) and (arg.did is not None):
        tos=arg.tos
        tdid=arg.did 

    ii={'action':'resolve',
        'module_uoa':'env',
        'host_os':hos,
        'target_os':tos,
        'device_id':tdid,
        'out':'con',
        'deps':{'library':copy.deepcopy(depl)},
        'quiet':'yes'
    }
    r=ck.access(ii)
    if r['return']>0: return r

    udepl=r['deps']['library'].get('choices',[]) # All UOAs of env for ARM Compute libs.
    if len(udepl)==0:
        return {'return':1, 'error':'no installed ARM Compute libs'}
    cdeps['library']['uoa']=udepl[0]
    depm=copy.deepcopy(cdeps['weights'])

    ii={'action':'resolve',
        'module_uoa':'env',
        'host_os':hos,
        'target_os':tos,
        'device_id':tdid,
        'out':'con',
        'deps':{'weights':copy.deepcopy(depm)},
        'quiet':'yes'
    }
    r=ck.access(ii)
    if r['return']>0: return r

    udepm=r['deps']['weights'].get('choices',[])
    if len(udepm)==0:
        return {'return':1, 'error':'no installed Weights'}
    cdeps['library']['uoa']=udepl[0]
    cdeps['weights']['uoa']=udepm[0]

    ii={'action':'pipeline',
        'prepare':'yes',
        'dependencies':cdeps,

        'module_uoa':'program',
        'data_uoa':program,

        'target_os':tos,
        'device_id':tdid,

        'no_state_check':'yes',
        'no_compiler_description':'yes',
        'skip_calibration':'yes',

        'env':{
          'CK_ENV_DATASET_IMAGENET_VAL':img_dir_val,
          'CK_BATCH_COUNT': batch_count,
          'CK_BATCHES_DIR':'batches',
          'CK_BATCH_LIST':'batches.txt',
          'CK_IMG_LIST':'images.txt',
          'CK_RESULTS_DIR': 'predictions',
          'CK_SKIP_IMAGES': 0
        },

        'cpu_freq':'max',
        'gpu_freq':'max',

        'flags':'-O3',
        'speed':'no',
        'energy':'no',

        'skip_print_timers':'yes',
        'out':'con'
    }

    r=ck.access(ii)
    if r['return']>0: return r
    fail=r.get('fail','')
    if fail=='yes':
        return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

    ready=r.get('ready','')
    if ready!='yes':
        return {'return':11, 'error':'pipeline not ready'}

    state=r['state']
    tmp_dir=state['tmp_dir']

    # Remember resolved deps for this benchmarking session.
    xcdeps=r.get('dependencies',{})
    # Clean pipeline.
    if 'ready' in r: del(r['ready'])
    if 'fail' in r: del(r['fail'])
    if 'return' in r: del(r['return'])

    pipeline=copy.deepcopy(r)
    for lib_uoa in udepl:
        # Load Tensorflow lib.
        ii={'action':'load',
            'module_uoa':'env',
            'data_uoa':lib_uoa}
        r=ck.access(ii)
        if r['return']>0: return r
        # Get the tags from e.g. 'TensorFlow library (from sources, cuda)'
        lib_name=r['data_name']
        lib_tags=re.match('ARM Compute Library \((?P<tags>.*)\)', lib_name)        
        lib_tags=lib_tags.group('tags').replace(' ', '').replace(',', '-')
        # Skip some libs with "in [..]" or "not in [..]".
        skip_compile='no'
        # For each Weights .*************************************************
        for model_uoa in udepm:
            # Load Caffe model.
            ii={'action':'load',
                'module_uoa':'env',
                'data_uoa':model_uoa}
            r=ck.access(ii)
            if r['return']>0: return r
            # Get the tags from e.g. 'Caffe model (net and weights) (deepscale, squeezenet, 1.1)'
            model_name=r['data_name']
            alpha = float(r['dict']['env']['CK_ENV_MOBILENET_MULTIPLIER'])
            rho = int(r['dict']['env']['CK_ENV_MOBILENET_RESOLUTION'])
            record_repo='local'
            record_uoa='mobilenets-'+experiment_type+'-'+str(rho)+'-'+str(alpha)+'-'+lib_tags+'-'+lib_uoa

            # Prepare pipeline.
            ck.out('---------------------------------------------------------------------------------------')
            ck.out('%s - %s' % (lib_name, lib_uoa))
            ck.out('%s - %s' % (model_name, model_uoa))
            ck.out('Experiment - %s:%s' % (record_repo, record_uoa))

            # Prepare autotuning input.
            cpipeline=copy.deepcopy(pipeline)
            # Reset deps and change UOA.
            new_deps={'library':copy.deepcopy(depl),
                      'weights':copy.deepcopy(depm)}

            new_deps['library']['uoa']=lib_uoa
            new_deps['weights']['uoa']=model_uoa
            jj={'action':'resolve',
                'module_uoa':'env',
                'host_os':hos,
                'target_os':tos,
                'device_id':tdid,
                'deps':new_deps}
            r=ck.access(jj)
            if r['return']>0: return r

            cpipeline['dependencies'].update(new_deps)

            cpipeline['no_clean']=skip_compile
            cpipeline['no_compile']=skip_compile

            # Prepare common meta for ReQuEST tournament
            features=copy.deepcopy(cpipeline['features'])
            platform_dict['features'].update(features)

            r=ck.access({'action':'prepare_common_meta',
                         'module_uoa':'request.asplos18',
                         'platform_dict':platform_dict,
                         'deps':cpipeline['dependencies'],
                         'request_dict':request_dict})
            if r['return']>0: return r

            record_dict=r['record_dict']

            meta=r['meta']

#            record_uoa+='-'+meta['stimestamp']

            if random_name:
               rx=ck.gen_uid({})
               if rx['return']>0: return rx
               record_uoa=rx['data_uid']

            tags=r['tags']

            tags.append(experiment_type)

            tags.append('explore-mobilenets-'+experiment_type)
            tags.append(lib_tags)
            tags.append(platform_tags)
            tags.append(str(rho))
            tags.append(str(alpha))

            ii={'action':'autotune',
               'module_uoa':'pipeline',
               'data_uoa':'program',
               'choices_order':[
                           [
                               '##choices#env#CK_BATCH_SIZE'
                           ],
                           [
                               '##choices#env#CK_ENV_MOBILENET_RESOLUTION'
                           ],
                           [
                               '##choices#env#CK_ENV_MOBILENET_WIDTH_MULTIPLIER'
                           ]
               ],
               'choices_selection':[
                           {'type':'loop', 'start':bs['start'], 'stop':bs['stop'], 'step':bs['step'], 'default':bs['default']},
                           {'type':'loop', 'choice': [rho], 'default': 224},
                           {'type':'loop', 'choice': [alpha], 'default': 1.0},

               ],

               'features_keys_to_process':['##choices#*'],

               'iterations':-1,
               'repetitions': num_repetitions,

               'record':'yes',
               'record_failed':'yes',

               'record_params':{
                   'search_point_by_features':'yes'
               },

               'tags':tags,
               'meta':meta,

               'record_dict':record_dict,

               'record_repo':record_repo,
               'record_uoa':record_uoa,

               'pipeline':cpipeline,
               'out':'con'
            }
            r=ck.access(ii)
            if r['return']>0: return r

            fail=r.get('fail','')
            if fail=='yes':
                return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

### end pipeline
    return {'return':0}

##############################################################################################
parser = argparse.ArgumentParser(description='Pipeline')
parser.add_argument("--target_os", action="store", dest="tos")
parser.add_argument("--device_id", action="store", dest="did")
parser.add_argument("--accuracy", action="store_true", default=False, dest="accuracy")
parser.add_argument("--repetitions", action="store", default=3, dest="repetitions")
parser.add_argument("--random_name", action="store_true", default=False, dest="random_name")
parser.add_argument("--share_platform", action="store_true", default=False, dest="share_platform")

myarg=parser.parse_args()

r=do({}, myarg)
if r['return']>0: ck.err(r)
