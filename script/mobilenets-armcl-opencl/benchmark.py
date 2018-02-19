#! /usr/bin/python
import ck.kernel as ck
import copy
import re
import argparse,json
import os

# ReQuEST tournament tag
request_tag1='request'
request_tag2='request-asplos18'

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
    num_repetitions=arg.repetitions
    experiment_type = 'performance'
    batch_count = 2
    if (arg.accuracy):
        experiment_type = 'accuracy'


    # Detect basic platform info.
    ii={'action':'detect',
        'module_uoa':'platform',
        'out':'out'}
    r=ck.access(ii)
    if r['return']>0: return r

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

               'record_dict':{
                     'subview_uoa':'f84ca49f79a1446a',    # ReQuEST default table view from ck-autotuning repo
                                                          # ck-autotuning:experiment.view:request-default

                     'program_species':'4b8bbc192ec57f63' # ck-autotuning:program.species:image.classification
                                                          # common class of apps
               },  

               'record_repo':record_repo,
               'record_uoa':record_uoa,

               'tags':[request_tag1, request_tag2, experiment_type, 'explore-mobilenets-'+experiment_type, lib_tags, platform_tags, str(rho), str(alpha)],

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


parser = argparse.ArgumentParser(description='Pipeline')
parser.add_argument("--target_os", action="store", dest="tos")
parser.add_argument("--device_id", action="store", dest="did")
parser.add_argument("--accuracy", action="store_true", default=False, dest="accuracy")
parser.add_argument("--repetitions", action="store", default=3, dest="repetitions")

myarg=parser.parse_args()


r=do({}, myarg)
if r['return']>0: ck.err(r)
 
