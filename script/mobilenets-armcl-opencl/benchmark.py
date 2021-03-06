#! /usr/bin/python
import ck.kernel as ck
import copy
import re
import argparse,json
import os

# ReQuEST description.
request_dict={
  'report_uid':'08da9685582866a0', # unique UID for a given ReQuEST submission generated manually by user (ck uid)
                                   # the same UID will be for the report (in the same repo)

  'repo_uoa':'ck-request-asplos18-mobilenets-armcl-opencl',
  'repo_uid':'7698eaf859b79f2b',

  'repo_cmd':'ck pull repo --url=https://github.com/dividiti/ck-request-asplos18-mobilenets-armcl-opencl',

  'farm':'', # if farm of machines

  'algorithm_species':'4b8bbc192ec57f63' # image classification
}

# Some tested experimental platforms.
platform_config={
  "HiKey960\x00": {
    "name":     "Linaro HiKey960",
    "id":       "linaro-hikey960",
    "gpu":      "Mali-G71 MP8",
    "gpu_mhz":  "807 MHz"
  },
  "Rockchip RK3399 Firefly Board (Linux Opensource)\x00": {
    "name":     "Firefly RK3399",
    "id":       "firefly-rk3399",
    "gpu":      "Mali-T860 MP4",
    "gpu_mhz":  "800 MHz"
  },
  "BLA-L09": {
    "name":     "Huawei Mate 10 Pro",
    "id":       "huawei-mate10pro",
    "gpu":      "Mali-G72 MP12",
    "gpu_mhz":  "767 MHz"
  }
}

# ArmCL-specific choices:
# WINOGRAD does not support 1x1 convolutions used in MobileNets.
convolution_methods = ['DEFAULT','GEMM','DIRECT']
default_convolution_method = 'DEFAULT'
# DEFAULT is the library CLTuner which is disabled by default.
kernel_tuners = ['NONE','DEFAULT']
default_kernel_tuner = 'NONE'
# NHWC is only supported from v18.08.
data_layouts = ['NCHW','NHWC']
default_data_layout ='NCHW'

def get_ImageNet_path(dataset_env):
    return dataset_env['meta']['env']['CK_ENV_DATASET_IMAGENET_VAL']

def select_ImageNet():
    res = ck.access({'action':'show',
                     'module_uoa':'env',
                     'tags':'dataset,imagenet,raw,val'})
    if res['return'] > 0:
        return res
    datasets = res.get('lst',[])
    if datasets:
        if len(datasets) == 1:
            return {'return': 0, 'dataset': datasets[0]}

        ck.out('')
        ck.out('More than one ImageNet dataset is found suitable for this script:')
        ck.out('')
        dataset_choices = []
        for d in datasets:
            dataset_choices.append({
                'data_uid': d['data_uid'],
                'data_uoa': get_ImageNet_path(d)
            })
        res = ck.access({'action': 'select_uoa',
                        'module_uoa': 'choice',
                        'choices': dataset_choices})
        if res['return'] > 0:
            return res
        for d in datasets:
            if d['data_uid'] == res['choice']:
                return {'return': 0, 'dataset': d}

    return {'return': 1, 'error': 'No installed ImageNet dataset found'}


def do(i, arg):
    # Process arguments.
    if (arg.accuracy):
        experiment_type = 'accuracy'
        num_repetitions = 1
        kernel_tuners = [default_kernel_tuner]
    else:
        experiment_type = 'performance'
        num_repetitions = arg.repetitions
    random_name = arg.random_name
    share_platform = arg.share_platform

    # Detect basic platform info.
    ii={'action':'detect',
        'module_uoa':'platform',
        'out':'con'}
    if share_platform: ii['exchange']='yes'
    r=ck.access(ii)
    if r['return']>0: return r

    # Keep to prepare ReQuEST meta.
    platform_dict=copy.deepcopy(r)

    # Host and target OS params.
    hos=r['host_os_uoa']
    hosd=r['host_os_dict']

    tos=r['os_uoa']
    tosd=r['os_dict']
    tdid=r['device_id']

    # Determine platform tags: if one of the known platforms, use its id; otherwise, 'unknown-platform'.
    # FIXME: only works when the target platform is the same as the host platform.
    platform_tags=platform_config.get(r['features']['platform']['model'], {'id':'unknown-platform'})['id']

    # The only supported program.
    program='mobilenets-armcl-opencl'

    # Select ImageNet dataset.
    r = select_ImageNet()
    if r['return'] > 0: return r
    imagenet_val = r['dataset']
    img_dir_val = get_ImageNet_path(imagenet_val)
    ck.out('ImageNet path: ' + img_dir_val)

    if arg.accuracy:
        # Use as many batches (of size 1), as there are JPEG images in the directory.
        batch_count = len([f for f in os.listdir(img_dir_val)
           if f.endswith('.JPEG') and os.path.isfile(os.path.join(img_dir_val, f))])
    else:
        # FIXME: Use 2 batches, using the first for warm up (to be excluded from the average)?
        batch_count = 1

    # Restrict accuracy testing to the ReQuEST fork of ArmCL. TODO: restrict to direct convolution for large datasets.
    if arg.accuracy and batch_count > 500:
        use_lib_tags = [ 'request-d8f69c13', '18.05-0acd60ed-request' ]
    else:
        use_lib_tags = [ 'request-d8f69c13', '18.11-b9abeae08', '18.08-52ba29e9', '18.05-0acd60ed-request', '18.05-b3a371bc', '18.03-e40997bb', '18.01-f45d5a9b', '17.12-48bc34ea' ]
    # On Firefly-RK3399, the version hash has only 7 characters, not 8.
    if platform_tags=='firefly-rk3399':
        use_lib_tags = [ tag[:-1] for tag in use_lib_tags ]

    ii={'action':'show',
        'module_uoa':'env',
        'tags':'dataset,imagenet,aux'}
    rx=ck.access(ii)
    if len(rx['lst']) == 0: return rx
    img_dir_aux = rx['lst'][0]['meta']['env']['CK_ENV_DATASET_IMAGENET_AUX']
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

    udepl=r['deps']['library'].get('choices',[]) # All UOAs of env for Arm Compute Libraries.
    if len(udepl)==0:
        return {'return':1, 'error':'no installed Arm Compute Libraries'}
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
          'CK_BATCH_COUNT':batch_count,
          'CK_SKIP_IMAGES':0
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

    experiment_count = 0

    pipeline=copy.deepcopy(r)
    for lib_uoa in udepl:
        # Load ArmCL lib.
        ii={'action':'load',
            'module_uoa':'env',
            'data_uoa':lib_uoa}
        r=ck.access(ii)
        if r['return']>0: return r
        lib_name=r['data_name']
        lib_tags=r['dict']['customize']['version']

        # Skip other libraries if one is explicitly specified.
        if arg.library_uoa and lib_uoa != arg.library_uoa: continue

        # Skip some libs with "in [..]" or "not in [..]".
        if arg.accuracy and lib_tags not in use_lib_tags: continue

        skip_compile='no'
        # For each MobileNets model.*************************************************
        for model_uoa in udepm:
            # Load model.
            ii={'action':'load',
                'module_uoa':'env',
                'data_uoa':model_uoa}
            r=ck.access(ii)
            if r['return']>0: return r
            model_name=r['data_name']

            # Skip other models if one is explicitly specified.
            if arg.model_uoa and model_uoa != arg.model_uoa: continue

            # Skip aggregate MobileNets packages.
            if 'mobilenet-all' in r['dict']['tags']: continue

            batch_size=1

            version=1
            multiplier=float(r['dict']['env']['CK_ENV_MOBILENET_MULTIPLIER'])
            resolution=int(r['dict']['env']['CK_ENV_MOBILENET_RESOLUTION'])

            record_repo='local'
            record_uoa='{}-{}-{}-mobilenet-v{}-{:.2f}-{}'.format(experiment_type, platform_tags, lib_tags, version, multiplier, resolution)

            # Skip the experiment if it already exists.
            if arg.resume:
                r = ck.access({'action':'search',
                               'module_uoa':'experiment',
                               'repo_uoa':record_repo,
                               'data_uoa':record_uoa})
                if r['return']>0: return r
                if len(r['lst']) > 0:
                    ck.out('Experiment "%s" already exists, skipping...' % record_uoa)
                    continue

            # Prepare pipeline.
            ck.out('---------------------------------------------------------------------------------------')
            ck.out('%s - %s' % (lib_name, lib_uoa))
            ck.out('%s - %s' % (model_name, model_uoa))
            ck.out('Experiment - %s:%s' % (record_repo, record_uoa))
            experiment_count += 1

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

            if random_name:
               rx=ck.gen_uid({})
               if rx['return']>0: return rx
               record_uoa=rx['data_uid']

            tags=r['tags']
            tags.append(experiment_type)
            tags.append('explore-mobilenets-'+experiment_type)
            tags.append(lib_tags)
            tags.append(platform_tags)
            tags.append(str(resolution))
            tags.append(str(multiplier))
            tags.append('mobilenet-v{}'.format(version))
            tags.append('mobilenet-v{}-{:.2f}-{}'.format(version, multiplier, resolution))

            ii={'action':'autotune',
               'module_uoa':'pipeline',
               'data_uoa':'program',
               'choices_order':[
                   [
                       '##choices#env#CK_BATCH_SIZE'
                   ],
                   [
                       '##choices#env#CK_ENV_MOBILENET_VERSION'
                   ],
                   [
                       '##choices#env#CK_ENV_MOBILENET_MULTIPLIER'
                   ],
                   [
                       '##choices#env#CK_ENV_MOBILENET_RESOLUTION'
                   ],
                   [
                       '##choices#env#CK_CONVOLUTION_METHOD'
                   ],
                   [
                       '##choices#env#CK_LWS_TUNER_TYPE'
                   ],
                   [
                       '##choices#env#CK_DATA_LAYOUT'
                   ],
               ],
               'choices_selection':[
                   {'type':'loop', 'choice':[batch_size], 'default':1}, # Only batch_size=1 is supported.
                   {'type':'loop', 'choice':[version],    'default':1}, # Only version=1 is supported.
                   {'type':'loop', 'choice':[multiplier], 'default':1.0},
                   {'type':'loop', 'choice':[resolution], 'default':224},
                   {'type':'loop', 'choice':convolution_methods, 'default':default_convolution_method},
                   {'type':'loop', 'choice':kernel_tuners, 'default':default_kernel_tuner},
                   {'type':'loop', 'choice':data_layouts, 'default':default_data_layout},
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

            if not arg.dry_run:
                r=ck.access(ii)
                if r['return']>0: return r
                fail=r.get('fail','')
                if fail=='yes':
                    return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

    if arg.dry_run:
        ck.out('---------------------------------------------------------------------------------------')
        ck.out('Experiment count: %d' % experiment_count)

### end pipeline
    return {'return':0}

##############################################################################################
parser = argparse.ArgumentParser(description='Pipeline')
parser.add_argument("--target_os", action="store", dest="tos")
parser.add_argument("--device_id", action="store", dest="did")
parser.add_argument("--accuracy", action="store_true", default=False, dest="accuracy")
parser.add_argument("--repetitions", action="store", default=10, dest="repetitions")
parser.add_argument("--random_name", action="store_true", default=False, dest="random_name")
parser.add_argument("--share_platform", action="store_true", default=False, dest="share_platform")
parser.add_argument("--dry_run", action="store_true", default=False, dest="dry_run")
parser.add_argument("--library_uoa", action="store", default='', dest="library_uoa")
parser.add_argument("--model_uoa", action="store", default='', dest="model_uoa")
parser.add_argument("--resume", action="store_true", default=False, dest="resume")

myarg=parser.parse_args()

r=do({}, myarg)
if r['return']>0: ck.err(r)
