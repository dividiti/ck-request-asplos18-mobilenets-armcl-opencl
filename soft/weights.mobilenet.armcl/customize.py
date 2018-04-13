#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
# Collective Knowledge (individual environment - setup)
#

import os

##############################################################################
# setup environment setup

def setup(i):
    """
    Input:  {
              cfg              - meta of this soft entry
              self_cfg         - meta of module soft
              ck_kernel        - import CK kernel module (to reuse functions)

              host_os_uoa      - host OS UOA
              host_os_uid      - host OS UID
              host_os_dict     - host OS meta

              target_os_uoa    - target OS UOA
              target_os_uid    - target OS UID
              target_os_dict   - target OS meta

              target_device_id - target device ID (if via ADB)

              tags             - list of tags used to search this entry

              env              - updated environment vars from meta
              customize        - updated customize vars from meta

              deps             - resolved dependencies for this soft

              interactive      - if 'yes', can ask questions, otherwise quiet
            }

    Output: {
              return       - return code =  0, if successful
                                         >  0, if error
              (error)      - error text if return > 0

              bat          - prepared string for bat file
            }

    """
    ck=i['ck_kernel']

    iv=i.get('interactive','')

    env=i.get('env',{})
    cfg=i.get('cfg',{})
    deps=i.get('deps',{})
    tags=i.get('tags',[])
    cus=i.get('customize',{})

    target_d=i.get('target_os_dict',{})
    win=target_d.get('windows_base','')
    remote=target_d.get('remote','')
    mingw=target_d.get('mingw','')
    tbits=target_d.get('bits','')

    envp=cus.get('env_prefix','')
    ienv=cus.get('install_env', {})

    host_d=i.get('host_os_dict',{})
    sdirs=host_d.get('dir_sep','')

    fp=cus.get('full_path','')
    pi=os.path.dirname(fp)

    ep = cus['env_prefix']
    env[ep] = pi
    if 'MOBILENET_VERSION' in ienv:
      env[ep + "_VERSION"] = ienv['MOBILENET_VERSION']
    if 'MOBILENET_MULTIPLIER' in ienv:
      env[ep + "_MULTIPLIER"] = ienv['MOBILENET_MULTIPLIER']
    if 'MOBILENET_RESOLUTION' in ienv:
      env[ep + "_RESOLUTION"] = ienv['MOBILENET_RESOLUTION']

    return {'return': 0, 'bat': ''}
