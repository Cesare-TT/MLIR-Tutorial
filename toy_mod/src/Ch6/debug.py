#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# **************************************************************************** #
# File: debug.py                                                               #
# Project: Ch6                                                                 #
# Created Date: Wednesday August 30th 2023                                     #
# Author: Qian Taotao                                                          #
# --------------------------------------------------------------------------   #
# Last Modified: Monday September 4th 2023 10:20:47 am                         #
# Modified By: Qian Taotao at <qian.taotao@byd.com>                            #
# --------------------------------------------------------------------------   #
#                           Copyright (c) 2023 BYD                             #
# --------------------------------------------------------------------------   #
# HISTORY:                                                                     #
# Date      	By			Comments                                           #
# ----------	-----------	-------------------------------------------------  #
# ****************************************************************************** #


from subprocess import Popen
from subprocess import STDOUT
from subprocess import PIPE

from pprint import pprint

def dump(stage):
    p = Popen(args=f'./toyc-ch6 ./codegen.toy -emit={stage} -opt ',
              shell=True,
              executable='/usr/bin/bash',
              stdout=PIPE,
              stderr=PIPE
              )

    match stage:
        case 'mlir':        filename='base.mlir'
        case 'mlir-affine': filename='affine.mlir'
        case 'mlir-llvm':   filename='llvm.mlir'
        case 'llvm':        filename='llvm.ll'
        case 'jit':         filename='run.log'

    with open(f'tmp/{filename}', 'w') as f:
        for out in p.communicate():
            f.write(out.decode())


if __name__ == '__main__':
    lowering_stage = ['mlir', 'mlir-affine', 'mlir-llvm', 'llvm', 'jit']
    for stage in lowering_stage:
        dump(stage)
