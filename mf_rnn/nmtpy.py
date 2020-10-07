#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse

lib_path = os.path.abspath('../')
sys.path.append(lib_path)
from nmtpytorch import logger
from nmtpytorch.config import Options
from nmtpytorch.utils.misc import setup_experiment, fix_seed
from nmtpytorch.utils.device import DeviceManager

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='nmtpy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="nmtpy trains/translates/tests a given "
                    "configuration/checkpoint/snapshot",
        argument_default=argparse.SUPPRESS)

    subparsers = parser.add_subparsers(dest='cmd', title='sub-commands',
                                       description='Valid sub-commands')

    # train command
    parser_train = subparsers.add_parser('train', help='train help')
    parser_train.add_argument('-C', '--config', type=str, default='configs/mfn_rnn.conf',
                              help="Experiment configuration file")
    parser_train.add_argument('-s', '--suffix', type=str, default="",
                              help="Optional experiment suffix.")
    parser_train.add_argument('-S', '--short', action='store_true',
                              help="Use short experiment id in filenames.")
    parser_train.add_argument('-d', '--device-id', type=str, default='0',
                              help='cpu or cuda: [N]  (Default: 0), not support parallel work')
    parser_train.add_argument('overrides', nargs="*", default=[],
                              help="(section).key:value overrides for config")

    ###################
    # translate command
    ###################
    parser_trans = subparsers.add_parser('translate', help='translate help')
    parser_trans.add_argument('-u', '--suppress-unk', action='store_true',
                              help='Do not generate <unk> tokens.')
    parser_trans.add_argument('-n', '--disable-filters', action='store_true',
                              help='Disable eval_filters given in config')
    parser_trans.add_argument('-N', '--n-best', action='store_true',
                              help='Generate n-best list of beam candidates.')
    parser_trans.add_argument('-b', '--batch-size', type=int, default=32,
                              help='Batch size for beam-search')
    parser_trans.add_argument('-k', '--beam-size', type=int, default=6,
                              help='Beam size for beam-search')
    parser_trans.add_argument('-m', '--max-len', type=int, default=200,
                              help='Maximum sequence length to produce (Default: 200)')
    parser_trans.add_argument('-a', '--lp-alpha', type=float, default=0.,
                              help='Apply length-penalty (Default: 0.)')
    parser_trans.add_argument('-d', '--device-id', type=str, default='1',
                              help='cpu or cuda: [N]  (Default: 0)')
    parser_trans.add_argument('models', type=str, nargs='+',
                              help="Saved model/checkpoint file(s)")
    parser_trans.add_argument('-tid', '--task-id', type=str, default='tran:Text, image:NumpySequence -> desc:Text',
                              help='Task to perform, e.g. en_src:Text, pt_src:Text -> fr_tgt:Text') 
    parser_trans.add_argument('-x', '--override', nargs="*", default=[],
                              help="(section).key:value overrides for config")

    # You can translate a set of splits defined in the .conf file
    parser_trans.add_argument('-s', '--splits', type=str, default='test',
                              help='Comma separated splits from config file')
    # Or you can provide another input configuration with -S
    # With this, the previous -s is solely used for the naming of the output file
    # and you can only give 1 split with -s that corresponds to new input you
    # define with -S.
    parser_trans.add_argument('-S', '--source', type=str, default=None,
                              help='Comma-separated key:value pairs to provide new inputs.')
    parser_trans.add_argument('-o', '--output', type=str,default = 'output.txt',
                              help='Output filename prefix')

    ##############
    # test command
    ##############
    parser_test = subparsers.add_parser('test', help='test help')
    parser_test.add_argument('-b', '--batch-size', type=int, default=64,
                             help='Batch size for beam-search')
    parser_test.add_argument('-d', '--device-id', type=str, default='0',
                             help='cpu or cuda: [N]  (Default: 0)')
    parser_test.add_argument('models', type=str, nargs='+',
                             help="Saved model/checkpoint file(s)")
    parser_test.add_argument('-tid', '--task-id', type=str, default='tran:Text, image:NumpySequence -> desc:Text',
                             help='Task to perform, e.g. en_src:Text, pt_src:Text -> fr_tgt:Text')
    parser_test.add_argument('-x', '--override', nargs="*", default=[],
                             help="(section).key:value overrides for config")
    parser_test.add_argument('-m', '--mode', choices=['eval', 'enc'],
                             default='eval',
                             help="Perform evaluation or dump encodings.")
    parser_test.add_argument('-s', '--splits', type=str, default = 'test',
                             help='Comma separated splits from config file')
    parser_test.add_argument('-S', '--source', type=str, default = None,
                             help='Comma-separated key:value pairs to provide new inputs.')

    # Parse command-line arguments first
    args = parser.parse_args()


    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    # Mode selection
    if args.cmd == 'train':
        # Parse configuration file and merge with the rest
        opts = Options(args.config, args.overrides)

        # Setup experiment folders
        setup_experiment(opts, args.suffix, args.short)

    # Reserve device(s)
    dev_mgr = DeviceManager(args.device_id)

    # translate entry point
    if args.cmd in ('translate', 'test'):
        logger.setup()
        fix_seed(1234)
        cmd = args.__dict__.pop('cmd')
        if cmd == 'translate':
            from nmtpytorch.translator import Translator
            translator = Translator(**args.__dict__)
            translator()
        elif cmd == 'test':
            from nmtpytorch.tester import Tester
            tester = Tester(**args.__dict__)
            tester()
        sys.exit(0)

    #################################
    # Training / Resuming entry point
    #################################
    import torch
    import platform
    import nmtpytorch
    from nmtpytorch import models
    from nmtpytorch.mainloop import MainLoop
    log = logger.setup(opts.train)

    # If given, seed that; if not generate a random seed and print it
    if opts.train['seed'] > 0:
        seed = fix_seed(opts.train['seed'])
    else:
        opts.train['seed'] = fix_seed()

    # Be verbose and fire the loop!
    log.info(opts)

    # Instantiate the model object
    model = getattr(models, opts.train['model_type'])(opts=opts)

    log.info("Python {} -- torch {} with CUDA {} (on machine '{}')".format(
        platform.python_version(), torch.__version__,
        torch.version.cuda, platform.node()))
    log.info("nmtpytorch {}".format(nmtpytorch.__version__))
    log.info(dev_mgr)
    log.info("Seed for further reproducibility: {}".format(opts.train['seed']))

    if 'SLURM_JOB_ID' in os.environ:
        log.info("SLURM Job ID: {}".format(os.environ['SLURM_JOB_ID']))
    loop = MainLoop(model, opts.train, dev_mgr)
    loop()
    sys.exit(0)
