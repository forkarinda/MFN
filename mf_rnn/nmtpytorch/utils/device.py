import re
import os
import shutil
import subprocess

import torch

DEVICE = None
DEVICE_IDS = None


class DeviceManager:
    __errors = {
        'BadDeviceFormat': 'Device can be cpu, gpu or [N]gpu, i.e. 2gpu',
        'NoDevFiles': 'Make sure you requested a GPU resource from your cluster.',
        'NoSMI': 'nvidia-smi is not installed. Are you on the correct node?',
        'EnvVar': 'Please set CUDA_VISIBLE_DEVICES explicitly.',
        'NoMultiGPU': 'Multi-GPU not supported for now.',
        'NotEnoughGPU': 'You requested {} GPUs while you have access to only {}.',
    }

    def __init__(self, dev):
        self.dev = dev.lower()
        self.pid = os.getpid()
        self.req_cpu = False
        self.req_gpu = False
        self.req_n_gpu = 0
        self.req_multi_gpu = False
        self.nvidia_smi = False
        self.cuda_dev_ids = None
        
        if not re.match('(cpu|[0-9])$', self.dev):
            raise RuntimeError(self.__errors['BadDeviceFormat'])

        if self.dev == 'cpu':
            self.req_cpu = True
            self.dev = torch.device('cpu')
        else:
            self.req_gpu = True
            self.req_n_gpu = 1
            # Set master device (is always cuda:0 since we force env.var
            # restriction)
            self.cuda_dev_ids = self.dev
            self.dev = torch.device('cuda:'+self.dev)

            if self.nvidia_smi is None:
                raise RuntimeError(self.__errors['NoSMI'])
            if self.cuda_dev_ids == "NoDevFiles":
                raise RuntimeError(self.__errors['NoDevFiles'])
            elif self.cuda_dev_ids is None:
                raise RuntimeError(self.__errors['EnvVar'])
            
            # How many GPUs do we have access to?
            self.cuda_dev_ids = [int(de) for de in self.cuda_dev_ids.split(',')]
            
         
            global DEVICE, DEVICE_IDS
            DEVICE = self.dev
            DEVICE_IDS = self.cuda_dev_ids

    def get_cuda_mem_usage(self, name=True):
        if self.req_cpu:
            return None

        pr = subprocess.run([
            self.nvidia_smi,
            "--query-compute-apps=pid,gpu_name,used_memory",
            "--format=csv,noheader"], stdout=subprocess.PIPE, universal_newlines=True)

        for line in pr.stdout.strip().split('\n'):
            pid, gpu_name, usage = line.split(',')
            if int(pid) == self.pid:
                if name:
                    return '{} -> {}'.format(gpu_name.strip(), usage.strip())
                return usage.strip()

        return 'N/A'

    def __repr__(self):
        if self.req_cpu:
            return "DeviceManager(dev='cpu')"
        return "DeviceManager({}, n_gpu={})".format(self.dev, self.req_n_gpu)
