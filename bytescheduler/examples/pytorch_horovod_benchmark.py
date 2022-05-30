from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import horovod.torch as hvd
import timeit
import numpy as np
import os

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument('--partition', type=int, default=None,
                    help='partition size')

nvtx = torch.cuda.nvtx
nvtx.range_push("Preparation Phase")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
nvtx.range_push("Hvd Init")
hvd.init()
nvtx.range_pop() # Hvd Init Done
if args.cuda:
    # Horovod: pin GPU to local rank.
    nvtx.range_push("Pin to GPU {}".format(hvd.local_rank()))
    torch.cuda.set_device(hvd.local_rank())
    nvtx.range_pop() # Pin to GPU Done

cudnn.benchmark = True

# Set up standard model.
nvtx.range_push("Setup model {} with {} classes".format(args.model, args.num_classes))
model = getattr(models, args.model)(num_classes=args.num_classes)
nvtx.range_pop() # Setup model Done

if args.cuda:
    # Move model to GPU.
    nvtx.range_push("Move {} to GPU".format(args.model))  
    model.cuda()
    nvtx.range_pop() # Move model Done

optimizer = optim.SGD(model.parameters(), lr=0.01)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# bytescheduler wrapper
use_bytescheduler = int(os.environ.get('USE_BYTESCHEDULER', '0'))
if use_bytescheduler > 0:
    nvtx.range_push("BSC Init")
    if args.partition:
        os.environ["BYTESCHEDULER_PARTITION"] = str(1000 * args.partition)
    import bytescheduler.pytorch.horovod as bsc
    bsc.init()
    nvtx.range_pop() # BSC Init Done

# Horovod: wrap optimizer with DistributedOptimizer.
nvtx.range_push("Wrap DistributedOptimizer")
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=model.named_parameters(),
                                     compression=compression)
nvtx.range_pop() # Wrap DistributedOptimizer Done
if use_bytescheduler > 0:
    nvtx.range_push("Wrap ScheduledOptimizer")
    optimizer = bsc.ScheduledOptimizer(model, optimizer, args.num_warmup_batches + args.num_iters * args.num_batches_per_iter)
    nvtx.range_pop() # Wrap ScheduledOptimizer Done

# Horovod: broadcast parameters & optimizer state.
nvtx.range_push("BCAST parameters & opt state")
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
nvtx.range_pop() # BCAST parameters & opt state Done

nvtx.range_push("Setup Data")
# Set up fake data
datasets = []
for _ in range(100):
    data = torch.rand(args.batch_size, 3, 224, 224)
    target = torch.LongTensor(args.batch_size).random_() % 1000
    if args.cuda:
        data, target = data.cuda(), target.cuda()
    datasets.append(data)
data_index = 0
nvtx.range_pop() # Setup Data Done
nvtx.range_pop() # Preparation Phase Done

def benchmark_step():
    global data_index

    data = datasets[data_index%len(datasets)]
    data_index += 1
    nvtx.range_push("FP")
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, target)
    nvtx.range_pop() # FP Done
    nvtx.range_push("BP")
    nvtx.range_push("Calc Grads")
    loss.backward()
    nvtx.range_pop() # Calc Grads Done
    nvtx.range_push("Update Parms")
    optimizer.step()
    nvtx.range_pop() # Update Parms Done
    nvtx.range_pop() # BP Done


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))

# Warm-up
nvtx.range_push("Warmup Phase")
log('Running warmup...')
timeit.timeit(benchmark_step, number=args.num_warmup_batches)
nvtx.range_pop() # Warmup Phase Done
    

# Benchmark
log('Running benchmark...')
img_secs = []
enable_profiling = args.profiler & (hvd.rank() == 0)

with torch.autograd.profiler.profile(True) as prof:
    for x in range(args.num_iters):
        nvtx.range_push("Iteration {}".format(x))
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)
        nvtx.range_pop()
if enable_profiling:
    prof.export_chrome_trace(os.path.join('pytorch-trace', args.model+'-'+str(hvd.rank()) +'.json'))
    # print(prof)
# Results
img_sec_mean = np.mean(img_secs)
img_sec_conf = 1.96 * np.std(img_secs)
log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
log('Total img/sec on %d %s(s): %.1f +-%.1f' %
    (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
