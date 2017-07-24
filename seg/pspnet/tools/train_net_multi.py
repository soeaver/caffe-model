import sys
from multiprocessing import Process

# sys.setrecursionlimit(100000)

sys.path.append('/home/prmct/workspace/py-RFCN-priv/caffe-priv/python')
# sys.path.append('/home/yanglu/workspace/py-R-FCN-multiGPU-master-0619/caffe/python')
import caffe

# _snapshot='./aug_single_resnet101_iter_5000.solverstate'
_weights = '/home/prmct/Program/classification/ilsvrc/resnet_v2/resnet101_v2/resnet101_v2_merge_bn_scale.caffemodel'

solver_prototxt = './solver.prototxt'
gpus = [0,1,2,3]
max_iter = 200000

def solve(proto, gpus, uid, rank, max_iter):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(proto)
    if rank == 0:
        # solver.restore(_snapshot)
        solver.net.copy_from(_weights)
    
    solver.net.layers[0].get_gpu_id(gpus[rank])

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()
    solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)

    for _ in range(max_iter):
        solver.step(1)


if __name__ == '__main__':
    uid = caffe.NCCL.new_uid()
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []

    for rank in range(len(gpus)):
        p = Process(target=solve,
                    args=(solver_prototxt, gpus, uid, rank, max_iter))
        p.daemon = False
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

