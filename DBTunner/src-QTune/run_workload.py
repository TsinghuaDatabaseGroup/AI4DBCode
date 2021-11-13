from configs import parse_args
from get_workload_from_file import get_workload_from_file
from run_job import run_job
import time

if __name__ == "__main__":

    argus = parse_args()

    # prepare_training_workloads
    workload = get_workload_from_file(argus["workload_file_path"])

    file_path = 'training-results/res_no_change-' + str(int(time.time()))
    for i in range(0, 3):
        pfs = open(file_path, 'a')
        avg_qps, avg_lat = run_job(int(argus["thread_num"]), workload)
        pfs.write("%d\t%s\t%s\n" % (i, avg_qps, avg_lat))
        pfs.close()

