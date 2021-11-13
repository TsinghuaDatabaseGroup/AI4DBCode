import argparse
import os

parser = argparse.ArgumentParser(description='generatedata')
parser.add_argument('--cols', type=int, help='datasets_dir', default=11)
parser.add_argument('--xt', type=int, help='datasets_dir', default=11)
args = parser.parse_args()
cols = args.cols
xt = args.xt

if cols == 2:
    os.chdir('./job_2')
    os.system('python run.py --run job-light-ranges')
    os.chdir('..')
elif cols == 4:
    os.chdir('./job_4')
    os.system('python run.py --run job-light-ranges')
    os.chdir('..')
elif cols == 6:
    os.chdir('./job_6')
    os.system('python run.py --run job-light-ranges')
    os.chdir('..')
elif cols == 8:
    os.chdir('./job_8')
    os.system('python run.py --run job-light-ranges')
    os.chdir('..')

elif xt == 2:
    os.chdir('./xt_2')
    os.system('python run.py --run job-light-ranges')
    os.chdir('..')
elif xt == 4:
    os.chdir('./xt_4')
    os.system('python run.py --run job-light-ranges')
    os.chdir('..')
elif xt == 6:
    os.chdir('./xt_6')
    os.system('python run.py --run job-light-ranges')
    os.chdir('..')
elif xt == 8:
    os.chdir('./xt_8')
    os.system('python run.py --run job-light-ranges')
    os.chdir('..')
elif xt == 10:
    os.chdir('./xt_10')
    os.system('python run.py --run job-light-ranges')
    os.chdir('..')
