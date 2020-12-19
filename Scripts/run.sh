#!/bin/bash
#PBS -N NER
#PBS -l select:1:ncpus=1:ngpus=1:gpu_cap=cuda75:mem=16gb:scratch_local=20gb
#PBS -l walltime=1:00:00
#PBS -q @adan.grid.cesnet.cz
#PBS -m abe

DATADIR=/storage/plzen1/home/amistera

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

module add python-3.6.2-gcc
module add conda-modules-py37
module add cuda-10.1

cp -r $DATADIR/ANLP $SCRATCHDIR || { echo >&2 "Error while copying data!"; exit 1; }

python3 main.py 

clean_scratch