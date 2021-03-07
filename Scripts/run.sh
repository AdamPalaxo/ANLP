#!/bin/bash
#PBS -N NER
#PBS -l select=1:ncpus=1:ngpus=1:gpu_cap=cuda75:mem=16gb:scratch_local=20gb
#PBS -l walltime=23:59:00
#PBS -q gpu
#PBS -m abe

DATADIR=/storage/plzen1/home/amistera

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# Load conda
module add conda-modules-py37
conda activate $DATADIR/conda

# Copy Data
cp -r $DATADIR/ANLP $SCRATCHDIR || { echo >&2 "Error while copying data!"; exit 1; }
cd $SCRATCHDIR/ANLP

# Run script
python3 main.py &> log.txt
cp log.txt $DATADIR/ANLP || { echo >&2 "Error while copying log file!"; exit 3; }

# Clean 
conda deactivate
clean_scratch
