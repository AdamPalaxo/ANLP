#!/bin/bash
#PBS -N ner_cased
#PBS -l select=1:ncpus=1:ngpus=1:gpu_cap=cuda75:mem=24gb:scratch_local=20gb:cluster=adan
#PBS -l walltime=12:00:00
#PBS -q gpu
#PBS -m abe

DATADIR=/storage/plzen1/home/amistera
LOGFILE=log.txt

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

# Load conda
module add conda-modules-py37
conda activate $DATADIR/conda

# Copy Data
cp -r $DATADIR/ANLP $SCRATCHDIR || { echo >&2 "Error while copying files!"; exit 1; }
cd $SCRATCHDIR/ANLP

# Run script
python3 main.py &> $LOGFILE
cp $LOGFILE $DATADIR/ANLP || { echo >&2 "Error while copying log file!"; exit 3; }

# Saves models
cp -r Model/ $DATADIR/ANLP || { echo >&2 "Error while copying model file!"; exit 3; }

# Clean 
conda deactivate
clean_scratch
