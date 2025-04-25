#!/bin/bash
#SBATCH --job-name=APC523PS4     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)

#SBATCH --mem-per-cpu=24G        # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jk9075@princeton.edu
#SBATCH --error=job.err
#SBATCH --output=job.out

module purge
module load anaconda3/2024.6

conda activate
srun --ntasks=1 --exclusive --cpus-per-task=8 --mem-per-cpu=24G  python3 p1/main.py &
srun --ntasks=1 --exclusive --cpus-per-task=8 --mem-per-cpu=16G python3 p3/main.py &
wait