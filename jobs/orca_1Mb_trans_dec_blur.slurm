#!/bin/bash
#SBATCH --job-name=trans_dec     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30G        # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:2 		     # of GPUs
#SBATCH --time=3-00:00:00        # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=louisv@princeton.edu

module purge
module load anaconda3/2024.2
conda activate orca_env

python -W ignore -X faulthandler train_h1esc_a_TransDecBlur.py --swa
