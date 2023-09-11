#!/bin/bash
#SBATCH --job-name=test  # Job name
#SBATCH -t 00:10:00                  # estimated time
#SBATCH -p -gpu             # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                    # take 1 GPU, see https://www.hlrn.de/doc/display/PUB/GPU+Usage for more options
#SBATCH -C scratch                   # ensure that I work on a node that has access to scratch
#SBATCH --nodes=1                    # total number of nodes
#SBATCH --ntasks=1                   # total number of tasks
#SBATCH --cpus-per-task=4            # number cores per task
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=n.schmidtott@stud.uni-goettingen.de  # email address
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

module load anaconda3
module load cuda
source activate dl_env

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Run the script:
python -u train.py
