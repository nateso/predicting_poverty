#!/bin/bash
#SBATCH --job-name=test  # Job name
#SBATCH -t 00:10:00                  # estimated time
#SBATCH -p medium             # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -C scratch                   # ensure that I work on a node that has access to scratch
#SBATCH --mail-type=all              # send mail when job begins and ends
#SBATCH --mail-user=n.schmidtott@stud.uni-goettingen.de  # email address
#SBATCH --output=./slurm_files/slurm-%x.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x.err      # where to write slurm error

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
python -u 99_test_file.py
