#!/bin/bash
#SBATCH --job-name=between_cons_LS               # Job name
#SBATCH -t 20:00:00                   # estimated time
#SBATCH -p gpu                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G v100:1              # Add the type of GPU used
#SBATCH -C scratch                    # ensure that I work on a node that has access to scratch
#SBATCH --mail-type=all               # send mail when job begins and ends
#SBATCH --mail-user=nathanael.schmidt-ott@wiwi.uni-goettingen.de  # email address
#SBATCH --output=./slurm_files/slurm-%x.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x.err      # where to write slurm error

module load anaconda3
module load cuda
source activate dl_env # Or whatever you called your environment.

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
nvcc -V

# Add some space between technical information and the actual output.
echo " "
echo " "
echo "============================================================================================"
echo "Training Output"
echo "============================================================================================"
echo " "

# Run the script:
# set the variable names for the script
model_name='between_cons_LS'
cv_object_name='between_cons_LS_cv'
target_var='avg_log_mean_pc_cons_usd_2017'

python -u 01_between_train_LS.py "$model_name" "$cv_object_name" "$target_var"

# add some description at the end to show that training is completed
echo " "
echo "============================================================================================"
echo "Training completed"
echo "============================================================================================"
echo " "
# Run the script with logger:
#python -u train_with_logger.py -l ~/${SLURM_JOB_NAME}_${SLURM_JOB_ID}  -t True -p True -d True -s True -f True
