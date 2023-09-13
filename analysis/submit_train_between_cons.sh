#!/bin/bash
#SBATCH --job-name=cons_between               # Job name
#SBATCH -t 01:00:00                   # estimated time
#SBATCH -p medium                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -C scratch                    # ensure that I work on a node that has access to scratch
#SBATCH --mail-type=all               # send mail when job begins and ends
#SBATCH --mail-user=nschmid5@uni-goettingen.de  # email address
#SBATCH --output=./results/slurm_files/slurm_cons_between.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./results/slurm_files/slurm_cons_between.err      # where to write slurm error

module load anaconda3
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

# Add some space between technical information and the actual output.
echo " "
echo " "
echo "============================================================================================"
echo "Training Output"
echo "============================================================================================"
echo " "

# Run the script:
# set the variable names for the script
between_object_name='between_cons'
between_target_var='avg_log_mean_pc_cons_usd_2017'
ls_cv_pth='results/model_objects/between_cons_LS_cv.pkl'
rs_cv_pth='results/model_objects/between_cons_RS_cv.pkl'

python -u 03_between_train_combine.py "$between_object_name" "$between_target_var" "$ls_cv_pth" "$rs_cv_pth"

# add some description at the end to show that training is completed
echo " "
echo "============================================================================================"
echo "Training completed"
echo "============================================================================================"
echo " "
