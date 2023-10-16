#!/bin/bash
#SBATCH --job-name=04c_asset_within               # Job name
#SBATCH -t 01:00:00                   # estimated time
#SBATCH -p gpu                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G v100:1              # Add the type of GPU used
#SBATCH -C scratch                    # ensure that I work on a node that has access to scratch
#SBATCH --mail-type=all               # send mail when job begins and ends
#SBATCH --mail-user=nschmid5@uni-goettingen.de  # email address
#SBATCH --output=./results/slurm_files/slurm_04c_within_asset.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./results/slurm_files/slurm_04c_within_asset.err      # where to write slurm error

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
object_name='within_asset'
target_var='mean_asset_index_yeh'
ls_cv_pth='results/model_objects/within_asset_MS_cv.pkl'
rs_cv_pth='results/model_objects/within_asset_RS_cv.pkl'

python -u dl_02c_within_train_combine.py "$object_name" "$target_var" "$ls_cv_pth" "$rs_cv_pth"

# add some description at the end to show that training is completed
echo " "
echo "============================================================================================"
echo "Training completed"
echo "============================================================================================"
echo " "
