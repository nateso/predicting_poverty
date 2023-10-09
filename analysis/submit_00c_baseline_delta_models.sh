#!/bin/bash
#SBATCH --job-name=00c_baseline_delta              # Job name
#SBATCH -t 10:00:00                   # estimated time
#SBATCH -p medium                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -c 8
#SBATCH -C scratch                    # ensure that I work on a node that has access to scratch
#SBATCH --mail-type=all               # send mail when job begins and ends
#SBATCH --mail-user=nschmid5@uni-goettingen.de # email address
#SBATCH --output=./results/slurm_files/slurm_00c_delta.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./results/slurm_files/slurm_00c_delta.err      # where to write slurm error

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

echo " "
echo " "
echo "............................................................................................"
echo "Training Consumption Expenditure Model DELTA BASELINE"
echo "............................................................................................"
echo " "

between_target_var='avg_log_mean_pc_cons_usd_2017'
within_target_var='log_mean_pc_cons_usd_2017'
use_ls_vars='False'
file_pth='results/baseline/rep_cv_res_cons_delta.pkl'

python -u 01b_train_base_delta.py "$between_target_var" "$within_target_var" "$use_ls_vars" "$file_pth"

echo " "
echo " "
echo "............................................................................................"
echo "Training Consumption Expenditure Model DELTA BASELINE + LS"
echo "............................................................................................"
echo " "

use_ls_vars='True'
file_pth='results/baseline_ls/rep_cv_res_cons_delta.pkl'

python -u 01b_train_base_delta.py "$between_target_var" "$within_target_var" "$use_ls_vars" "$file_pth"



echo " "
echo " "
echo "............................................................................................"
echo "Training Asset Index Model DELTA BASELINE"
echo "............................................................................................"
echo " "
between_target_var='avg_mean_asset_index_yeh'
within_target_var='mean_asset_index_yeh'
use_ls_vars='False'
file_pth='results/baseline/rep_cv_res_asset_delta.pkl'

python -u 01_train_base.py "$between_target_var" "$within_target_var" "$use_ls_vars" "$file_pth"


echo " "
echo " "
echo "............................................................................................"
echo "Training Asset Index Model DELTA BASELINE + LS"
echo "............................................................................................"
echo " "
use_ls_vars='True'
file_pth='results/baseline_ls/rep_cv_res_asset_delta.pkl'

python -u 01_train_base.py "$between_target_var" "$within_target_var" "$use_ls_vars" "$file_pth"


# add some description at the end to show that training is completed
echo " "
echo "============================================================================================"
echo "Training completed"
echo "============================================================================================"
echo " "
