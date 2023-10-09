#!/bin/bash
#SBATCH --job-name=00a_baseline              # Job name
#SBATCH -t 10:00:00                   # estimated time
#SBATCH -p medium                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -c 8
#SBATCH -C scratch                    # ensure that I work on a node that has access to scratch
#SBATCH --mail-type=all               # send mail when job begins and ends
#SBATCH --mail-user=nschmid5@uni-goettingen.de # email address
#SBATCH --output=./results/slurm_files/slurm_00a_baseline.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./results/slurm_files/slurm_00a_baseline.err      # where to write slurm error

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
echo "Training Consumption Expenditure Model baseline"
echo "............................................................................................"
echo " "

between_target_var='avg_log_mean_pc_cons_usd_2017'
within_target_var='log_mean_pc_cons_usd_2017'
use_ls_vars='False'
remove_eth='False'
file_pth='results/baseline/rep_cv_res_cons.pkl'

python -u 01_train_base.py "$between_target_var" "$within_target_var" "$use_ls_vars" "$remove_eth" "$file_pth"


echo " "
echo " "
echo "............................................................................................"
echo "Training Consumption Expenditure Model baseline - No Ethiopia"
echo "............................................................................................"
echo " "

remove_eth='True'
file_pth='results/baseline/rep_cv_res_cons_no_eth.pkl'

python -u 01_train_base.py "$between_target_var" "$within_target_var" "$use_ls_vars" "$remove_eth" "$file_pth"


echo " "
echo " "
echo "............................................................................................"
echo "Training Asset Index Model baseline"
echo "............................................................................................"
echo " "
between_target_var='avg_mean_asset_index_yeh'
within_target_var='mean_asset_index_yeh'
use_ls_vars='False'
remove_eth='True'
file_pth='results/baseline/rep_cv_res_asset.pkl'

python -u 01_train_base.py "$between_target_var" "$within_target_var" "$use_ls_vars" "$remove_eth" "$file_pth"

echo " "
echo " "
echo "............................................................................................"
echo "Training Asset Index Model baseline - no ETHIOPIA"
echo "............................................................................................"
echo " "
between_target_var='avg_mean_asset_index_yeh_no_eth'
within_target_var='mean_asset_index_yeh_no_eth'
use_ls_vars='False'
remove_eth='True'
file_pth='results/baseline/rep_cv_res_asset_no_eth.pkl'

python -u 01_train_base.py "$between_target_var" "$within_target_var" "$use_ls_vars" "$remove_eth" "$file_pth"


# add some description at the end to show that training is completed
echo " "
echo "============================================================================================"
echo "Training completed"
echo "============================================================================================"
echo " "
