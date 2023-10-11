#!/bin/bash
#SBATCH --job-name=01b_cons_RS_between              # Job name
#SBATCH -t 20:00:00                   # estimated time
#SBATCH -p gpu                     # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G v100:1              # Add the type of GPU used
#SBATCH -C scratch                    # ensure that I work on a node that has access to scratch
#SBATCH --mail-type=all               # send mail when job begins and ends
#SBATCH --mail-user=nschmid5@uni-goettingen.de # email address
#SBATCH --output=./results/slurm_files/slurm_01b_between_cons_RS.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./results/slurm_files/slurm_01b_between_cons_RS.err      # where to write slurm error

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
model_name='between_cons_RS'
cv_object_name='between_cons_RS_cv'
target_var='avg_log_mean_pc_cons_usd_2017'

data_type='RS_v2'
id_var='cluster_id'
img_folder='RS_v2_between'
stats_file='RS_v2_between_img_stats.pkl'

resnet_params='{"input_channels": 5, "use_pretrained_weights":false, "scaled_weight_init":false}'

python -u dl_01_between_train.py "$model_name" "$cv_object_name" "$target_var" "$data_type" "$id_var" "$img_folder" "$stats_file" "$resnet_params"

# add some description at the end to show that training is completed
echo " "
echo "============================================================================================"
echo "Training completed"
echo "============================================================================================"
echo " "