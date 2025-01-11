#! /bin/bash
#SBATCH -A rafaelgetto
#SBATCH --nodes=1       # Number of nodes
#SBATCH --gres=gpu:4    # Number of GPUs per node
#SBATCH --nodelist=gnode091     # Specifies the node or nodes by name where the job has to be executed
#SBATCH --ntasks=4              # Total number of tasks across all nodes
#SBATCH --ntasks-per-node=4     # Number of tasks per node
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --mem-per-cpu=5G        # specifies the amount of memory (RAM) allocated per CPU core
#SBATCH --time=5-00:00:00         # Time limit: days-hours:minutes:seconds
#SBATCH --mail-type=BEGIN,END,FAIL      # Mail events

# --exclude=<node_name>: Excludes specific nodes from being used by your job.
# --nodelist=<node_name>: Directly specifies the node or nodes by name where you want your job to run.
#  Example --nodelist=node001,node002  # Requests to run on nodes named node001 and node002
#  For augmentation-heavy tasks, start by allocating 2-4 CPU cores per task as a baseline.
#  If preprocessing becomes a bottleneck, scale up to 6-8 cores per task to further speed up data augmentation and loading.
#  Use top or htop to monitor CPU usage during training; adjust the number of cores if you observe CPU saturation or underutilization.


# conda initialization 
source /home2/rafaelgetto/miniconda3/etc/profile.d/conda.sh; 

# activate conda environment 
conda activate cr_clip;
echo "conda environment activated";

echo "Creating ssd_scratch/cvit/rafaelgetto directory";
mkdir /ssd_scratch/cvit/rafaelgetto;

echo "Sending all the dataset files to ssd_scratch";
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/str_cr_supervised_bctr_arl_train_lmdb /ssd_scratch/cvit/rafaelgetto;
rsync -aP rafaelgetto@ada.iiit.ac.in:/share3/rafaelgetto/str_cr_supervised_bctr_arl_val_lmdb /ssd_scratch/cvit/rafaelgetto;

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py;

echo "deactivate environment";
conda deactivate; 
