#date: 2022-06-30T17:16:54Z
#url: https://api.github.com/gists/f65c1e5d0f6c9fafee463dffa71b77be
#owner: https://api.github.com/users/Alfred-N

#Allocation
salloc --gres=gpu:0 --mem=1GB --cpus-per-task=1 --constrain=khazadum --time=1:00:00

#Run
srun --gres=gpu:0 --mem=1GB --cpus-per-task=1 --time=6:00:00 \
     --constrain=khazadum --mail-user=[your_email]@kth.se\                        
     --mail-type=BEGIN,END,FAIL \
     --output %J.out --error %J.err \
     python train.py

#----------------------------------------------------------
#-----------------------sbatch script ---------------------
#!/usr/bin/env bash
#SBATCH --mem  5GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain "khazadum|rivendell|belegost|shire"
#SBATCH --mail-type FAIL
#SBATCH --mail-user [your_email]@kth.se
#SBATCH --output /Midgard/home/%u/my_project/logs/%J_slurm.out
#SBATCH --error  /Midgard/home/%u/my_project/logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
nvidia-smi
. ~/miniconda3/etc/profile.d/conda.sh
conda activate project_env
python train.py --cuda --learning-rate ${LEARNING_RATE} --epochs ${EPOCHS}
#----------------------------------------------------------

#submit batch job
sbatch --export LEARNING_RATE=.001,EPOCHS=10 example.sbatch

#Squeue: how many allocations do I have? who is in the queue right now?
squeue1

#Sinfo: what is the status of the cluster?
sinfo --Format=nodehost,cpus,memory,gres,statecompact

#Scontrol: show/edit current jobs or nodes
scontrol hold JOB_ID
scontrol requeue JOB_ID
scontrol show job JOB_ID
scontrol show node belegost

#Scancel: cancel your job(s)
scancel JOB_ID
scancel --user ${USER}
scancel --state PENDING

#Sacct: info about past jobs, did they fail? How long did they run? How much RAM did they use?
sacct -j JOB_ID
sacct --starttime yyyy-mm-dd -u [your_user] \
      --format=jobid,start,maxrss,elapsed,state%30,exitcode,avediskread,avediskwrite,allocgres,nodelist

#Sshare: what is my user priority1? who gets the allocation first?
sshare --all
