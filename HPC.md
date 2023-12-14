## resources
- https://wiki.tudelft.nl/bin/view/Research/InsyCluster/ConnectingSSH
- https://docs.google.com/presentation/d/10A0_0eNRBYd87E1h1YN6bsIFaZaua5qJkfBbnBKAr6o/present?slide=id.g13d50e386b_2_2 
- https://login.hpc.tudelft.nl/
- https://teams.microsoft.com/_#/apps/3e0a4fec-499b-4138-8e7c-71a9d88a62ed/sections/MyNotebook
- https://www.notion.so/HPC-start-980c577864c34f3a94591259df0520a7

### login to bastion
```bash
ssh runyuma@student-linux.tudelft.nl
```
```bash
ssh login.daic
```
or 
```bash
ssh login1.hpc.tudelft.nl
```
### cd to your home directory
```bash
 cd "/tudelft.net/staff-umbrella/rarma"
```
### conda 
```
module use /opt/insy/modulefiles
module load miniconda/3.9
```
### interactive session
```bash
sinteractive --cpus-per-task=2 --mem=5000mb ###Request for resource.
```
### submit job
```bash
sbatch train.sbatch
```

## Current jobid
- 9157701
- 9159811


## Other commands
### show all jobs
- squeue -u NetID
### show efficiency
- seff jobid

## copy to local
```bash
scp -p -r runyuma@student-linux.tudelft.nl:/tudelft.net/staff-umbrella/rarma/src/RA/tmp/final_tb /home/marunyu
```
