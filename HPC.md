##
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