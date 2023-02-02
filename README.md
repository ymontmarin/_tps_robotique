# ENS robotics, 2022

This repository contains the exercices for the robotics class at ENS, 2022.
The exercices are organized by notebook. Each notebook corresponds to one chapter of the class.
The notebooks are in Python and based on the software [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Set up

### Clone this repository
ssh
```bash
git clone git@github.com:ymontmarin/tps_robotic_ens_2022.git
```

https
```bash
git clone https://github.com/ymontmarin/tps_robotic_ens_2022.git
```

### Install miniconda
On Linux or OSX only:
https://docs.conda.io/en/latest/miniconda.html


Go into the repository folder.

Create and activate conda environment

```bash
conda env create -f robotic_course_env.yml
conda activate robotic_course
```
Launch jupyter
```bash

jupyter-lab
```

### Update the notebooks

If the repository changes (for example when new tutorials are pushes), you need to update your local
version by "pulling" it from the repository.
On a native installation, just go in the folder containing the tutorials and execute ```git pull```
