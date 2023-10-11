# Robotic course, XXXX, 20YY

This repository contains the exercices for the robotics class at XXXX, 20YY.
The exercices are organized by notebook. Each notebook corresponds to one chapter of the class.
The notebooks are in Python and based on the software [Pinocchio](https://github.com/stack-of-tasks/pinocchio).

## Set up

### Clone this repository

Using Git via SSH:

```bash
git clone git@github.com:ymontmarin/XXXX_20YY_tps_robotic.git
```

Or via HTTPS:

```bash
git clone https://github.com/ymontmarin/XXXX_20YY_tps_robotic.git
```

### Install miniconda

- Linux: https://docs.conda.io/en/latest/miniconda.html
- macOS: https://docs.conda.io/en/latest/miniconda.html
- Windows: https://www.anaconda.com/download/

Only a little snippet is applied to your home .bashrc, everything else will be segmented!

- Go to your local copy of the repository.
- Open a terminal.
- Create and activate conda environment:

```bash
conda env create -f robotic_course_env.yml
```

From now on, when you want to work on the TPs you only need to do:
```bash
conda activate robotic_course
```
And launch the notebook with:
```bash
jupyter-lab
```
Don't forget, the notebook is accessible from any browser using `localhost:8888` in the adress bar.
Meshcat visualisation can be access in full page in `localhost:700N/static/` where N denotes the Nth meshcat instance created with the running kernel.


### Update the notebooks

If the repository changes (for example when new tutorials are pushes), you need to update your local
version by "pulling" it from the repository.
On a native installation, just go in the folder containing the tutorials and execute ```git pull```.


# TODO
- Update from running TPs
- Clear all output for pushed version
