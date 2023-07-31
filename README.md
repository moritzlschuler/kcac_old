# PLEASE FIND THE MAINTAINED VERSION OF THIS REPO HERE: https://github.com/lingkaching/kcac

# psddAQL
Constrained Amortized Q-Learning using a PSDD

# Installation:
1. Clone this library
1. create a conda env with python 3.6. and optionally run `conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge` for GPU support
1. install libraries from `requirements.txt` into your python 3.6 environment
1. install gymGame, gym-ERSLE and gym-BSS from my other github repos.
1. move to the `psddAgent/utils/` folder
1. clone psdd from github (https://github.com/mschuler-smu/psdd)
3. replace CMakeLists.txt in psdd with CMakeLists.txt provided in `psddAgent/utils/move_into_psdd/`
5. from psdd folder run `cmake .`
6. run `make`
7. copy mult_sdd2psdd.cpp into psdd folder from `psddAgent/utils/move_into_psdd/`
9. run `g++ -no-pie -o mult_sdd2psdd mult_sdd2psdd.cpp src/psdd_manager.cpp src/psdd_node.cpp src/psdd_parameter.cpp src/psdd_unique_table.cpp src/random_double_generator.cpp -Iinclude -Llib/linux -lsdd -lgmp`
1. run scripts from `scripts/`
1. if you get the error `ModuleNotFoundError: No module named 'pysdd.sdd'` try reinstalling the PySDD package by running `pip install -vvv --upgrade --force-reinstall --no-binary :all: --no-deps pysdd`.
