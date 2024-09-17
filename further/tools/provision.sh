cp -a assets/dotfiles/. ~/

bash install_conda.sh

. ~/.bash_profile

# Install pretraining.
conda create -n default python=3.10 -y
conda activate default
bash install_pretraining_deps.sh
