cp -a assets/dotfiles/. ~/

bash install_conda.sh

conda init
. ~/.bash_profile

# Install necessary packages
conda install -c conda-forge \
    htop \
    tmux \
    gh \
    tree \
    -y


# Install pretraining.
conda create -n default python=3.10 -y
conda activate default
bash install_pretraining_deps.sh
