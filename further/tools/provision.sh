cp -a assets/. ~/

sh install_conda.sh

# Source conda
. ~/.bash_profile

# Install necessary packages
conda install -c conda-forge \
    htop \
    tmux \
    gh \
    -y
