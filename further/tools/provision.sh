cp assets/.* ~/

sh install_conda.sh

# Source conda
. ~/.profile

# Install necessary packages
conda install -c conda-forge \
    htop \
    tmux \
    -y