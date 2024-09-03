# cp -a assets/dotfiles/. ~/

# sh install_conda.sh

# Source conda
. ~/.bash_profile

# Install necessary packages
conda install -c conda-forge \
    htop \
    tmux \
    gh \
    -y

# Install pip packages
pip install -r assets/requirements.txt