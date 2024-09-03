cp -a assets/dotfiles/. ~/

sh install_conda.sh

conda init
conda create -n default python=3.10 -y
conda activate default

# Source conda
. ~/.bash_profile

# Install necessary packages
conda install -c conda-forge \
    htop \
    tmux \
    gh \
    -y

# Install pip packages
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn==1.0.3.post
pip install -e ../..