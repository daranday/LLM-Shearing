# Check if 'conda' command is available
if command -v conda &> /dev/null; then
    echo "Conda is available. Exiting script."
    exit 0
fi

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -P /tmp
bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init