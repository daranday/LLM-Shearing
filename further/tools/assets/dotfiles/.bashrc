export PATH=/run/determined/pythonuserbase/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/run/determined/workdir/.vscode-server/cli/servers/Stable-eaa41d57266683296de7d118f574d0c2652e1fc4/server/bin/remote-cli:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games
export DET_MASTER=10.182.1.43

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        . "/opt/conda/etc/profile.d/conda.sh"
    else
        export PATH="/opt/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate default

# some more ls aliases
alias ll='ls -alFhtr'
alias la='ls -A'
alias l='ls -CF'
alias gs='git status'
alias gv='git remote -v'
alias gb='git branch'
alias gl='git log'
alias gci='git commit'
alias gdiff='git --no-pager diff --name-only'
alias hs='hg status'
alias hl='hg log'
alias hsl='hg sl'
alias hc='hg checkout'
alias ha='hg amend'
alias hgblend='hg uncommit && hg amend'
alias grep='grep --color=auto'
alias br='bazel run'
alias bb='bazel build'
alias bt='bazel test'
alias st='streamlit'
function gitup { git commit -a -m "${1:-Update code}"; git pull --rebase && git push; }