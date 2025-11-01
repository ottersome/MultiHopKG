#!/bin/zsh
# Run this with source. i.e. source ./ottrs_env_activate.sh

eval "$(pyenv init - zsh)"
# Ensure you run pyenv local x.xx at somepoint here
eval "$(poetry env activate)"
