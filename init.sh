#!/bin/bash
set -e

read -rep "create pre-commit hook for updating python dependencies? (y/n): " ans
if [ $ans == "y" ]; then
  # move pre-commit hook into local .git folder for activation
  cp ./hooks/pre-commit.sample ./.git/hooks/pre-commit
fi

read -rep "download and deploy MIMIC-III best models? (y/n): " ans
if [ $ans == "y" ]; then
  wget https://github.com/YerevaNN/mimic3-benchmarks/releases/download/v1.0.0-alpha/pretrained-weights.zip -P ./keras_models
  cd ./keras_models
  unzip *zip
  cd ..
fi
