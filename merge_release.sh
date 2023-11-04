#!/bin/bash

git pull
git fetch release
git merge release/main -m "Merging release repository"
