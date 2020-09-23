#!/bin/sh]
git add *.py
git add *.sh
git add notebook
git commit -m "$*"
git push origin master