#!/bin/sh

if [ "$2" == true ]; then
    pipenv run python td.py --noise_level $1 --no_negative
else
    pipenv run python td.py --noise_level $1
fi