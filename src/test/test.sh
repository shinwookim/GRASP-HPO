#!/bin/sh

# assuming .venv is in working directoy and requirements.txt has been pip installed
#alias activate=". ../.venv/bin/activate"
#activate

for iter in 0 1 2 3
do

    python3 main/XGBoostGRASP_HPO.py $iter >> test/out/test.txt
    echo "Iteration $iter complete"

done