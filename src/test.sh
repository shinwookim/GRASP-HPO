#!/bin/sh

# assuming .venv is in working directoy and requirements.txt has been pip installed
alias activate=". .venv/bin/activate"
activate

for iter in 1 2 3 4 5
do

    python3 XGBoostGRASP_HPO.py > out/test$iter.txt
    echo "Iteration $iter complete"

done