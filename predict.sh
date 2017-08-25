#!/bin/bash

for i in `seq 1 16`; do
    
    python convnet_pred_script.py $i
    
done

