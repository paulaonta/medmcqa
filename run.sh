#!/bin/bash

# Define the train and validation datasets
#train=( '4_ans_only_train_MIR_rm.csv' '4_5_ans_train_MIR_rm.csv')
#val=( '4_ans_only_val_MIR_rm.csv' '4_5_ans_val_MIR_rm.csv')

# Define the test datasets
#test=( '4_5_ans_test_rm_context.csv') #4_ans_only_test_MIR_rm_context.csv') # '4_5_ans_test_rm_context.csv')

# Define the random seeds
seed=(42 60 50 46 55)

# Train the models with both test datasets
#for t in "${test[@]}"
#do
#    for i in "${!train[@]}"
#    do
#        for s in "${seed[@]}"
#        do
#            python3 train.py --model bert-base-uncased --seed "$s" --train "${train[$i]}" --val "${val[$i]}" --test "$t" --use_context
#            python3 train2.py --model bert-base-uncased --seed "$s" --train "${train[$i]}" --val "${val[$i]}" --test "$t" --use_context
#        done
#    done
#done

for s in "${seed[@]}"
do            
    python3 train.py --model bert-base-uncased --seed "$s" --train train_MIR_rm.csv --val val_MIR_rm.csv --test test_KB_31kasu.csv

done

