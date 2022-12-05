#!/bin/bash

pip3 install -r requirements.txt

python3 LStoM/process_data.py -i /Users/Katerina/Desktop/projects/MIDI_to_leadsheet_parser/POP909-Dataset/POP909 -o ./processed_data -q $@
python3 LStoM/create_stats_dict.py -i ./processed_data -o ./yaml_files $@
python3 LStoM/train.py -sd ./yaml_files/stats_config_train_valid.yaml -ds ./yaml_files/data_split.yaml -o ./model_results -of ./model_results/test_model.pt -v $@
