# Author : Debanjali Biswas

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Sequence Ordering of Alignments
"""

import os
import numpy as np
import pandas as pd

from constants import (
    folder,
    alignment_file,
    recipe_folder_name,
)
from utils import (
    fetch_parsed_recipe,
    fetch_action_ids,
)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class SequenceModel:

    def test_sequence_model(self):
        
        dish_list = os.listdir(folder)
    
        dish_list = [dish for dish in dish_list if not dish.startswith(".")]
        dish_list.sort()
        
        correct_predictions = 0
        num_actions =  0
        
        for dish in dish_list:
        
            data_folder = os.path.join(folder, dish)  # dish folder
            recipe_folder = os.path.join(data_folder, recipe_folder_name)  # recipe folder
                
        
            alignment_file_path = os.path.join(
                    data_folder, alignment_file
                )  # alignment file
        
            # Gold Standard Alignments between all recipes for dish
            alignments = pd.read_csv(
                    alignment_file_path, sep="\t", header=0, skiprows=0, encoding="utf-8"
                )
                    
            # Group by Recipe pairs
            group_alignments = alignments.groupby(["file1", "file2"])
            dish_correct_predictions = 0
            dish_num_actions = 0
             
            for key in group_alignments.groups.keys():
                
                #print('Recipe Pair: ')
                #print(key)
                    
                recipe1_filename = os.path.join(recipe_folder, key[0] + ".conllu")
                recipe2_filename = os.path.join(recipe_folder, key[1] + ".conllu")
                
                parsed_recipe1 = fetch_parsed_recipe(recipe1_filename)
                parsed_recipe2 = fetch_parsed_recipe(recipe2_filename)
                
                action_ids1 = fetch_action_ids(parsed_recipe1)
                #print('Actions in Recipe 1: ')
                #print(action_ids1)
                
                action_ids2 = fetch_action_ids(parsed_recipe2)
                #print('Actions in Recipe 2: ')
                #print(action_ids2)
                            
                if len(action_ids1) < len(action_ids2):
                
                    predictions = action_ids2[:len(action_ids1)]
                
                else: 
                    
                    predictions = action_ids2
                    predictions.extend([0] * (len(action_ids1) - len(action_ids2)))
                
                predictions = np.array(predictions)
                #print('Predictions: ') 
                #print(predictions)
                
                recipe_pair_alignment = group_alignments.get_group(key)
                
                true_labels = list()
                
                for i in action_ids1:
                    
                    # True Action Id
                    action_line = recipe_pair_alignment.loc[
                        recipe_pair_alignment["token1"] == i
                    ]
                    
                    if not action_line.empty:
                        
                        label = action_line["token2"].item()
                        true_labels.append(label)
                        
                    else:
                        true_labels.append(0)
                    
                true_labels = np.array(true_labels)
                #print('True Labels:') 
                #print(true_labels)
                
                score = [predictions == true_labels]
                
                dish_correct_predictions += np.sum(score)
                dish_num_actions += len(action_ids1)
                
                
            dish_accuracy = dish_correct_predictions  * 100 / dish_num_actions
            correct_predictions += dish_correct_predictions
            num_actions += dish_num_actions
            
            print("Accuracy of Dish {} : {:.2f}".format(dish, dish_accuracy))
            
        model_accuracy = correct_predictions * 100 / num_actions
    
        print("Model Accuracy: {:.2f}".format(model_accuracy))
                    
            
