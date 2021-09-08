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
Testing for Reverse and Transitivity property

"""
# -*- coding: utf-8 -*-

# importing libraries
import os
import sys
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from model import AlignmentModel
from transformers import BertTokenizer, BertModel
from utils import fetch_recipe, load_checkpoint, save_predictions
from constants import recipe_folder_name, folder, alignment_file, destination_folder1, OUTPUT_DIM, LR, FOLDS, HIDDEN_DIM1, HIDDEN_DIM2, EPOCHS
from training_testing import Folds

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


#####################################

class Property:


    def transitive_property(self, alignment_file, dish):
    
        data_folder = os.path.join(folder, dish)  # dish folder
        recipe_folder = os.path.join(data_folder, recipe_folder_name)  # recipe folder
        recipe_list = os.listdir(recipe_folder)

        recipe_list = [recipe for recipe in recipe_list if not recipe.startswith(".")]
        
        alignment_file_path = os.path.join(
            data_folder, alignment_file
            )  # alignment file

        # Gold Standard Alignments between all recipes for dish
        alignments = pd.read_csv(
            alignment_file_path, sep="\t", header=0, skiprows=0, encoding="utf-8"
            )
        
        # Group by Recipe pairs
        group_alignments = alignments.groupby(["file1", "file2"])
        keys_list = list(group_alignments.groups.keys())
        
        closure = alignments
        
        while True:

            merge = closure.merge(closure, left_on = ['file2', 'token2'], right_on = ['file1', 'token1'])
            merge.drop(columns = ['file2_x', 'file1_y', 'token2_x', 'token1_y'], inplace = True)
            merge.rename(columns = {'file1_x': 'file1', 'token1_x': 'token1', 'file2_y': 'file2', 'token2_y': 'token2'}, inplace = True)
       
            new_closure = pd.concat([closure, merge])
            new_closure.drop_duplicates(inplace = True, ignore_index = True)
            
            if closure.equals(new_closure):
                break
        
            closure = new_closure
            
        #group_alignments = closure.groupby(["file1", "file2"])
        
        #for key in keys_list:
            
            #group_alignments = closure.groupby(["file1", "file2"])
            
            #closure.drop(group_alignments.get_group(key).index, inplace = True)
        
        closure.to_csv(os.path.join(data_folder, 'transitive_alignments.tsv'),sep = '\t', index = False, encoding = 'utf-8')
    

#####################################


    def reverse_property(self,
                         dish,
        emb_model,
        tokenizer,
        model,
        device,
        correct_predictions=0,
        num_actions=0,):

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

        results_df = pd.DataFrame(
                columns=["Action1_id", "True_Label", "Predicted_Label"]
            )

        for key in group_alignments.groups.keys():

            recipe1_filename = os.path.join(recipe_folder, key[0] + ".conllu")
            recipe2_filename = os.path.join(recipe_folder, key[1] + ".conllu")

            embedding_vector1, vector_lookup_list1, recipe_dict1 = fetch_recipe(
                recipe1_filename, emb_model, tokenizer, device, 'bert',
            )
            embedding_vector2, vector_lookup_list2, recipe_dict2 = fetch_recipe(
                recipe2_filename, emb_model, tokenizer, device, 'bert',
            )

            recipe_pair_alignment = group_alignments.get_group(key)

            for node in recipe_dict2[1:]:

                # True Action Id
                action_line = recipe_pair_alignment.loc[
                    recipe_pair_alignment["token2"] == node["Action_id"]
                ]

                if not action_line.empty:

                    true_label = list(action_line["token1"])
                
                else:
                    
                    true_label = [0]
                

                action2 = node["Action"]
                parent_list2 = node["Parent_List"]
                child_list2= node["Child_List"]

                # Generate predictions using our Alignment Model

                prediction = model(
                            action2.to(device),
                            parent_list2,
                            child_list2,
                            embedding_vector2,
                            vector_lookup_list2,
                            recipe_dict1,
                            embedding_vector1,
                            vector_lookup_list1,
                        )

                num_actions += 1

                # Predicted Action Id
                pred_label = recipe_dict1[torch.argmax(prediction).item()][
                        "Action_id"
                    ]

                if pred_label in true_label:
                        correct_predictions += 1

                results_dict = {
                            "Action1_id": node["Action_id"],
                            "True_Label": true_label,
                            "Predicted_Label": pred_label,
                        }

                        # Store the prediction
                results_df = results_df.append(results_dict, ignore_index=True)

        return correct_predictions, num_actions, results_df


#####################################


    def testing_reverse_property(self, num_folds, emb_model, tokenizer, model, optimizer, device):
    
        dish_list = os.listdir(folder)

        dish_list = [dish for dish in dish_list if not dish.startswith(".")]
        dish_list.sort()
        
        test_dish_id = len(dish_list) - 1
        
        total_correct_predictions = 0
        total_actions = 0
        
        test_dish_id = len(dish_list) - 1

        
        for fold in range(num_folds):
            
            saved_file_path = os.path.join(
                destination_folder1, "model" + str(fold + 1) + ".pt"
            )  # Model saved path
            
    
            model, optimizer, _ = load_checkpoint(saved_file_path, model, optimizer, device)
            
            dish = dish_list[test_dish_id]
            
            model.eval()
            
            with torch.no_grad():

                 (correct_predictions, num_actions, results_df) = self.reverse_property(dish,
                         emb_model,
                         tokenizer,
                         model,
                         device,
                         correct_predictions=0,
                         num_actions=0,)
                 
            dish_accuracy = correct_predictions * 100 / num_actions

            save_predictions(destination_folder1, results_df, dish)

            total_correct_predictions += correct_predictions
            total_actions += num_actions
            
            print("Accuracy of Dish {} : {:.2f}".format(dish_list[test_dish_id], dish_accuracy))
            
            test_dish_id -= 1

            if test_dish_id == -1:

                test_dish_id = len(dish_list) - 1

        model_accuracy = total_correct_predictions * 100 / total_actions

        print("Model Accuracy: {:.2f}".format(model_accuracy))

        return
    

#####################################


def main():
    
    property_name = sys.argv[1]
    
    p = Property()
    
    tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased"
            )  # Bert Tokenizer
        
    emb_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(
                device
            )  # Bert Model for Embeddings
            
    embedding_dim = emb_model.config.to_dict()[
                "hidden_size"
            ]  # BERT embedding dimension
        
    model = AlignmentModel(embedding_dim, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, device).to(
                device
            )  # Out Alignment Model with features
    
    print(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer for training
    criterion = nn.CrossEntropyLoss()  # Loss function
    
    if property_name == 'Reverse':
    
        print('Testing Reverse property....\n')
        
        p.testing_reverse_property(FOLDS, emb_model, tokenizer, model, optimizer, device)
        
    elif property_name == 'Transitivity':
        
        F = Folds()
        
        print('Generating transitive data...\n')
        
        dish_list = os.listdir(folder)

        dish_list = [dish for dish in dish_list if not dish.startswith(".")]
        dish_list.sort()
        
        alignment_file ='alignments.tsv'
        
        for dish in dish_list:
            p.transitive_property(alignment_file, dish)
            
        print('Cross Validation folds with Augmented transitive data....\n')
        
        print("-------Cross Validation Folds-------")

        F.run_folds(
            'bert',
            emb_model, 
            tokenizer,
            model,
            optimizer,
            criterion,
            EPOCHS,
            FOLDS,
            device,
            False,
        )
          
    else:
        
        print('Property name should be in [Reverse, Transitivity]')


#####################################



if __name__ == "__main__":
    
    main()
