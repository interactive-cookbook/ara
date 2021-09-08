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
Simple Baseline Model: Common Action pair heuristics

"""


from collections import Counter
from utils import fetch_parsed_recipe, fetch_action_node, fetch_action_ids


class NaiveModel:
    
    def __init__(self, device):

        self.device = device
        
    def generate_action_text(self, action_id, parsed_recipe):
        """
        Generate Action words for a given action id

        Parameters
        ----------
        action_id : Int
            Action Id.
        parsed_recipe :  List
            Conllu parsed file to generate a list of sentences.

        Returns
        -------
        action_text : String
            Action Words.

        """
        
        action_tokens = fetch_action_node(action_id, parsed_recipe, self.device)
        
        action_text = ""
        
        for i in action_tokens:
            action_text += parsed_recipe[0][i - 1]["form"] + " "
            
        action_text = action_text.strip().lower()
        
        return action_text
    
    
    
    def convert_ids_to_text(self, pairs, parsed_recipe1, parsed_recipe2):
        """
        Convert ids to words

        Parameters
        ----------
        pairs : List
            List of pairs action ids.
        parsed_recipe1 : List
            Conllu parsed file to generate a list of sentences for Recipe1.
        parsed_recipe2 : List
            Conllu parsed file to generate a list of sentences for Recipe2.

        Returns
        -------
        action_pairs : List
            List of action pairs in text.

        """
        
        action_pairs = list()
        
        for pair in pairs:
            
            action1 = pair[0]
            action2 = pair[1]
            
            action_text1 = self.generate_action_text(action1, parsed_recipe1)
            
            if action2 == 0:
                action_text2 = 'None'
                
            else:
                action_text2 = self.generate_action_text(action2, parsed_recipe2)
                
            action_pair_tuple = (action_text1,action_text2)
                
            action_pairs.append(action_pair_tuple)
        
        return action_pairs
            
    
    
    def generate_action_pairs(self, recipe_pair_alignment, recipe1_filename, recipe2_filename):
        """
        Generate list of action pairs between 2 recipes

        Parameters
        ----------
        recipe_pair_alignment : Pandas GroupBy object
            Alignment pairs for a particular recipe pair
        recipe1_filename : String
            Recipe Filename 1.
        recipe2_filename : String
            Recipe Filename 2.

        Returns
        -------
        action_pairs : List
            List of aligned action pairs.

        """
        
        parsed_recipe1 = fetch_parsed_recipe(recipe1_filename) # Parsed recipe file for recipe 1
        parsed_recipe2 = fetch_parsed_recipe(recipe2_filename) # Parsed recipe file for recipe 2
        
        action_ids1 = recipe_pair_alignment['token1'] # List of action ids for recipe 1
        action_ids2 = recipe_pair_alignment['token2'] # List of action ids for recipe 2

        pairs = list(zip(action_ids1,action_ids2)) # list of aligned action pair ids

        action_pairs = self.convert_ids_to_text(pairs, parsed_recipe1, parsed_recipe2)
        
        return parsed_recipe1, parsed_recipe2, action_pairs
    
    
    
    def generate_vocabulary(self, action_pair_list):
        """
        Generate Action Pair vocabulary

        Parameters
        ----------
        action_pair_list : List
            List of aligned action pairs for all dishes and recipes.

        Returns
        -------
        action_pair_vocabulary : Dict
            Dictionary containing the action pairs are keys and their corresponding frequencies as values.

        """
        
        action_pair_vocabulary = Counter(action_pair_list)
       
        return action_pair_vocabulary
    
    
    def fetch_all_actions(self, parsed_recipe):
        
        action_ids = fetch_action_ids(parsed_recipe)
        
        action_list = list()
        
        for i in action_ids:
           
            action_list.append(self.generate_action_text(i, parsed_recipe))
        
        return action_list
    
    
    
    def filter_vocab(self, action, vocab):
        
        
        filteredVocab = Counter()
        
        for k, v in vocab.items():
            
            if action in k:
                filteredVocab[k] = v
                
        return filteredVocab
        

    
    def fetch_aligned_actions(self, action_pair_list, 
                              action_pair_vocabulary, 
                              parsed_recipe2, 
                              correct_predictions,
                              num_actions, 
                              results_df):
        
        for pair in action_pair_list:
            
            num_actions += 1
                        
            action = pair[0]
            true_label = pair[1]
            
            #print(action)
            
            possible_actions = self.fetch_all_actions(parsed_recipe2)
            possible_actions.append('None')
            
            filteredVocab = self.filter_vocab(action, action_pair_vocabulary)
            #print(filteredVocab)
            
            check = True
            
            if filteredVocab :
                
                count = 1
                
                while(count <= len(filteredVocab.keys())):
                    
                    predicted_label = filteredVocab.most_common(count)[count - 1][0][1]
                    
                    count += 1
                    
                    if( predicted_label in possible_actions):
                        
                        check = False
                        break
                
            if(check):
                
                predicted_label = "None"
                
            if predicted_label == true_label:
                correct_predictions += 1
                
            
            result_dict = {
                "Action": action,
                "True_Label": true_label,
                "Predicted_Label": predicted_label
                }
            
            results_df = results_df.append(result_dict, ignore_index=True)
        
        #correct_predictions += predictions
            
        return correct_predictions, num_actions, results_df
            
                
            
            
                    
                    
            
            
            
        
        
            
        
