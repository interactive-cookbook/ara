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
Utility functions
"""

# importing libraries
import os
import re
import torch
import pickle
import matplotlib.pyplot as plt

from conllu import parse
from matplotlib import style
from constants import prediction_file


style.use("ggplot")


def generate_recipe_dict(recipe, action_list, device):
    """
    Generate List of Recipe Dictionary

    Parameters
    ----------
    recipe : List
        Conllu parsed file for recipe.
    action_list : List
        List of action ids in recipe.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    recipe_node_list : List of Dict
        List of Dictionary for every action in the recipe.
        Dictionary contains Action_id, Action, Parent_List, Child_List

    """

    # No Alignment case
    empty_dict = {
        "Action_id": 0,
        "Action": torch.empty(0),
        "Parent_List": [],
        "Child_List": [],
    }

    recipe_node_list = [empty_dict]

    for action_id in action_list:

        action_node, parent_list, child_list = generate_data_line(
            action_id, recipe, device
        )
        # Recipe Dictionary
        recipe_dict = {
            "Action_id": action_id,
            "Action": action_node,
            "Parent_List": parent_list,
            "Child_List": child_list,
        }

        # Append Dictionary to List
        recipe_node_list.append(recipe_dict)

    return recipe_node_list


#####################################


def generate_elmo_embeddings(emb_model, tokenizer, recipe, device):
    """
    Generate Elmo Embeddings

    Parameters
    ----------
    emb_model : ElmoEmbedding object
        Elmo Embedding model from Flair.
    tokenizer : flair.data.Sentence object
        Flair Data Sentence.
    recipe : List
        Conllu parsed file for recipe.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    embedding_vectors : Dict
        Embedding dictionary for a particular Recipe;
            where keys are vector_lookup_list token_ids and values are their corresponding Elmo embeddings.
    vector_lookup_list : Dict
        Look up dictionary for a particular Recipe embeddings;
            where key is the Conllu file token 'id' and values are list of token_ids generated using Elmo.

    """
    
    
    recipe_text = ""
    recipe_text_list = []

    for line in recipe[0]:
        recipe_text += line["form"] + " "
        recipe_text_list.append(line["form"])

    recipe_text = recipe_text.rstrip()

    recipe_tokens = tokenizer(recipe_text)  # Flair Sentence Representation
    
    emb_model.embed(recipe_tokens) # Elmo Embeddings

    embedding_vector = {}

    for i, token in enumerate(recipe_tokens):

        embedding_vector[i] = token.embedding
    
    #print(embedding_vector)

    vector_lookup_list = {}
    
    for i, token in enumerate(recipe_tokens):

        vector_lookup_list[i + 1] = [i]
    
    #print(vector_lookup_list)

    return embedding_vector, vector_lookup_list
    

#####################################


def generate_bert_embeddings(emb_model, tokenizer, recipe, device):
    """
    Generate Bert Embeddings for a recipe

    Parameters
    ----------
    emb_model : BertModel object
            Bert Embedding Model from HuggingFace.
    tokenizer : BertTokenizer object
            Tokenizer.
    recipe : List
        Conllu parsed file for recipe.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    embedding_vectors : Dict
        Embedding dictionary for a particular Recipe;
            where keys are vector_lookup_list token_ids and values are their corresponding BERT embeddings.
    vector_lookup_list : Dict
        Look up dictionary for a particular Recipe embeddings;
            where key is the Conllu file token 'id' and values are list of token_ids generated using BERT.

    """

    # TODO this is a more pythonic way for the code below
    # until recipe_tokens = ...
    # recipe_text_list = [line['form'] for line in recipe[0]]
    # recipe_text = " ".join(recipe_text_list)

    recipe_text = ""
    recipe_text_list = []

    for line in recipe[0]:
        recipe_text += line["form"] + " "
        recipe_text_list.append(line["form"])

    recipe_text = recipe_text.rstrip()

    recipe_tokens = tokenizer.encode(recipe_text)  # Tokenize Recipe
    recipe_tensor = torch.LongTensor([recipe_tokens]).to(device)

    emb_model.eval()

    with torch.no_grad():

        outputs = emb_model(recipe_tensor)

        hidden = outputs[2]

    token_embeddings = torch.stack(hidden, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)

    # Stores the token vectors
    token_vecs_sum = []

    # For each token in the recipe...
    for token in token_embeddings:

        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-2:], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)

    embedding_vector = {}

    for i, emb in enumerate(token_vecs_sum):

        embedding_vector[i] = emb

    vector_lookup_list = {}
    count = 1

    for i, token in enumerate(recipe_text_list):

        tokenize = tokenizer.tokenize(token)

        vector_lookup_list[i + 1] = list(range(count, count + len(tokenize)))

        count += len(tokenize)

    return embedding_vector, vector_lookup_list


#####################################


def fetch_parsed_recipe(recipe_filename):
    """
    Fetch conllu parsed file

    Parameters
    ----------
    recipe_filename : String
        Recipe Filename.

    Returns
    -------
    parsed_recipe : List
       Conllu parsed file to generate a list of sentences.

    """
    
    file = open(recipe_filename, "r", encoding="utf-8")  # Recipe file
    conllu_file = file.read()  # Reading recipe file

    parsed_recipe = parse(conllu_file)  # Parsed Recipe File
    
    return parsed_recipe


#####################################


def fetch_recipe(recipe_filename, emb_model, tokenizer, device, embedding_name):
    """
    Fetch List of recipe dictionary and Embedding vector dictionary

    Parameters
    ----------
    recipe_filename : String
        Recipe Filename.
    emb_model : Embedding object
        Model.
    tokenizer : Tokenizer object
        Tokenizer.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    embedding_vectors : Dict
        Embedding dictionary for a particular Recipe;
            where keys are vector_lookup_list token_ids and values are their corresponding word embeddings (BERT/ELMO).
    vector_lookup_list : Dict
        Look up dictionary for a particular Recipe embeddings;
            where key is the Conllu file token 'id' and values are list of token_ids generated using BERT/ELMO.
    recipe_dict_list : List of Dict
        List of Dictionary for every action in the recipe.
        Dictionary contains Action_id, Action, Parent_List, Child_List

    """

    parsed_recipe = fetch_parsed_recipe(recipe_filename) # Parsed Recipe File
    
    if(embedding_name == 'bert'):

        embedding_vector, vector_lookup_list = generate_bert_embeddings(
            emb_model, tokenizer, parsed_recipe, device
        )  # Embeddings for Recipe
    
    elif (embedding_name == 'elmo'):
        
        embedding_vector, vector_lookup_list = generate_elmo_embeddings(
            emb_model, tokenizer, parsed_recipe, device
        )  # Embeddings for Recipe

    action_list = fetch_action_ids(parsed_recipe)  # List of actions in recipe

    recipe_dict_list = generate_recipe_dict(
        parsed_recipe, action_list, device
    )  # List of Recipe dictionary

    return embedding_vector, vector_lookup_list, recipe_dict_list


#####################################


def fetch_split_action(action_token_id, parsed_recipe):
    """
    Fetch tagging actions for a particular split action node

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.

    Returns
    -------
    tagging_tokens : string
        Tagging action sequence for a particular action node.

    """

    tagging_tokens = []
    token_id = action_token_id

    if token_id >= len(parsed_recipe[0]):
        return tagging_tokens

    line = parsed_recipe[0][token_id]

    while line["xpos"] == "I-A":  # Checking for intermediate action node

        tagging_tokens.append(line["id"])
        token_id += 1
        line = parsed_recipe[0][token_id]

    return tagging_tokens


#####################################


def fetch_action_node(action_token_id, parsed_recipe, device):
    """
    Fetch Action String for a particular action node

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    action : Tensor
        Action sequence for a particular action node.

    """

    action = [parsed_recipe[0][action_token_id - 1]["id"]]

    tagged_action = fetch_split_action(action_token_id, parsed_recipe)

    if tagged_action:
        action.extend(tagged_action)

    # action_tokens = tokenizer.encode(action) # Tokenize action node

    action_tokens = torch.LongTensor(action).to(device)

    return action_tokens


#####################################


def fetch_parent_node(action_token_id, parsed_recipe, device):
    """
    Fetch List of Children for a particular action

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    parent_list : List of Tensors
        List of parents for that particular action.

    """

    action_line = parsed_recipe[0][action_token_id - 1]

    parent_list = list()

    parent_id = action_line["head"]

    if parent_id != 0:

        parent = [parsed_recipe[0][parent_id - 1]["id"]]

        tagged_action = fetch_split_action(parent_id, parsed_recipe)

        if tagged_action:
            parent.extend(tagged_action)

        # parent_token = tokenizer.encode(parent) # Tokenize parent node

        parent_token = torch.LongTensor(parent).to(device)

        parent_list.append(parent_token)  # Append to parent_list

        other_parents = action_line["deps"]  # Other parents not belonging to head

        if other_parents:

            other_parents = re.split("[( )]", other_parents)

            unwanted = {"[", "]"}

            other_parents = [l for l in other_parents if l not in unwanted]

            for node in other_parents:

                parent_id = int(node.split(",")[0])
                parent = [parsed_recipe[0][parent_id - 1]["id"]]

                tagged_action = fetch_split_action(parent_id, parsed_recipe)

                if tagged_action:
                    parent.extend(tagged_action)

                # parent_token = tokenizer.encode(parent)

                parent_token = torch.LongTensor(parent).to(device)

                parent_list.append(parent_token)

    return parent_list


#####################################


def fetch_child_node(action_token_id, parsed_recipe, device):
    """
    Fetch List of Children for a particular action

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    child_list : List of Tensors
        List of children for that particular action.
    """

    child_list = list()

    for line in parsed_recipe[0]:

        if line["xpos"] == "B-A" and line["head"] == action_token_id:

            child_id = line["id"]
            child = [parsed_recipe[0][child_id - 1]["id"]]

            tagged_action = fetch_split_action(child_id, parsed_recipe)

            if tagged_action:
                child.extend(tagged_action)

            # child_token = tokenizer.encode(child) # Tokenize Child node

            child_token = torch.LongTensor(child).to(device)

            child_list.append(child_token)  # Append child to child_list

    return child_list


#####################################


def generate_data_line(action_token_id, parsed_recipe, device):
    """
    Generate action, parent list and child list for a particular action node

    Parameters
    ----------
    action_token_id : Int
        Action Token Id.
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    action : Tensor
        Action node.
    parent_list : List of Tensors
        List of parents for that particular action.
    child_list : List of Tensors
        List of children for that particular action.

    """

    action = fetch_action_node(
        action_token_id, parsed_recipe, device
    )  # Fetch Action node sequence

    parent_list = fetch_parent_node(
        action_token_id, parsed_recipe, device
    )  # Fetch List of parents for a particular action

    child_list = fetch_child_node(
        action_token_id, parsed_recipe, device
    )  # Fetch List of children for a particular action

    return action, parent_list, child_list


#####################################


def fetch_action_ids(parsed_recipe):
    """
    Fetch all action ids in a conllu parse file

    Parameters
    ----------
    parsed_recipe : List
        Conllu parsed file to generate a list of sentences.

    Returns
    -------
    indices : List
        List of all action ids in a conllu parse file.

    """

    indices = list()

    for line in parsed_recipe[0]:

        if line["xpos"] == "B-A":  # Checking for Action node
            indices.append(line["id"])

    return indices


#####################################


def save_vocabulary(path, vocab):
    """
    Save Naive Model Vocabulary

    Parameters
    ----------
    path : string
        path for saving the vocab.
    vocab : Dict
        Dictionary containing the action pairs are keys and their corresponding frequencies as values.

    Returns
    -------
    None.

    """
    
    with open(path, 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Model saved to ==> {path}")
    

#####################################


def load_vocabulary(path):
    """
    Load Saved Naive model Vocabulary

    Parameters
    ----------
    path : string
        path for saving the vocab.
        
    Returns
    -------
    vocab : Dict
        Saved Dictionary containing the action pairs are keys and their corresponding frequencies as values.

    """
    
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    
    return vocab
    

#####################################
    

def save_checkpoint(save_path, model, optimizer, valid_loss, valid_accuracy):
    """
    Function to save model checkpoint.

    Parameters
    ----------
    save_path : string
        path for saving the model checkpoints.
    model : object
        model.
    optimizer : object
        optimizer.
    valid_loss : float
        Validation loss.
    valid_accuracy : float
        validation accuracy.

    Returns
    -------
    None.

    """

    if save_path == None:
        return

    state_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
    }

    torch.save(state_dict, save_path)
    print(f"Model saved to ==> {save_path}")


#####################################


def load_checkpoint(load_path, model, optimizer, device):
    """
    Function to load model checkpoint.

    Parameters
    ----------
    load_path : string
        path for load the model checkpoints.
    model : object
        model.
    optimizer : object
        optimizer.
    device : object
        torch device where model tensors are saved.


    Returns
    -------
    model : AlignmentModel object
        Model after training
    optimizer : Adam optimizer object
        Optimizer after training.
    state_dict['valid_accuracy'] : Float
        Validation accuracy from the loaded state value dictionary.

    """

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f"Saved Model loaded from <== {load_path}")

    model.load_state_dict(
        state_dict["model_state_dict"],
        strict=False,
    )
    optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    return model, optimizer, state_dict["valid_accuracy"]


#####################################


def save_metrics(
    save_path,
    train_loss_list,
    valid_loss_list,
    train_accuracy_list,
    valid_accuracy_list,
    epoch_list,
):
    """
    Function to save model metrics.

    Parameters
    ----------
    save_path : string
        path for save the model metrics.
    train_loss_list : list
        list of training loss.
    valid_loss_list : list
        list of validation loss.
    train_accuracy_list : list
        List of traininbg accuracies
    valid_accuracy_list : list
        list of validation accuracies
    epoch_list : list
        list of epoch steps.

    Returns
    -------
    None.

    """

    if save_path == None:
        return

    state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "train_accuracy_list": train_accuracy_list,
        "valid_accuracy_list": valid_accuracy_list,
        "epoch_list": epoch_list,
    }

    torch.save(state_dict, save_path)
    print(f"Metrics saved to ==> {save_path}")


#####################################


def load_metrics(load_path, device):
    """
    Function to load model metrics.

    Parameters
    ----------
    load_path : string
        path for load the model metrics.
    device : object
        torch device where model tensors are saved.

    Returns
    -------
    dict
        metric dictionary containing training loss/accuracy and validation loss/accuracy at each epoch

    """

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f"Saved Metrics loaded from <== {load_path}")

    return state_dict


#####################################


def save_predictions(destination_folder, results_df, dish):
    """
    Save the predictions from the model during testing

    Parameters
    ----------
    destination_folder : string
        Destination folder name.
    results_df : Dataframe
        Results dataframe.
    dish : string
        dish name.

    Returns
    -------
    None.

    """

    save_prediction_path = os.path.join(
        destination_folder, dish + "_" + prediction_file
    )

    # Saving the results
    results_df.to_csv(save_prediction_path, sep="\t", index=False, encoding="utf-8")

    print("Predictions for Dish {} saved to ==> {}".format(dish, save_prediction_path))


#####################################


def create_acc_loss_graph(file_path, device, save_graph_path):
    """
    Generate Training/Validation Accuracy and Loss Graph

    Parameters
    ----------
    file_path : String
        path for load the model metrics.
    device : object
        torch device where model tensors are saved.
    save_graph_path : String
        Path for saving the graph.

    Returns
    -------
    None.

    """

    state_dict = load_metrics(file_path, device)
    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(
        state_dict["epoch_list"], state_dict["train_loss_list"], label="train_loss"
    )
    ax1.plot(
        state_dict["epoch_list"], state_dict["valid_loss_list"], label="valid_loss"
    )
    ax1.legend(loc=2)

    ax2.plot(
        state_dict["epoch_list"],
        state_dict["train_accuracy_list"],
        label="train_accuracy",
    )
    ax2.plot(
        state_dict["epoch_list"],
        state_dict["valid_accuracy_list"],
        label="valid_accuracy",
    )
    ax2.legend(loc=2)

    plt.show()

    ax1.autoscale()
    ax2.autoscale()
    fig.savefig(save_graph_path, dpi=fig.dpi)
    
    
'''dish_list = os.listdir(folder)
dish_list = [dish for dish in dish_list if not dish.startswith(".")]
dish_list.sort()
print(dish_list)

dish = dish_list[0]

transitive_property(folder, dish)'''