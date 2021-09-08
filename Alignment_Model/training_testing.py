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
Training and Testing Process functions
"""

# importing libraries


import os
import torch
import pandas as pd

from datetime import datetime
from constants import (
    folder,
    alignment_file,
    recipe_folder_name,
    destination_folder1,
    destination_folder2,
    destination_folder3,
    destination_folder4,
)
from utils import (
    fetch_recipe,
    save_metrics,
    save_checkpoint,
    load_checkpoint,
    save_predictions,
    create_acc_loss_graph,
    save_vocabulary,
    load_vocabulary
)


# Training and Testing Process Class
class Folds:
    def run_model(
        self,
        dish,
        emb_model,
        tokenizer,
        model,
        device,
        embedding_name,
        criterion=None,
        optimizer=None,
        total_loss=0.0,
        step=0,
        correct_predictions=0,
        num_actions=0,
        mode="Training",
        model_name="Alignment Model",
    ):
        """
        Function to run the Model

        Parameters
        ----------
        dish : String
            Dish.
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        device : object
            torch device where model tensors are saved.
        criterion : Cross Entropy Loss Function, optional
            Loss Function. The default is None.
        optimizer : Adam optimizer object, optional
            Optimizer. The default is None.
        total_loss : Float, optional
            Total Loss after Training/Validation. The default is 0.0.
        step : Int, optional
            Each Training/Validation step. The default is 0.
        correct_predictions : Int, optional
            Correction predictions for a Dish. Defaults is 0.
        num_actions : Int, optional
            Number of actions in a Dish. Defaults is 0.
        mode : String, optional
            Mode of Process - ("Training", "Validation", "Testing"). The default is "Training".
        model_name : String, optional
            Model name - ("Alignment Model", "Simple Model"). Default is "Alignment Model".


        """

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

        if mode == "Testing":
            results_df = pd.DataFrame(
                columns=["Action1_id", "True_Label", "Predicted_Label"]
            )

        for key in group_alignments.groups.keys():

            recipe1_filename = os.path.join(recipe_folder, key[0] + ".conllu")
            recipe2_filename = os.path.join(recipe_folder, key[1] + ".conllu")

            embedding_vector1, vector_lookup_list1, recipe_dict1 = fetch_recipe(
                recipe1_filename, emb_model, tokenizer, device, embedding_name,
            )
            embedding_vector2, vector_lookup_list2, recipe_dict2 = fetch_recipe(
                recipe2_filename, emb_model, tokenizer, device, embedding_name,
            )

            recipe_pair_alignment = group_alignments.get_group(key)

            for node in recipe_dict1[1:]:

                if mode == "Training":
                    optimizer.zero_grad()

                # True Action Id
                action_line = recipe_pair_alignment.loc[
                    recipe_pair_alignment["token1"] == node["Action_id"]
                ]

                if not action_line.empty:

                    true_label = action_line["token2"].item()

                    # True Action Id index
                    labels = [
                        i
                        for i, node in enumerate(recipe_dict2)
                        if node["Action_id"] == true_label
                    ]
                    labels_tensor = torch.LongTensor([labels[0]]).to(device)

                    action1 = node["Action"]
                    parent_list1 = node["Parent_List"]
                    child_list1 = node["Child_List"]

                    # Generate predictions using our Alignment Model

                    if model_name == "Alignment Model":
                        prediction = model(
                            action1.to(device),
                            parent_list1,
                            child_list1,
                            embedding_vector1,
                            vector_lookup_list1,
                            recipe_dict2,
                            embedding_vector2,
                            vector_lookup_list2,
                        )

                    elif model_name == "Simple Model":
                        prediction = model(
                            action1.to(device),
                            embedding_vector1,
                            vector_lookup_list1,
                            recipe_dict2,
                            embedding_vector2,
                            vector_lookup_list2,
                        )

                    # print(prediction)

                    num_actions += 1

                    # Predicted Action Id
                    pred_label = recipe_dict2[torch.argmax(prediction).item()][
                        "Action_id"
                    ]

                    if true_label == pred_label:
                        correct_predictions += 1

                    if mode == "Training" or mode == "Validation":
                        loss = criterion(prediction, labels_tensor)  # Loss

                        if mode == "Training" and not true_label == pred_label:
                            loss.backward()
                            optimizer.step()

                        total_loss += loss.item()
                        step += 1

                    elif mode == "Testing":

                        results_dict = {
                            "Action1_id": node["Action_id"],
                            "True_Label": true_label,
                            "Predicted_Label": pred_label,
                        }

                        # Store the prediction
                        results_df = results_df.append(results_dict, ignore_index=True)

        if mode == "Training" or mode == "Validation":

            return correct_predictions, num_actions, total_loss, step

        elif mode == "Testing":

            return correct_predictions, num_actions, results_df

        return None

    #####################################

    def train(
        self, dish_list, embedding_name, emb_model, tokenizer, model, criterion, optimizer, device
    ):
        """
        Train Function

        Parameters
        ----------
        dish_list : List
            List of dishes.
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        criterion : Cross Entropy Loss Function
            Loss Function.
        optimizer : Adam optimizer object
            Optimizer.
        device : object
            torch device where model tensors are saved.

        """

        train_loss = 0.0
        step = 0
        correct_predictions = 0
        num_actions = 0

        model.train()

        mode = "Training"
        
        for dish in dish_list:

            correct_predictions, num_actions, train_loss, step = self.run_model(
                dish=dish,
                embedding_name = embedding_name,
                emb_model=emb_model,
                tokenizer=tokenizer,
                model=model,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
                total_loss=train_loss,
                step=step,
                correct_predictions=correct_predictions,
                num_actions=num_actions,
                mode=mode,
            )

        average_train_loss = train_loss / (step - 1)
        average_train_accuracy = correct_predictions * 100 / num_actions

        return average_train_loss, average_train_accuracy

    #####################################

    def valid(self, dish, embedding_name, emb_model, tokenizer, model, criterion, device):
        """
        Valid Function

        Parameters
        ----------
        dish : String
            Dish.
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        criterion : Cross Entropy Loss Function
            Loss Function.
        device : object
            torch device where model tensors are saved.


        """

        valid_loss = 0.0
        step = 0
        correct_predictions = 0
        num_actions = 0

        model.eval()

        mode = "Validation"

        with torch.no_grad():

            correct_predictions, num_actions, valid_loss, step = self.run_model(
                dish=dish,
                embedding_name = embedding_name,
                emb_model=emb_model,
                tokenizer=tokenizer,
                model=model,
                device=device,
                criterion=criterion,
                total_loss=valid_loss,
                step=step,
                correct_predictions=correct_predictions,
                num_actions=num_actions,
                mode=mode,
            )

        average_valid_loss = valid_loss / step
        average_valid_accuracy = correct_predictions * 100 / num_actions

        return average_valid_loss, average_valid_accuracy

    #####################################

    def test(self, dish_list, embedding_name, emb_model, tokenizer, model, destination_folder, device):
        """
        Test Function

        Parameters
        ----------
        dish_list : List
            List of dishes.
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        destination_folder: String
            Destination folder.
        device : object
            torch device where model tensors are saved.
        """

        mode = "Testing"

        accuracy_list = (
            []
        )  # List of tuples (#correct predictions, #actions) for each dish in dish_list.

        for dish in dish_list:

            with torch.no_grad():

                correct_predictions, num_actions, results_df = self.run_model(
                    dish=dish,
                    embedding_name = embedding_name,
                    emb_model=emb_model,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    mode=mode,
                )

            dish_accuracy = correct_predictions * 100 / num_actions

            save_predictions(destination_folder, results_df, dish)

            accuracy_list.append([correct_predictions, num_actions, dish_accuracy])

        return accuracy_list

    #####################################

    def training_process(
        self,
        dish_list,
        embedding_name,
        emb_model,
        tokenizer,
        model,
        optimizer,
        criterion,
        num_epochs,
        saved_file_path,
        saved_metric_path,
        saved_graph_path,
        device,
    ):
        """
        Training Process function

        Parameters
        ----------
        dish_list : List
            List of all recipes in training set
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        optimizer : Adam optimizer object
            Optimizer.
        criterion : Cross Entropy Loss Function
            Loss Function.
        num_epochs : Int
            Number of Epochs.
        saved_file_path : String
            Trainied Model path.
        saved_metric_path : Sring
            Training Metrics file path.
        saved_graph_path : String
            Training/Validation accuracy and loss graph path.
        device : object
            torch device where model tensors are saved.


        """

        # initialize running values
        train_loss_list = []  # List of Training loss for every epoch
        valid_loss_list = []  # List of Validation loss for every epoch
        epoch_list = []  # List of Epochs
        train_accuracy_list = []
        valid_accuracy_list = []

        best_valid_accuracy = best_train_accuracy = float(
            "-Inf"
        )  # Stores the best Validation/Training Accuracy
        best_valid_loss = best_train_loss = float(
            "Inf"
        )  # Stores the best Validation/Training Loss

        valid_dish_id = len(dish_list) - 1  # Validation dish index
        train_dish_list = dish_list.copy()
        valid_dish = train_dish_list.pop(valid_dish_id)

        # Training loop

        for epoch in range(num_epochs):

            # Calculate average training loss per epoch
            average_train_loss, average_train_accuracy = self.train(
                train_dish_list,
                embedding_name,
                emb_model,
                tokenizer,
                model,
                criterion,
                optimizer,
                device,
            )

            # calculate average validation loss per epoch
            average_valid_loss, average_valid_accuracy = self.valid(
                valid_dish, embedding_name, emb_model, tokenizer, model, criterion, device
            )

            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)
            train_accuracy_list.append(average_train_accuracy)
            valid_accuracy_list.append(average_valid_accuracy)
            epoch_list.append(epoch)

            # print progress
            print(
                "Epoch [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}".format(
                    epoch + 1,
                    num_epochs,
                    average_train_loss,
                    average_train_accuracy,
                    average_valid_loss,
                    average_valid_accuracy,
                )
            )

            # saving checkpoint
            if best_valid_accuracy < average_valid_accuracy:
                best_valid_accuracy = average_valid_accuracy
                best_train_accuracy = average_train_accuracy
                best_valid_loss = average_valid_loss
                best_train_loss = average_train_loss
                save_checkpoint(
                    saved_file_path,
                    model,
                    optimizer,
                    best_valid_loss,
                    best_valid_accuracy,
                )
                save_metrics(
                    saved_metric_path,
                    train_loss_list,
                    valid_loss_list,
                    train_accuracy_list,
                    valid_accuracy_list,
                    epoch_list,
                )

        # create_acc_loss_graph(epoch_list, train_loss_list, valid_loss_list)

        save_metrics(
            saved_metric_path,
            train_loss_list,
            valid_loss_list,
            train_accuracy_list,
            valid_accuracy_list,
            epoch_list,
        )

        print("Finished Training!")

        create_acc_loss_graph(saved_metric_path, device, saved_graph_path)

        return (
            best_train_accuracy,
            best_valid_accuracy,
            best_train_loss,
            best_valid_loss,
        )

    #####################################

    def testing_process(
        self,
        dish_list,
        embedding_name,
        emb_model,
        tokenizer,
        model,
        optimizer,
        saved_file_path,
        saved_metric_path,
        destination_folder,
        device,
    ):
        """
        Testing Process function

        Parameters
        ----------
        dish_list : List
            List of all recipes in testing set
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        optimizer : Adam optimizer object
            Optimizer.
        saved_file_path : String
            Trainied Model path.
        saved_metric_path : Sring
            Training Metrics file path.
        destination_folder: String
            Destination folder.
        device : object
            torch device where model tensors are saved.

        """

        model, optimizer, _ = load_checkpoint(saved_file_path, model, optimizer, device)

        # train_loss_list, valid_loss_list, epoch_list = load_metrics(saved_metric_path, device)

        accuracy_list = self.test(
            dish_list, embedding_name, emb_model, tokenizer, model, destination_folder, device
        )

        total_correct_predictions = 0
        total_actions = 0

        model.eval()

        for i, accuracy_line in enumerate(accuracy_list):

            dish_accuracy = accuracy_line[2]

            total_correct_predictions += accuracy_line[0]
            total_actions += accuracy_line[1]

            print("Accuracy of Dish {} : {:.2f}".format(dish_list[i], dish_accuracy))

        model_accuracy = total_correct_predictions * 100 / total_actions

        print("Model Accuracy: {:.2f}".format(model_accuracy))

        return accuracy_list, model_accuracy, total_correct_predictions, total_actions
    
    
#####################################


    def run_folds(
        self,
        embedding_name,
        emb_model,
        tokenizer,
        model,
        optimizer,
        criterion,
        num_epochs,
        num_folds,
        device,
        with_feature=True,
    ):
        """
        Running 10 fold cross validation for alignment models

        Parameters
        ----------
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        model : AlignmentModel object
            Alignment model.
        optimizer : Adam optimizer object
        num_epochs : Int
            Number of Epochs.
        num_folds : Int
            Number of Folds.
        device : object
            torch device where model tensors are saved.
        with_feature : boolean; Optional
            Check whether to add features or not. Default value True.

        """

        dish_list = os.listdir(folder)

        dish_list = [dish for dish in dish_list if not dish.startswith(".")]
        dish_list.sort()

        fold_result_df = pd.DataFrame(
            columns=[
                "Fold",
                "Train Loss",
                "Train_Accuracy",
                "Valid Loss",
                "Valid_Accuracy",
                "Test_Accuracy",
                "Correct_Predictions",
                "Num_Actions",
            ]
        )  # , "Test_Dish1_accuracy", "Test_Dish2_accuracy"])

        test_dish_id = len(dish_list) - 1 

        if with_feature:
            destination_folder = destination_folder1

        else:
            destination_folder = destination_folder2

        for fold in range(num_folds):

            start = datetime.now()

            saved_file_path = os.path.join(
                destination_folder, "model" + str(fold + 1) + ".pt"
            )  # Model saved path
            saved_metric_path = os.path.join(
                destination_folder, "metric" + str(fold + 1) + ".pt"
            )  # Metric saved path
            saved_graph_path = os.path.join(destination_folder, "loss_acc_graph.png")

            train_dish_list = dish_list.copy()
            test_dish_list = [
                train_dish_list.pop(test_dish_id)
            ]  # , train_dish_list.pop(test_dish_id - 1)]

            test_dish_id -= 1

            if test_dish_id == -1:

                test_dish_id = len(dish_list) - 1

            print("Fold [{}/{}]".format(fold + 1, num_folds))

            print("-------Training-------")

            (
                train_accuracy,
                valid_accuracy,
                train_loss,
                valid_loss,
            ) = self.training_process(
                train_dish_list,
                embedding_name,
                emb_model,
                tokenizer,
                model,
                optimizer,
                criterion,
                num_epochs,
                saved_file_path,
                saved_metric_path,
                saved_graph_path,
                device,
            )

            print("-------Testing-------")

            (
                test_accuracy_list,
                test_accuracy,
                total_correct_predictions,
                total_actions,
            ) = self.testing_process(
                test_dish_list,
                embedding_name,
                emb_model,
                tokenizer,
                model,
                optimizer,
                saved_file_path,
                saved_metric_path,
                destination_folder,
                device,
            )

            fold_result = {
                "Fold": fold + 1,
                "Train Loss": train_loss,
                "Train_Accuracy": train_accuracy,
                "Valid Loss": valid_loss,
                "Valid_Accuracy": valid_accuracy,
                "Test_Accuracy": test_accuracy,
                "Correct_Predictions": total_correct_predictions,
                "Num_Actions": total_actions,
            }  # ,
            # "Test_Dish1_accuracy" : test_accuracy_list[0][2],
            # "Test_Dish2_accuracy" : test_accuracy_list[1][2]}

            fold_result_df = fold_result_df.append(fold_result, ignore_index=True)

            end = datetime.now()

            elapsedTime = end - start
            elapsed_duration = divmod(elapsedTime.total_seconds(), 60)

            print(
                "Time elapsed: {} mins and {:.2f} secs".format(
                    elapsed_duration[0], elapsed_duration[1]
                )
            )
            print("--------------")

        save_result_path = os.path.join(destination_folder, "fold_results.tsv")

        # Saving the results
        fold_result_df.to_csv(save_result_path, sep="\t", index=False, encoding="utf-8")

        print("Fold Results saved in ==>" + save_result_path)
        

#####################################
        

    def test_simple_model(self, embedding_name, emb_model, tokenizer, simple_model, device):
        """
        Testing Cosine Similarity Baseline

        Parameters
        ----------
        embedding_name : String
            Embedding name Bert/Elmo
        emb_model : Embedding Model object
            Model.
        tokenizer : Tokenizer object
            Tokenizer.
        simple_model : SimpleModel object
            Simple Baseline model.
        device : object
            torch device where model tensors are saved.

        """

        total_correct_predictions = 0
        total_actions = 0

        dish_list = os.listdir(folder)

        test_result_df = pd.DataFrame(columns=["Dish", "Correct_Predictions", "Num_Actions","Accuracy"])

        dish_list = [dish for dish in dish_list if not dish.startswith(".")]
        dish_list.sort()

        saved_file_path = os.path.join(
            destination_folder3, "model_result.tsv"
        )  # Model saved path

        for dish in dish_list:

            correct_predictions, num_actions, results_df = self.run_model(
                dish,
                emb_model,
                tokenizer,
                simple_model,
                device,
                embedding_name,
                mode="Testing",
                model_name="Simple Model",
            )

            save_predictions(destination_folder3, results_df, dish)

            accuracy = correct_predictions * 100 / num_actions

            test_result = {
                "Dish": dish,
                "Correct_Predictions": correct_predictions,
                "Num_Actions": num_actions,
                "Accuracy": accuracy,
            }

            test_result_df = test_result_df.append(test_result, ignore_index=True)

            total_correct_predictions += correct_predictions
            total_actions += num_actions

        model_accuracy = total_correct_predictions * 100 / total_actions

        test_result = {
            "Dish": "Overall",
            "Correct_Predictions": total_correct_predictions,
            "Num_Actions": total_actions,
            "Accuracy": model_accuracy,
        }

        test_result_df = test_result_df.append(test_result, ignore_index=True)

        print("Model Accuracy: {:.2f}".format(model_accuracy))

        test_result_df.to_csv(saved_file_path, sep="\t", index=False, encoding="utf-8")

        print("Results saved in ==>" + saved_file_path)
        

#####################################

        
        
    def basic_training(self,
                       model,
                       dish_list,
                       saved_file_path):
        
            action_pair_list = list()
        
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
            
                for key in group_alignments.groups.keys():

                    recipe1_filename = os.path.join(recipe_folder, key[0] + ".conllu")
                    recipe2_filename = os.path.join(recipe_folder, key[1] + ".conllu")
                    
                    recipe_pair_alignment = group_alignments.get_group(key)
                
                    # Generate action pairs in text format 
                    _, _, action_pairs = model.generate_action_pairs(recipe_pair_alignment, recipe1_filename, recipe2_filename)
                
                    action_pair_list.extend(action_pairs)
        
            # Generate Vocabulary of action pairs
            action_pair_vocabulary = model.generate_vocabulary(action_pair_list)
        
            # Save Vocab 
            save_vocabulary(saved_file_path, action_pair_vocabulary)
        

#####################################
    
    
    def basic_testing(self,
                      model,
                      dish_list,
                      saved_file_path,
                      destination_folder,
                      test_result_df):
        
        total_correct_predictions = 0
        total_actions = 0
   
        
        
        vocab = load_vocabulary(saved_file_path) #load saved vocabulary
        #print(vocab)

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
            
            num_actions = 0
            correct_predictions = 0
            
            results_df = pd.DataFrame(
                columns=["Action", "True_Label", "Predicted_Label"]
            )
            
            for key in group_alignments.groups.keys():

               recipe1_filename = os.path.join(recipe_folder, key[0] + ".conllu")
               recipe2_filename = os.path.join(recipe_folder, key[1] + ".conllu")
                
               recipe_pair_alignment = group_alignments.get_group(key)
               
               _, parsed_recipe2, action_pairs = model.generate_action_pairs(recipe_pair_alignment, recipe1_filename, recipe2_filename)
               
               correct_predictions, num_actions, results_df = model.fetch_aligned_actions(action_pairs, 
                                                                                          vocab, 
                                                                                          parsed_recipe2,
                                                                                          correct_predictions,
                                                                                          num_actions,
                                                                                          results_df)
               
            total_correct_predictions += correct_predictions
            total_actions += num_actions
               
            save_predictions(destination_folder, results_df, dish)

            accuracy = correct_predictions * 100 / num_actions
            
            print("Dish Accuracy: {:.2f}".format(accuracy))

            test_result = {
                "Dish": dish,
                "Correct_Predictions": correct_predictions,
                "Num_Actions": num_actions,
                "Accuracy": accuracy,
                }
            
            test_result_df = test_result_df.append(test_result, ignore_index=True)
            
        model_accuracy = total_correct_predictions * 100 / total_actions
        
        print("Model Accuracy: {:.2f}".format(model_accuracy))

        return model_accuracy, total_correct_predictions, total_actions, test_result_df
        

#####################################


        
    def run_naive_folds( self,
        model,
        num_folds
        ):
        """
        Running 10 fold cross validation for naive baseline

        Parameters
        ----------
        model : NaiveModel object
            Naive Baseline model
        num_folds : Int

        """

        dish_list = os.listdir(folder)

        dish_list = [dish for dish in dish_list if not dish.startswith(".")]
        dish_list.sort()

        fold_result_df = pd.DataFrame(
            columns=[
                "Fold",
                "Test_Accuracy",
                "Correct_Predictions",
                "Num_Actions",
            ]
        )  # , "Test_Dish1_accuracy", "Test_Dish2_accuracy"])

        test_dish_id = len(dish_list) - 1
        
        destination_folder = destination_folder4
        
        test_result_df = pd.DataFrame(columns=["Dish", "Correct_Predictions", "Num_Actions","Accuracy"])
        overall_predictions = 0
        overall_actions = 0 

        for fold in range(num_folds):

            start = datetime.now()

            saved_file_path = os.path.join(
                destination_folder, "model" + str(fold + 1) + ".pt"
            )  # Model saved path

            train_dish_list = dish_list.copy()
            test_dish_list = [
                train_dish_list.pop(test_dish_id)
            ]  # , train_dish_list.pop(test_dish_id - 1)]

            test_dish_id -= 1

            if test_dish_id == -1:

                test_dish_id = len(dish_list) - 1

            print("Fold [{}/{}]".format(fold + 1, num_folds))

            print("-------Training-------")

            self.basic_training(
                model,
                train_dish_list,
                saved_file_path,
            )

            print("-------Testing-------")

            (
                test_accuracy,
                total_correct_predictions,
                total_actions,
                test_result_df
            ) = self.basic_testing(
                model,
                test_dish_list,
                saved_file_path,
                destination_folder,
                test_result_df
            )
                
            overall_predictions += total_correct_predictions
            overall_actions += total_actions

            fold_result = {
                "Fold": fold + 1,
                "Test_Accuracy": test_accuracy,
                "Correct_Predictions": total_correct_predictions,
                "Num_Actions": total_actions,
            }  # ,
            # "Test_Dish1_accuracy" : test_accuracy_list[0][2],
            # "Test_Dish2_accuracy" : test_accuracy_list[1][2]}

            fold_result_df = fold_result_df.append(fold_result, ignore_index=True)

            end = datetime.now()

            elapsedTime = end - start
            elapsed_duration = divmod(elapsedTime.total_seconds(), 60)

            print(
                "Time elapsed: {} mins and {:.2f} secs".format(
                    elapsed_duration[0], elapsed_duration[1]
                )
            )
            print("--------------")
            
            
        overall_accuracy = overall_predictions * 100 / overall_actions
        
        print("Overall Model Accuracy: {:.2f}".format(overall_accuracy))
        
        fold_result = {
                "Fold": 'Overall',
                "Test_Accuracy": overall_accuracy,
                "Correct_Predictions": overall_predictions,
                "Num_Actions": overall_actions,
            }
        
        fold_result_df = fold_result_df.append(fold_result, ignore_index=True)

        save_result_path = os.path.join(destination_folder, "fold_results.tsv")
        
        results_file_path = os.path.join(
            destination_folder, "model_result.tsv"
        )  # Model saved path

        # Saving the results
        fold_result_df.to_csv(save_result_path, sep="\t", index=False, encoding="utf-8")
        
        test_result_df.to_csv(results_file_path, sep="\t", index=False, encoding="utf-8")

        print("Fold Results saved in ==>" + save_result_path)
        
