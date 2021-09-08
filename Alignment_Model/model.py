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
Alignment Model
"""

import torch
import torch.nn as nn


# Encoder to generate context vector using BERT/ELMO embeddings
class Encoder(nn.Module):
    def __init__(self, embedding_dim, device, feature_dim, with_feature):
        """
        Constructor

        Parameters
        ----------
        embedding_dim : int
            Embedding dimension.
        device : object
            torch device where model tensors are saved.
        feature_dim : int
            Number of features. For base model, 1 and for extended model, 3.
        with_feature : boolean
            Check whether to add features or not.
        """

        super().__init__()

        self.device = device

        self.feature_dim = feature_dim

        self.none_action_vector = nn.parameter.Parameter(torch.randn(1, embedding_dim))

        self.with_feature = with_feature

        self.embedding_dim = embedding_dim  # Embedding dim

        self.seqlstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim)

        self.parentlstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim)

        self.childlstm = nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim)

        # self.encodingmlp = nn.Sequential(nn.Linear(self.embedding_dim * self.feature_dim, self.embedding_dim * self.feature_dim,), # Linear layer1
        # nn.Sigmoid())

    def sequence_embedding(self, node, embedding_vectors, vector_lookup_list):
        """
         Sequence Embedding Function


         Parameters
         ----------
         node : Tensor node_sequence_length
             Node.
         embedding_vectors : Dict
             Embedding dictionary for a particular Recipe;
                 where keys are vector_lookup_list token_ids and values are their corresponding word embeddings (BERT/ELMO).
         vector_lookup_list : Dict
             Look up dictionary for a particular Recipe embeddings;
                 where key is the Conllu file token 'id' and values are list of token_ids generated using BERT/ELMO.

         Returns
         -------
        embedding: Tensor 1 X embedding_dim
             Embedding vector of the node. Returns none_action_vector for null action.

        """

        if len(node):

            # token_list = torch.zeros(1, self.embedding_dim).to(self.device) # embedding vector for the node sequence

            emb_id_list = []

            for token_id in node:

                emb_id_list.extend(vector_lookup_list[token_id.item()])

            input = [embedding_vectors[emb_id] for emb_id in emb_id_list]

            embedding = torch.squeeze(
                self.seqlstm(torch.cat(input).view(len(input), 1, -1))[1][1], dim=0
            )

            return embedding

            # return embedding # .view(1, 1, 1, self.embedding_dim)

            """for emb_id in emb_id_list :
                
                emb = torch.unsqueeze(embedding_vectors[emb_id], dim = 0) # 1 X embedding_dim
            
                token_list = token_list.add(emb) # 1 X embedding_dim
        
            embedding = token_list / len(emb_id_list) # 1 X embedding_dim"""

        else:

            return self.none_action_vector

    # return embedding # 1 X embedding_dim

    def forward(
        self, action, parent_list, child_list, embedding_vectors, vector_lookup_list
    ):
        """
        Encoder Model

        Parameters
        ----------
        action : Tensor  action_sequence_length
            Action Node.
        parent_list : List of Tensors
            List of Parent Nodes for Action node.
        child_list : List of Tensors
            List of Child Nodes for Action node.
        embedding_vectors : Dict
             Embedding dictionary for a particular Recipe;
                 where keys are vector_lookup_list token_ids and values are their corresponding word embeddings (BERT/ELMO).
        vector_lookup_list : Dict
             Look up dictionary for a particular Recipe embeddings;
                 where key is the Conllu file token 'id' and values are list of token_ids generated using BERT/ELMO.

        Returns
        -------
        encoding : Tensor feature_dim X embedding_dim
            Encoding vector for input.
            For extended model, (action, parent_list, child_list) and for base model, only (action).

        """

        # generating embedding vectors for action node sequence
        action_context = self.sequence_embedding(
            action, embedding_vectors, vector_lookup_list
        )  # 1 X embedding_dim

        if self.with_feature:

            parent_emb_list = child_emb_list = []

            # parent_list_context = torch.zeros(1, self.embedding_dim).to(self.device) # embedding vector for all the parent node sequence
            # child_list_context = torch.zeros(1, self.embedding_dim).to(self.device) # embedding vector for all the child node sequence

            if len(parent_list):
                # generating embedding vectors for each parent node sequence and concatinating them up to one vector
                for parent in parent_list:

                    parent_context = self.sequence_embedding(
                        parent, embedding_vectors, vector_lookup_list
                    )  # 1 X embedding_dim
                    parent_emb_list.append(parent_context)  # 1 X embedding_dim

                    parent_list_context = torch.squeeze(
                        self.parentlstm(
                            torch.cat(parent_emb_list).view(len(parent_emb_list), 1, -1)
                        )[1][1],
                        dim=0,
                    )

            else:
                parent_list_context = torch.zeros(1, self.embedding_dim).to(self.device)

            if len(child_list):
                # generating embedding vectors for each child node sequence and concatinating them up to one vector
                for child in child_list:

                    child_context = self.sequence_embedding(
                        child, embedding_vectors, vector_lookup_list
                    )  # 1 X embedding_dim
                    child_emb_list.append(child_context)  # 1 X embedding_dim

                    child_list_context = torch.squeeze(
                        self.childlstm(
                            torch.cat(child_emb_list).view(len(child_emb_list), 1, -1)
                        )[1][1],
                        dim=0,
                    )

            else:
                child_list_context = torch.zeros(1, self.embedding_dim).to(self.device)

            context = torch.cat(
                [action_context, parent_list_context, child_list_context], dim=0
            )  # 3 X embedding_dim

            # return context # 3 X embedding_dim

        else:

            context = action_context

        encoding = torch.unsqueeze(torch.flatten(context), dim=0)

        # encoding = self.encodingmlp(encoding)

        return encoding


# Linear Classifier to generate probability of alignment between an action from Recipe1 and an action from Recipe2
class Scorer(nn.Module):
    def __init__(
        self, feature_dim, embedding_dim, hidden_dim1, hidden_dim2, output_dim, device
    ):
        """
        Contructor

        Parameters
        ----------
        feature_dim : int
            Number of features. For base model, 1 and for extended model, 3.
        embedding_dim : int
            Embedding dimension.
        hidden_dim1 : int
            Hidden dimension for MLP layer 1 in scorer.
        hidden_dim2 : int
            Hidden dimension for MLP layer 2 in scorer.
        output_dim : int
            Output dimension of MLP in Scorer (always 1).
        device : object
            torch device where model tensors are saved.
        with_feature : boolean
            Check whether to add features or not.
        """

        super().__init__()

        self.device = device
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        # self.weights = nn.parameter.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        self.linear_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim * feature_dim, hidden_dim1),  # Linear layer1
            nn.Sigmoid(),
            nn.Linear(hidden_dim1, hidden_dim2),  # Linear layer1
            nn.Sigmoid(),
            nn.Linear(hidden_dim2, output_dim),  # Linear layer1
            nn.Sigmoid(),
        ).to(
            self.device
        )  # MLP structure

        # self.dropout = nn.Dropout(dropout) # Dropout

    def forward(self, encoding1, encoding2):
        """
        Scorer model using an element-wise multiplication and MLP

        Parameters
        ----------
        encoding1 : Tensor feature_dim X embedding_dim
            Encoding vector for action node from Recipe 1.
        encoding2 : Tensor feature_dim X embedding_dim
            Encoding vector for action node from Recipe 2.

        Returns
        -------
        pred : Tensor of size 1
            Classifier probability prediction.

        """

        # encoding1 = torch.unsqueeze(torch.flatten(context1), dim = 0) # 1 X feature_dim * embedding_dim
        # encoding2 = torch.unsqueeze(torch.flatten(context2), dim = 0)# 1 X feature_dim * embedding_dim

        encoding = torch.mul(encoding1, encoding2)  # 1 X feature_dim * embedding_dim

        pred = self.linear_classifier(encoding)  # 1 X output_dim

        # pred = self.dropout(pred) # 1 X output_dim

        pred = torch.squeeze(pred, 1)  # 1

        return pred  # 1


# Alignment model to predict which action from Recipe2 aligns (including none) with a particular action (Action1) from Recipe1
class AlignmentModel(nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim1, hidden_dim2, output_dim, device, with_feature=True
    ):
        """
        Alignment Model

        Parameters
        ----------
        embedding_dim : int
            Embedding Dimension
        hidden_dim1 : int
            Hidden dimension for MLP layer 1 in scorer.
        hidden_dim2 : int
            Hidden dimension for MLP layer 2 in scorer.
        output_dim : int
            Output dimension of MLP in Scorer (always 1).
        device : object
            torch device where model tensors are saved.
        with_feature : boolean; Optional
            Check whether to add features or not. Default value True.
        """

        super().__init__()
        
        
        self.embedding_dim = embedding_dim

        if with_feature:
            self.feature_dim = 3

        else:
            self.feature_dim = 1

        self.encoder = Encoder(
            self.embedding_dim, device, self.feature_dim, with_feature
        )  # Encoder class object

        self.scorer = Scorer(
            self.feature_dim,
            self.embedding_dim,
            hidden_dim1,
            hidden_dim2,
            output_dim,
            device,
        )  # Classifier class object

    def forward(
        self,
        action1,
        parent_list1,
        child_list1,
        embedding_vectors1,
        vector_lookup_list1,
        recipe2_actions,
        embedding_vectors2,
        vector_lookup_list2,
    ):
        """
        Alignment Model

        Parameters
        ----------
        action1 : Tensor  action_sequence_length
            Action Node from Recipe1.
        parent_list1 : List of Tensors
            List of Parent Nodes for Action node from Recipe1.
        child_list1 : List of Tensors
            List of Child Nodes for Action node from Recipe1.
        embedding_vectors1 : Dict
             Embedding dictionary for Recipe 1;
                 where keys are vector_lookup_list token_ids and values are their corresponding word embeddings (BERT/ELMO).
        vector_lookup_list1 : Dict
             Look up dictionary for Recipe 1 embeddings;
                 where key is the Conllu file token 'id' and values are list of token_ids generated using BERT/ELMO.
        recipe2_actions : List of dict
            List of all action dictionaries from Recipe2
                (action dictionaries contain action node and their corresponding lists of parent nodes and child nodes).
        embedding_vectors2 : Dict
             Embedding dictionary for Recipe 2;
                 where keys are vector_lookup_list token_ids and values are their corresponding word embeddings (BERT/ELMO).
        vector_lookup_list2 : Dict
             Look up dictionary for Recipe 2 embeddings;
                 where key is the Conllu file token 'id' and values are list of token_ids generated using BERT/ELMO.

        Returns
        -------
        prediction_list : Tensor # 1 X length of recipe2_actions
            Probabilities of all class alignment, where class is an action in Recipe2 (including none of these).

        """

        prediction_list = (
            []
        )  # List of probability predictions between an action in Recipe1 and all actions in Recipe2 (including none of these)

        encoding1 = self.encoder(
            action1, parent_list1, child_list1, embedding_vectors1, vector_lookup_list1
        )  # embedding_dim X 3

        for node in recipe2_actions:

            encoding2 = self.encoder(
                node["Action"],
                node["Parent_List"],
                node["Child_List"],
                embedding_vectors2,
                vector_lookup_list2,
            )  # embedding_dim X 3

            pred = self.scorer(encoding1, encoding2)  # Probability prediction

            prediction_list.append(pred)

        prediction_list = torch.stack(
            prediction_list, dim=1
        )  # 1 X length of recipe2_actions

        return prediction_list  # 1 X length of recipe2_actions
