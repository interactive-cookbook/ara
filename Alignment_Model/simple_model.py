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
Simple Baseline Model: Cosine Similarity
"""

import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    
    def __init__(self, embedding_dim, device):

        super().__init__()

        self.device = device

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.embedding_dim = embedding_dim
        
        
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
             Embedding vector of the node.

        """

        token_list = torch.zeros(1, self.embedding_dim).to(
            self.device
        )  # embedding vector for the node sequence

        if len(node):

            emb_id_list = []

            for token_id in node:

                emb_id_list.extend(vector_lookup_list[token_id.item()])

            for emb_id in emb_id_list:

                emb = torch.unsqueeze(
                    embedding_vectors[emb_id], dim=0
                )  # 1 X embedding_dim

                token_list = token_list.add(emb)  # 1 X embedding_dim

            embedding = token_list / len(emb_id_list)  # 1 X embedding_dim

        else:

            embedding = token_list

        return embedding  # 1 X embedding_dim

    def forward(
        self,
        action1,
        embedding_vectors1,
        vector_lookup_list1,
        recipe2_actions,
        embedding_vectors2,
        vector_lookup_list2,
    ):
        """
        Simple Baseline model using argmax(cosine similarities)

        Parameters
        ----------
        action1 : Tensor  action_sequence_length
            Action Node from Recipe1.
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
        cosine_list_tensor : Tensor 1 X length of recipe2_actions
            Cosine Similarities of all class alignment, where class is an action in Recipe2 (including none of these).

        """

        action1_emb = self.sequence_embedding(
            action1, embedding_vectors1, vector_lookup_list1
        )
        cosine_list = []

        for node in recipe2_actions:

            action2_emb = self.sequence_embedding(
                node["Action"], embedding_vectors2, vector_lookup_list2
            )

            cosine_similarity = self.cos(action1_emb, action2_emb).item()

            cosine_list.append(cosine_similarity)

        cosine_list_tensor = torch.Tensor(cosine_list)

        return cosine_list_tensor
