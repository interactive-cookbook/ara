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
Main Function

Usage: python main.py model_name embedding_name

(model_name: ['Sequence' : Sequential ordering of alignments, 
              'Simple' : Cosine model, 
              'Naive' : Common Action Pair Heuristics model,
              'Alignment-no-feature' : Base Alignment model, 
              'Alignment-with-feature' : Extended Alignment model])

(embedding_name : ['bert' : BERT embeddings,
                   'elmo' : ELMO embeddings])
"""

# importing libraries

import torch
import flair
import argparse
import torch.nn as nn
import torch.optim as optim

from model import AlignmentModel
from simple_model import SimpleModel
from sequence_model import SequenceModel
from naive_model import NaiveModel
from transformers import BertTokenizer, BertModel
from flair.data import Sentence
from flair.embeddings import ELMoEmbeddings
from training_testing import Folds
from constants import OUTPUT_DIM, LR, EPOCHS, FOLDS, HIDDEN_DIM1, HIDDEN_DIM2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flair.device = device


def main():
    
    parser = argparse.ArgumentParser(description = 'Automatic Alignment model')
    parser.add_argument('model_name', type=str, help='Model Name')
    parser.add_argument('--embedding_name', type=str, default='bert', help='Embedding Name (Default is bert)')
    args = parser.parse_args()

    model_name = args.model_name
    
    embedding_name = args.embedding_name

    print("-------Loading Model-------")

    # Loading Model definition
    
    if embedding_name == 'bert' :

        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )  # Bert Tokenizer
    
        emb_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(
            device
        )  # Bert Model for Embeddings
        
        embedding_dim = emb_model.config.to_dict()[
            "hidden_size"
        ]  # BERT embedding dimension
    
        # print(bert)
    
    elif embedding_name == 'elmo' :
        
        tokenizer = Sentence #Flair sentence for ELMo embeddings
        
        emb_model = ELMoEmbeddings('small')
        
        embedding_dim = emb_model.embedding_length

    TT = Folds()  # calling the Training and Testing class

    if model_name == "Alignment-with-feature":

        model = AlignmentModel(embedding_dim, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, device).to(
            device
        )  # Out Alignment Model with features

        print(model)
        """for name, param in model.named_parameters():
            if param.requires_grad:
                    print(name)"""

        optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer for training
        criterion = nn.CrossEntropyLoss()  # Loss function

        print("-------Cross Validation Folds-------")

        TT.run_folds(
            embedding_name, 
            emb_model, tokenizer, model, optimizer, criterion, EPOCHS, FOLDS, device
        )

    elif model_name == "Alignment-no-feature":

        model = AlignmentModel(
            embedding_dim, HIDDEN_DIM1, HIDDEN_DIM2, OUTPUT_DIM, device, False
        ).to(
            device
        )  # Out Alignment Model w/o features

        print(model)

        optimizer = optim.Adam(model.parameters(), lr=LR)  # optimizer for training
        criterion = nn.CrossEntropyLoss()  # Loss function

        print("-------Cross Validation Folds-------")

        TT.run_folds(
            embedding_name,
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

    elif model_name == "Simple":

        simple_model = SimpleModel(embedding_dim, device).to(device) # Simple Cosine Similarity Baseline

        print(simple_model)

        print("-------Testing (Simple Baseline) -------")

        TT.test_simple_model(embedding_name, emb_model, tokenizer, simple_model, device)
        
        
    elif model_name == 'Naive':
        
        naive_model = NaiveModel(device) # Naive Common Action Pair Heuristics Baseline
        
        print('Common Action Pair Heuristics Model')
        
        print("-------Cross Validation Folds-------")
        
        TT.run_naive_folds(
            naive_model,
            FOLDS
            )
        
    elif model_name == 'Sequence':
        
        sequence_model = SequenceModel()
        
        print('Sequential Alignments')
        
        sequence_model.test_sequence_model()

    else:

        print(
            "Incorrect Argument: Model_name should be ['Simple', 'Naive', 'Alignment-no-feature', 'Alignment-with-feature']"
        )


if __name__ == "__main__":
    main()
