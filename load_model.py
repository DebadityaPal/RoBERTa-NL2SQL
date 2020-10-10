import torch
import os
from seq2sql_model_classes import Seq2SQL_v1
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

device = torch.device("cuda")

def get_roberta_model():

    # Initializing a RoBERTa configuration
    configuration = RobertaConfig()

    # Initializing a model from the configuration
    Roberta_Model = RobertaModel(configuration).from_pretrained("roberta-base")
    Roberta_Model.to(device)

    # Accessing the model configuration
    configuration = Roberta_Model.config

    #get the Roberta Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    return Roberta_Model, tokenizer, configuration


def get_seq2sql_model(roberta_hidden_layer_size, number_of_layers = 2,
                    hidden_vector_dimensions = 100,
                    number_lstm_layers = 2,
                    dropout_rate = 0.3,
                    load_pretrained_model=False, model_path=None):
    
    '''
    
    get_seq2sql_model
    Arguments:
    roberta_hidden_layer_size: sizes of hidden layers of Roberta model
    number_of_layers : total number of layers
    hidden_vector_dimensions : dimensions of hidden vectors
    number_lstm_layers : total number of lstm layers
    dropout_rate : value of dropout rate
    load_pretrained_model : want to load pretrained model(true or false)
    model_path : The path to the directory in which the model is contained
    
    Returns:
    model: returns the model
    
    '''

    # number_of_layers = "The Number of final layers of RoBERTa to be used in downstream task."
    # hidden_vector_dimensions : "The dimension of hidden vector in the seq-to-SQL module."
    # number_lstm_layers : "The number of LSTM layers." in seqtosqlmodule

    sql_main_operators = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    sql_conditional_operators = ['=', '>', '<', 'OP']

    number_of_neurons = roberta_hidden_layer_size * number_of_layers  # Seq-to-SQL input vector dimenstion

    model = Seq2SQL_v1(number_of_neurons, hidden_vector_dimensions, number_lstm_layers, dropout_rate, len(sql_conditional_operators), len(sql_main_operators))
    model = model.to(device)

    if load_pretrained_model:
        assert model_path != None
        if torch.cuda.is_available():
            res = torch.load(model_path)
        else:
            res = torch.load(model_path, map_location='cpu')
        model.load_state_dict(res['model'])

    return model

def get_optimizers(model, model_roberta,learning_rate_model=1e-3,learning_rate_roberta=1e-5):
    '''
    get_optimizers
    Arguments:
    model: returned model from get_seq2sql_model
    model_roberta : returned model from get_roberta_model
    fine_tune : want to fine tune(true or false)
    learning_rate_model : learning rate of model (from get_seq2sql_model)
    learning_rate_roberta : learning rate of roberta model (from get_roberta_model)
    
    Returns:
    opt: returns the optimised model (from get_seq2sql_model)
    opt_roberta : returns the optimised roberta model (from get_roberta_model)

    '''

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=learning_rate_model, weight_decay=0)

    opt_roberta = torch.optim.Adam(filter(lambda p: p.requires_grad, model_roberta.parameters()),
                                lr=learning_rate_roberta, weight_decay=0)

    return opt, opt_roberta

