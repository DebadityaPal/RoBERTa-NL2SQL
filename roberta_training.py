import torch

device = torch.device("cuda")

def get_where_column(conds):
    """
    [ [where_column, where_operator, where_value],
      [where_column, where_operator, where_value], ...
    ]
    """
    where_column = []
    for cond in conds:
        where_column.append(cond[0])
    return where_column


def get_where_operator(conds):
    """
    [ [where_column, where_operator, where_value],
      [where_column, where_operator, where_value], ...
    ]
    """
    where_operator = []
    for cond in conds:
        where_operator.append(cond[1])
    return where_operator


def get_where_value(conds):
    """
    [ [where_column, where_operator, where_value],
      [where_column, where_operator, where_value], ...
    ]
    """
    where_value = []
    for cond in conds:
        where_value.append(cond[2])
    return where_value


def get_ground_truth_values(canonical_sql_queries):
    
    ground_select_column = []
    ground_select_aggregate = []
    ground_where_number = []
    ground_where_column = []
    ground_where_operator = []
    ground_where_value = []
    for _, canonical_sql_query in enumerate(canonical_sql_queries):
        ground_select_column.append( canonical_sql_query["sel"] )
        ground_select_aggregate.append( canonical_sql_query["agg"])

        conds = canonical_sql_query['conds']
        if not canonical_sql_query["agg"] < 0:
            ground_where_number.append( len( conds ) )
            ground_where_column.append( get_where_column(conds) )
            ground_where_operator.append( get_where_operator(conds) )
            ground_where_value.append( get_where_value(conds) )
        else:
            raise EnvironmentError
    return ground_select_column, ground_select_aggregate, ground_where_number, ground_where_column,\
           ground_where_operator, ground_where_value


def get_wemb_roberta(roberta_config, model_roberta, tokenizer, nlu_t, hds, max_seq_length, num_out_layers_n=1, num_out_layers_h=1):
    '''
    wemb_n : word embedding of natural language question
    wemb_h : word embedding of header
    l_n : length of natural question
    l_hs : length of header
    nlu_tt : Natural language double tokenized
    t_to_tt_idx : map first level tokenization to second level tokenization
    tt_to_t_idx : map second level tokenization to first level tokenization
    '''
    # get contextual output of all tokens from RoBERTa
    all_encoder_layer, i_nlu, i_headers,\
    l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx = get_roberta_output(model_roberta, tokenizer, nlu_t, hds, max_seq_length)
    # all_encoder_layer: RoBERTa outputs from all layers.
    # i_nlu: start and end indices of question in tokens
    # i_headers: start and end indices of headers

    # get the wemb
    wemb_n = get_wemb_n(i_nlu, l_n, roberta_config.hidden_size, roberta_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_n)

    wemb_h = get_wemb_h(i_headers, l_hpu, l_hs, roberta_config.hidden_size, roberta_config.num_hidden_layers, all_encoder_layer,
                        num_out_layers_h)

    return wemb_n, wemb_h, l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx

def get_roberta_output(model_roberta, tokenizer, nlu_t, headers, max_seq_length):
    """
    Here, input is toknized further by RoBERTa Tokenizer and fed into RoBERTa
    INPUT
    :param model_roberta:
    :param tokenizer: RoBERTa toknizer
    :param nlu: Question
    :param nlu_t: tokenized natural_language_utterance.
    :param headers: Headers of the table
    :param max_seq_length: max input token length
    OUTPUT
    tokens: RoBERTa input tokens
    nlu_tt: RoBERTa-tokenized input natural language questions
    orig_to_tok_index: map the index of 1st-level-token to the index of 2nd-level-token
    tok_to_orig_index: inverse map.
    """

    l_n = []
    l_hs = []  # The length of columns for each batch

    input_ids = []
    input_mask = []

    i_nlu = []  # index to retreive the position of contextual vector later.
    i_headers = []

    nlu_tt = []

    t_to_tt_idx = []
    tt_to_t_idx = []
    for b, nlu_t1 in enumerate(nlu_t):

        batch_headers = headers[b]
        l_hs.append(len(batch_headers))


        # 1. Tokenization using RoBERTa Tokenizer
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  
        for (i, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1))  
            sub_tokens = tokenizer.tokenize(token, is_pretokenized=True)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(i)
                nlu_tt1.append(sub_token) 
        nlu_tt.append(nlu_tt1)
        tt_to_t_idx.append(tt_to_t_idx1)
        t_to_tt_idx.append(t_to_tt_idx1)

        l_n.append(len(nlu_tt1))

        # <s> nlu </s> col1 </s> col2 </s> ...col-n </s>
        # 2. Generate RoBERTa inputs & indices.
        tokens, i_nlu1, i_batch_headers = generate_inputs(tokenizer, nlu_tt1, batch_headers)
        input_ids1 = tokenizer.convert_tokens_to_ids(tokens)

        # Input masks
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.
        while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)

        assert len(input_ids1) == max_seq_length
        assert len(input_mask1) == max_seq_length

        input_ids.append(input_ids1)
        input_mask.append(input_mask1)

        i_nlu.append(i_nlu1)
        i_headers.append(i_batch_headers)

    # Convert to tensor
    all_input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    all_input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)

    # 4. Generate RoBERTa output.
    check, _, all_encoder_layer = model_roberta(input_ids=all_input_ids, attention_mask=all_input_mask, output_hidden_states=True)
    all_encoder_layer = list(all_encoder_layer)

    assert all((check == all_encoder_layer[-1]).tolist())

    # 5. generate l_hpu from i_headers
    l_hpu = gen_l_hpu(i_headers)

    return all_encoder_layer, i_nlu, i_headers, \
           l_n, l_hpu, l_hs, \
           nlu_tt, t_to_tt_idx, tt_to_t_idx


def generate_inputs(tokenizer, nlu1_tok, hds1):
    tokens = []

    tokens.append("<s>")
    i_st_nlu = len(tokens)  # to use it later

    for token in nlu1_tok:
        tokens.append(token)
    i_ed_nlu = len(tokens)
    tokens.append("</s>")

    i_headers = []

    for i, hds11 in enumerate(hds1):
        i_st_hd = len(tokens)
        sub_tok = tokenizer.tokenize(hds11)
        tokens += sub_tok
        i_ed_hd = len(tokens)
        i_headers.append((i_st_hd, i_ed_hd))
        if i < len(hds1)-1:
            tokens.append("</s>")
        elif i == len(hds1)-1:
            tokens.append("</s>")
        else:
            raise EnvironmentError

    i_nlu = (i_st_nlu, i_ed_nlu)

    return tokens, i_nlu, i_headers

def gen_l_hpu(i_headers):
    """
    # Treat columns as if it is a batch of natural language utterance with batch-size = # of columns * # of batch_size
    i_headers = [(17, 18), (19, 21), (22, 23), (24, 25), (26, 29), (30, 34)])
    """
    l_hpu = []

    for i_header in i_headers:
        for index_pair in i_header:
            l_hpu.append(index_pair[1] - index_pair[0])

    return l_hpu

def get_wemb_n(i_nlu, l_n, hS, num_hidden_layers, all_encoder_layer, num_out_layers_n):
    """
    Get the representation of each tokens.
    """
    bS = len(l_n)
    l_n_max = max(l_n)
    wemb_n = torch.zeros([bS, l_n_max, hS * num_out_layers_n]).to(device)
    for b in range(bS):

        l_n1 = l_n[b]
        i_nlu1 = i_nlu[b]
        for i_noln in range(num_out_layers_n):
            i_layer = num_hidden_layers - 1 - i_noln
            st = i_noln * hS
            ed = (i_noln + 1) * hS
            wemb_n[b, 0:(i_nlu1[1] - i_nlu1[0]), st:ed] = all_encoder_layer[i_layer][b, i_nlu1[0]:i_nlu1[1], :]

    return wemb_n

def get_wemb_h(i_headers, l_hpu, l_hs, hS, num_hidden_layers, all_encoder_layer, num_out_layers_h):
    """
    As if
    [ [table-1-col-1-tok1, t1-c1-t2, ...],
       [t1-c2-t1, t1-c2-t2, ...].
       ...
       [t2-c1-t1, ...,]
    ]
    """
    bS = len(l_hs)
    l_hpu_max = max(l_hpu)
    num_of_all_hds = sum(l_hs)
    wemb_h = torch.zeros([num_of_all_hds, l_hpu_max, hS * num_out_layers_h]).to(device)
    b_pu = -1
    for b, i_header in enumerate(i_headers):
        for b1, index_pair in enumerate(i_header):
            b_pu += 1
            for i_nolh in range(num_out_layers_h):
                i_layer = num_hidden_layers - 1 - i_nolh
                st = i_nolh * hS
                ed = (i_nolh + 1) * hS
                wemb_h[b_pu, 0:(index_pair[1] - index_pair[0]), st:ed] \
                    = all_encoder_layer[i_layer][b, index_pair[0]:index_pair[1],:]

    return wemb_h
