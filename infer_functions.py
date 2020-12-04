from matplotlib.pylab import *
from roberta_training import *
from seq2sql_model_testing import *
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import os
re_ = re.compile(' ')

def tokenize_corenlp_direct_version(client, nlu1):
    nlu1_tok = []
    for sentence in client.annotate(nlu1).sentence:
        for tok in sentence.token:
            nlu1_tok.append(tok.originalText)
    return nlu1_tok

def sent_split(documents):
    words = []
    for sent in sent_tokenize(documents):
      for word in word_tokenize(sent):
        words.append(word)
    return words

def load_jsonl(path_file, toy_data=False, toy_size=4, shuffle=False, seed=1):
    data = []

    with open(path_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if toy_data and idx >= toy_size and (not shuffle):
                break
            t1 = json.loads(line.strip())
            data.append(t1)

    if shuffle and toy_data:
        # When shuffle required, get all the data, shuffle, and get the part of data.
        print(
            f"If the toy-data is used, the whole data loaded first and then shuffled before get the first {toy_size} data")

        python_random.Random(seed).shuffle(data)  # fixed
        data = data[:toy_size]

    return data

def sort_and_generate_pr_w(pr_sql_i):
    pr_wc = []
    pr_wo = []
    pr_wv = []
    for b, pr_sql_i1 in enumerate(pr_sql_i):
        conds1 = pr_sql_i1["conds"]
        pr_wc1 = []
        pr_wo1 = []
        pr_wv1 = []

        # Generate
        for i_wn, conds11 in enumerate(conds1):
            pr_wc1.append( conds11[0])
            pr_wo1.append( conds11[1])
            pr_wv1.append( conds11[2])

        # sort based on pr_wc1
        idx = argsort(pr_wc1)
        pr_wc1 = array(pr_wc1)[idx].tolist()
        pr_wo1 = array(pr_wo1)[idx].tolist()
        pr_wv1 = array(pr_wv1)[idx].tolist()

        conds1_sorted = []
        for i, idx1 in enumerate(idx):
            conds1_sorted.append( conds1[idx1] )


        pr_wc.append(pr_wc1)
        pr_wo.append(pr_wo1)
        pr_wv.append(pr_wv1)

        pr_sql_i1['conds'] = conds1_sorted

    return pr_wc, pr_wo, pr_wv, pr_sql_i

def process(data,tokenize):
    final_all = []
    badcase = 0
    for i, one_data in enumerate(data):
        nlu_t1 = one_data["question_tok"]

        # 1. 2nd tokenization using RoBERTa Tokenizer
        charindex2wordindex = {}
        total = 0
        tt_to_t_idx1 = []  # number indicates where sub-token belongs to in 1st-level-tokens (here, CoreNLP).
        t_to_tt_idx1 = []  # orig_to_tok_idx[i] = start index of i-th-1st-level-token in all_tokens.
        nlu_tt1 = []  # all_doc_tokens[ orig_to_tok_idx[i] ] returns first sub-token segement of i-th-1st-level-token
        for (ii, token) in enumerate(nlu_t1):
            t_to_tt_idx1.append(
                len(nlu_tt1))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenize.tokenize(token, is_pretokenized=True)
            for sub_token in sub_tokens:
                tt_to_t_idx1.append(ii)
                nlu_tt1.append(sub_token)  # all_doc_tokens are further tokenized using RoBERTa tokenizer

            token_ = re_.sub('',token)
            for iii in range(len(token_)):
                charindex2wordindex[total+iii]=ii
            total += len(token_)

        one_final = one_data
        # one_table = table[one_data["table_id"]]
        final_question = [0] * len(nlu_tt1)
        one_final["bertindex_knowledge"] = final_question
        final_header = [0] * len(one_data["header"])
        one_final["header_knowledge"] = final_header
        for ii,h in enumerate(one_data["header"]):
            h = h.lower()
            hs = h.split("/")
            for h_ in hs:
                flag, start_, end_ = contains2(re_.sub('', h_), "".join(one_data["question_tok"]).lower())
                if flag == True:
                    try:
                        start = t_to_tt_idx1[charindex2wordindex[start_]]
                        end = t_to_tt_idx1[charindex2wordindex[end_]]
                        for iii in range(start,end):
                            final_question[iii] = 4
                        final_question[start] = 4
                        final_question[end] = 4
                        one_final["bertindex_knowledge"] = final_question
                    except:
                        # print("!!!!!")
                        continue

        for ii,h in enumerate(one_data["header"]):
            h = h.lower()
            hs = h.split("/")
            for h_ in hs:
                flag, start_, end_ = contains2(re_.sub('', h_), "".join(one_data["question_tok"]).lower())
                if flag == True:
                    try:
                        final_header[ii] = 1
                        break
                    except:
                        # print("!!!!")
                        continue

        one_final["header_knowledge"] = final_header

        if "bertindex_knowledge" not in one_final and len(one_final["sql"]["conds"])>0:
            one_final["bertindex_knowledge"] = [0] * len(nlu_tt1)
            badcase+=1

        final_all.append([one_data["question_tok"],one_final["bertindex_knowledge"],one_final["header_knowledge"]])
    return final_all
  

def contains2(small_str,big_str):
    if small_str in big_str:
        start = big_str.index(small_str)
        return True,start,start+len(small_str)-1
    else:
        return False,-1,-1

def infer(nlu1,
          table_id, headers, types, tokenizer, 
          model, model_roberta, roberta_config, max_seq_length, num_target_layers,
          beam_size=4):
   
    model.eval()
    model_roberta.eval()

    # Get inputs
    nlu = [nlu1]

    nlu_t1 = sent_split(nlu1)
    nlu_t = [nlu_t1]
    hds = [headers]
    hs_t = [[]]

    data = {}
    data['question_tok'] = nlu_t[0]
    data['table_id'] = table_id
    data['header'] = headers
    data = [data]

    tb = {}
    tb['id'] = table_id
    tb['header'] = headers
    tb['types'] = types
    tb = [tb]

    tk = tokenizer

    check = process(data, tk)
    knowledge = [check[0][1]]
    header_knowledge = [check[0][2]]

    wemb_n, wemb_h, l_n, l_hpu, l_hs, \
    nlu_tt, t_to_tt_idx, tt_to_t_idx \
        = get_wemb_roberta(roberta_config, model_roberta, tokenizer, nlu_t, hds, max_seq_length,
                        num_out_layers_n=num_target_layers, num_out_layers_h=num_target_layers)

    prob_sca, prob_w, prob_wn_w, pr_sc, pr_sa, pr_wn, pr_sql_i = model.beam_forward(wemb_n, l_n, wemb_h, l_hpu,
                                                                                    l_hs, tb,
                                                                                    nlu_t, nlu_tt,
                                                                                    tt_to_t_idx, nlu,
                                                                                    beam_size=beam_size,
                                                                                    knowledge=knowledge,
                                                                                    knowledge_header=header_knowledge)
    pr_wc, pr_wo, pr_wv, pr_sql_i = sort_and_generate_pr_w(pr_sql_i)
    
    
    if len(pr_sql_i) != 1:
        raise EnvironmentError
    pr_sql_q1 = generate_sql_q(pr_sql_i, tb)
    pr_sql_q = [pr_sql_q1]

    print(f'START ============================================================= ')
    print(f'{hds}')
    print(f'nlu: {nlu}')
    print(f'pr_sql_i : {pr_sql_i}')
    print(f'pr_sql_q : {pr_sql_q}')
    print(f'---------------------------------------------------------------------')

    return pr_sql_i
