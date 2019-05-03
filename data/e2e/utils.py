import os, sys, json

def create_w2i_i2w_from_bpe_model(bpe_model_vocab):    
    print("Creating word2index and index2word ...")
    word2index = {}
    index2word = {}

    index = -1

    with open(bpe_model_vocab,"r",encoding="utf8") as f:    
        for line in f:
            index+=1
            word = line.split("\t")[0]
            word2index[word] = index
            index2word[str(index)] = word

    # just to be safe, overwrite special markers
    word2index['<PAD>'] = 0
    word2index['<UNK>'] = 1
    word2index['<BOS>'] = 2
    word2index['<EOS>'] = 3
    index2word['0']='<PAD>'
    index2word['1']='<UNK>'
    index2word['2']='<BOS>'
    index2word['3']='<EOS>'

    #json.dump(word2index, open(os.path.join(arg["output_folder"],"word2index.json"),"w",encoding="utf-8"), sort_keys=True)
    #json.dump(index2word, open(os.path.join(arg["output_folder"],"index2word.json"),"w",encoding="utf-8"), sort_keys=True) 
    print("Vocab size : {}".format(len(word2index)))
    return word2index, index2word
    