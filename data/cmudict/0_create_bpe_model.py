import sentencepiece as spm
import os, json, sys

if not os.path.exists("bpe_models"):
        os.makedirs("bpe_models")

# 2K 
with open(os.path.join("raw","cmudict-0.7b")) as f:
    content = f.readlines()
content = [x.strip() for x in content]
X_text = []
y_text = []
for line in content:
    if line.startswith(";;;"):
        continue
    parts = line.split()
    X_text.append(parts[0])
    y_text.append(parts[1:])
    #print(X_text[-1])
    #print(y_text[-1])

with open(os.path.join("raw","X.txt"),"w",encoding="utf8") as f:
    for line in X_text:
        f.write(line.strip()+"\n")
        
with open(os.path.join("raw","y.txt"),"w",encoding="utf8") as f:
    for line in y_text:
        text = ""
        
        f.write(" ".join(line).strip()+"\n")

# just for x    
spm.SentencePieceTrainer.Train('--input=raw/X.txt --model_prefix=bpe_models/cmudict.128.bpe.X --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=128')
spm.SentencePieceTrainer.Train('--input=raw/X.txt --model_prefix=bpe_models/cmudict.1024.bpe.X --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=1024')
spm.SentencePieceTrainer.Train('--input=raw/X.txt --model_prefix=bpe_models/cmudict.4096.bpe.X --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=4096')

# generate w2i/i2w for y
with open(os.path.join("raw","cmudict-0.7b.symbols")) as f:
    content = f.readlines()
symbols = [x.strip() for x in content]

word2index = {}
index2word = {}
for index, word in enumerate(symbols):
    word2index[word] = index+4
    index2word[str(index+4)] = word
        
# just to be safe, overwrite special markers
word2index['<PAD>'] = 0
word2index['<UNK>'] = 1
word2index['<BOS>'] = 2
word2index['<EOS>'] = 3
index2word['0']='<PAD>'
index2word['1']='<UNK>'
index2word['2']='<BOS>'
index2word['3']='<EOS>'

json.dump(word2index, open(os.path.join("ready", "y_word2index.json"),"w",encoding="utf-8"), sort_keys=True)
json.dump(index2word, open(os.path.join("ready", "y_index2word.json"),"w",encoding="utf-8"), sort_keys=True)

