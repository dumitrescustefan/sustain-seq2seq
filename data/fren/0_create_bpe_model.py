import sentencepiece as spm
import os

if not os.path.exists("bpe_models"):
        os.makedirs("bpe_models")

# 2K 
spm.SentencePieceTrainer.Train('--input=raw/JRC-Acquis.en-fr.fr --model_prefix=bpe_models/jrc-acquis.2K.bpe.fr --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=2048')
spm.SentencePieceTrainer.Train('--input=raw/JRC-Acquis.en-fr.en --model_prefix=bpe_models/jrc-acquis.2K.bpe.en --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=2048')

