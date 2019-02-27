import sentencepiece as spm
import os

if not os.path.exists("bpe_models"):
        os.makedirs("bpe_models")

# 1K 
spm.SentencePieceTrainer.Train('--input=raw/SETIMES.en-ro.ro --model_prefix=setimes.1K.bpe.ro --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=1024')
spm.SentencePieceTrainer.Train('--input=raw/SETIMES.en-ro.en --model_prefix=setimes.1K.bpe.en --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=1024')

# 8K 
spm.SentencePieceTrainer.Train('--input=raw/SETIMES.en-ro.ro --model_prefix=setimes.8K.bpe.ro --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=8096')
spm.SentencePieceTrainer.Train('--input=raw/SETIMES.en-ro.en --model_prefix=setimes.8K.bpe.en --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=8096')

#16K
spm.SentencePieceTrainer.Train('--input=raw/SETIMES.en-ro.ro --model_prefix=setimes.16K.bpe.ro --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=16192')
spm.SentencePieceTrainer.Train('--input=raw/SETIMES.en-ro.en --model_prefix=setimes.16K.bpe.en --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=16192')
