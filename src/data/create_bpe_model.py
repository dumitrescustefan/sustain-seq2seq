import sentencepiece as spm

#spm.SentencePieceTrainer.Train('--input=corpus.1M.txt --model_prefix=m --vocab_size=1000 --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true')

#spm.SentencePieceTrainer.Train('--input=Z_news.2015.en.shuffled --model_prefix=news.1K.ws.bpe --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true  --input_sentence_size=1000000 --shuffle_input_sentence=true --vocab_size=1024')

# CNNDM
# 1K 
spm.SentencePieceTrainer.Train('--input=cnndm.txt --model_prefix=cnndm.1K.bpe --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=1024')

# 8K 
spm.SentencePieceTrainer.Train('--input=cnndm.txt --model_prefix=cnndm.8K.bpe --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=8096')

#32K
spm.SentencePieceTrainer.Train('--input=cnndm.txt --model_prefix=cnndm.32K.bpe --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=32768')

# NEWS SHUFFLED
# 1K 
spm.SentencePieceTrainer.Train('--input=Z_news.2015.en.shuffled --model_prefix=news.1K.bpe --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=1024')

# 8K 
spm.SentencePieceTrainer.Train('--input=Z_news.2015.en.shuffled --model_prefix=news.4K.bpe --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=8096')

#32K
spm.SentencePieceTrainer.Train('--input=Z_news.2015.en.shuffled --model_prefix=news.32K.bpe --unk_surface=<UNK> --character_coverage=1.0 --pad_id=0 --pad_piece=<PAD> --unk_id=1 --unk_piece=<UNK> --bos_id=2 --bos_piece=<BOS> --eos_id=3 --eos_piece=<EOS> --model_type=bpe --num_threads=8 --split_by_whitespace=true --shuffle_input_sentence=true --vocab_size=32768')


#--user_defined_symbols=!,\",&,@,#,_,-