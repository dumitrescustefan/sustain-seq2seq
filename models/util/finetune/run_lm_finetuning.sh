rm -rf out
rm data/complex/cached*

python3 lm_finetuning2.py \
--train_data_file data/complex/data.txt \
--eval_data_file data/complex/eval.txt \
--output_dir out \
--model_type gpt2 \
--block_size 200 \
--model_name_or_path gpt2 \
--do_train \
--do_eval \
--evaluate_during_training 	\
--num_train_epochs 3 \
--overwrite_output_dir \
--per_gpu_train_batch_size 1 \
--per_gpu_eval_batch_size 1