#NLI models


##ANLI

cd anli/src/

conda activate LawngNLI2
python training_allenai_longformer-base-4096.py --model_class_name "allenai/longformer-base-4096" -n 1 -g 1 --single_gpu -nr 0 --max_length 2048 --gradient_accumulation_steps 64 --eval_frequency 0 --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 256  --train_data snli_train:none,mnli_train:none,fever_train:none,anli_r1_train:none,anli_r2_train:none,anli_r3_train:none  --train_weights 1,1,1,10,20,10  --eval_data snli_dev:none,mnli_m_dev:none,mnli_mm_dev:none,anli_r1_dev:none,anli_r2_dev:none,anli_r3_dev:none  --experiment_name "$model_name|snli+mnli+fnli+r1*10+r2*20+r3*10|nli" --seed 42 --fp16
conda activate LawngNLI

python training_google_bigbird-roberta-base.py --model_class_name "google/bigbird-roberta-base" -n 1 -g 1 --single_gpu -nr 0 --max_length 2048 --gradient_accumulation_steps 128 --eval_frequency 0 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 256  --train_data snli_train:none,mnli_train:none,fever_train:none,anli_r1_train:none,anli_r2_train:none,anli_r3_train:none  --train_weights 1,1,1,10,20,10  --eval_data snli_dev:none,mnli_m_dev:none,mnli_mm_dev:none,anli_r1_dev:none,anli_r2_dev:none,anli_r3_dev:none  --experiment_name "$model_name|snli+mnli+fnli+r1*10+r2*20+r3*10|nli" --seed 42 --fp16

python training_zlucia_custom-legalbert.py --model_class_name "zlucia/custom-legalbert" -n 1 -g 1 --single_gpu -nr 0 --max_length 156 --gradient_accumulation_steps 8 --eval_frequency 0 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 256  --train_data snli_train:none,mnli_train:none,fever_train:none,anli_r1_train:none,anli_r2_train:none,anli_r3_train:none  --train_weights 1,1,1,10,20,10  --eval_data snli_dev:none,mnli_m_dev:none,mnli_mm_dev:none,anli_r1_dev:none,anli_r2_dev:none,anli_r3_dev:none  --experiment_name "$model_name|snli+mnli+fnli+r1*10+r2*20+r3*10|nli" --seed 42 --fp16

python training_nlpaueb_legal-bert-base-uncased.py --model_class_name "nlpaueb/legal-bert-base-uncased" -n 1 -g 1 --single_gpu -nr 0 --max_length 156 --gradient_accumulation_steps 8 --eval_frequency 0 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 256  --train_data snli_train:none,mnli_train:none,fever_train:none,anli_r1_train:none,anli_r2_train:none,anli_r3_train:none  --train_weights 1,1,1,10,20,10  --eval_data snli_dev:none,mnli_m_dev:none,mnli_mm_dev:none,anli_r1_dev:none,anli_r2_dev:none,anli_r3_dev:none  --experiment_name "$model_name|snli+mnli+fnli+r1*10+r2*20+r3*10|nli" --seed 42 --fp16

cd ../..


##DocNLI

cd DocNLI/Code/DocNLI/

for model_name in 'allenai/longformer-base-4096' 'google/bigbird-roberta-base'
do
	python train_docNLI_Longformer_storeModel.py --model $model_name --task_name rte --do_train --num_train_epochs 5 --train_batch_size 32 --gradient_accumulation_steps 4 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 2048 --seed 42 --fp16
done

for model_name in 'zlucia/custom-legalbert' 'nlpaueb/legal-bert-base-uncased' 'roberta-large'
do
	python train_docNLI_Longformer_storeModel.py --model $model_name --task_name rte --do_train --do_lower_case --num_train_epochs 5 --train_batch_size 32 --gradient_accumulation_steps 4 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 512 --seed 42 --fp16
done

for model_name in 'facebook/bart-large'
do
	python train_docNLI_Longformer_storeModel2.py --model $model_name --task_name rte --do_train --do_lower_case --num_train_epochs 5 --train_batch_size 32 --gradient_accumulation_steps 32 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 1024 --seed 42 --fp16
done

for model_name in 'albert-xxlarge-v2'
do
	python train_docNLI_Longformer_storeModel3.py --model $model_name --task_name rte --do_train --do_lower_case --num_train_epochs 5 --train_batch_size 32 --gradient_accumulation_steps 8 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 512 --seed 42 --fp16
done

cd ../../..


##ConTRoL-dataset

cd ConTRoL-dataset/basline/src/

for model_name in 'allenai/longformer-base-4096'
do
	python training.py --model_class_name "$model_name" -n 1 -g 1 --single_gpu -nr 0 --max_length 2048 --epochs 10 --learning_rate 2e-5 --gradient_accumulation_steps 4 --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 16 --save_prediction --train_data ConTRoL_train:none --train_weights 1 --eval_data ConTRoL_dev:none --eval_frequency 0 --experiment_name  "$model_name|ConTRoL" --seed 42 --fp16 --transfer True
done

for model_name in 'google/bigbird-roberta-base'
do
	python training2.py --model_class_name "$model_name" -n 1 -g 1 --single_gpu -nr 0 --max_length 2048 --epochs 10 --learning_rate 2e-5 --gradient_accumulation_steps 16 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 16 --save_prediction --train_data ConTRoL_train:none --train_weights 1 --eval_data ConTRoL_dev:none --eval_frequency 0 --experiment_name  "$model_name|ConTRoL" --seed 42 --transfer True
done

for model_name in 'zlucia/custom-legalbert' 'nlpaueb/legal-bert-base-uncased'
do
	python training3.py --model_class_name "$model_name" -n 1 -g 1 --single_gpu -nr 0 --max_length 512 --epochs 10 --learning_rate 2e-5 --gradient_accumulation_steps 2 --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 16 --save_prediction --train_data ConTRoL_train:none --train_weights 1 --eval_data ConTRoL_dev:none --eval_frequency 0 --experiment_name  "$model_name|ConTRoL" --seed 42 --fp16 --transfer True
done

for model_name in 'ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli'
do
	python training4.py --model_class_name "$model_name" -n 1 -g 1 --single_gpu -nr 0 --max_length 1024 --epochs 10 --learning_rate 2e-5 --gradient_accumulation_steps 16 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 16 --save_prediction --train_data ConTRoL_train:none --train_weights 1 --eval_data ConTRoL_dev:none --eval_frequency 0 --experiment_name  "$model_name|ConTRoL" --seed 42
done

for model_name in 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
do
	python training.py --model_class_name "$model_name" -n 1 -g 1 --single_gpu -nr 0 --max_length 512 --epochs 10 --learning_rate 2e-5 --gradient_accumulation_steps 16 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 16 --save_prediction --train_data ConTRoL_train:none --train_weights 1 --eval_data ConTRoL_dev:none --eval_frequency 0 --experiment_name  "$model_name|ConTRoL" --seed 42
done

for model_name in 'ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli'
do
	python training5.py --model_class_name "$model_name" -n 1 -g 1 --single_gpu -nr 0 --max_length 512 --epochs 10 --learning_rate 2e-5 --gradient_accumulation_steps 16 --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 16 --save_prediction --train_data ConTRoL_train:none --train_weights 1 --eval_data ConTRoL_dev:none --eval_frequency 0 --experiment_name  "$model_name|ConTRoL" --seed 42 --fp16
done

cd ../../..


##LawngNLI

cd Nystromformer
epoch=2

##may require
##sed -i 's|attn_probs_only_global.transpose(1, 2), value_vectors_only_global.transpose(1, 2)|attn_probs_only_global.transpose(1, 2).clone(), value_vectors_only_global.transpose(1, 2).clone()|' transformers/models/longformer/modeling_longformer.py

for d in 'LawngNLI_hypothesis_only' 'LawngNLI_short_premise_only' 'LawngNLI_short_premise_filtered_BM25_only' 'LawngNLI_long_premise_filtered_BM25_only' 'LawngNLI_long_premise' 'LawngNLI_short_premise' 'LawngNLI_long_premise_filtered_BM25' 'LawngNLI_short_premise_filtered_BM25'
do
	mkdir glue/mnli/$d
	cp ../"$d".tsv glue/mnli/$d/train.tsv
	cp ../"$d"_val.tsv glue/mnli/$d/val.tsv
	for dataset in 'anli' 'ConTRoL-dataset' 'vanilla'
	do
	for model_name in 'allenai_longformer-base-4096' 'google_bigbird-roberta-base'
	do 
	cd "$model_name"
	python ../code/run_glue3.py --dataset "$d" --model_name "$model_name" --batch_size 32 --lr 1e-5 --epoch $((2*${epoch})) --task mnli --checkpoint "$dataset"
	cd ..
	done
	for model_name in 'zlucia_custom-legalbert' 'nlpaueb_legal-bert-base-uncased' 'facebook_bart-large' 'albert-xxlarge-v2' 'roberta-large'
	do 
	cd "$model_name"
	python ../code/run_glue3.py --dataset "$d" --model_name "$model_name" --batch_size 32 --lr 1e-5 --epoch $((2*${epoch})) --task mnli --checkpoint "$dataset"
	cd ..
	done
	done
	for dataset in 'DocNLI'
	do
	for model_name in 'allenai_longformer-base-4096'
	do 
	cd "$model_name"
	python ../code/run_glue5.py --dataset "$d" --model_name "$model_name" --batch_size 32 --lr 1e-5 --epoch $epoch --task mnli --checkpoint "$dataset"
	cd ..
	done
	for model_name in 'google_bigbird-roberta-base'
	do 
	cd "$model_name"
	python ../code/run_glue5.py --dataset "$d" --model_name "$model_name" --batch_size 32 --lr 1e-5 --epoch $epoch --task mnli --checkpoint "$dataset"
	cd ..
	done
	for model_name in 'zlucia_custom-legalbert' 'nlpaueb_legal-bert-base-uncased' 'facebook_bart-large' 'albert-xxlarge-v2' 'roberta-large'
	do 
	cd "$model_name"
	python ../code/run_glue5.py --dataset "$d" --model_name "$model_name" --batch_size 32 --lr 1e-5 --epoch $epoch --task mnli --checkpoint "$dataset"
	cd ..
	done
	done
done

cd ..

##

cd Nystromformer
cp '../anli/saved_models/allenai_longformer-base-4096_model.pt' 'allenai_longformer-base-4096/model/model/cp-anli.model'
cp '../anli/saved_models/google_bigbird-roberta-base_model.pt' 'google_bigbird-roberta-base/model/model/cp-anli.model'
cp '../anli/saved_models/zlucia_custom-legalbert_model.pt' 'zlucia_custom-legalbert/model/model/cp-anli.model'
cp '../anli/saved_models/nlpaueb_legal-bert-base-uncased_model.pt' 'nlpaueb_legal-bert-base-uncased/model/model/cp-anli.model'
wget https://huggingface.co/ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'facebook_bart-large/model/model/cp-anli.model'
wget https://huggingface.co/ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'albert-xxlarge-v2/model/model/cp-anli.model'
wget https://huggingface.co/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'roberta-large/model/model/cp-anli.model'
wget https://huggingface.co/allenai/longformer-base-4096/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'allenai_longformer-base-4096/model/model/cp-vanilla.model'
wget https://huggingface.co/google/bigbird-roberta-base/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'google_bigbird-roberta-base/model/model/cp-vanilla.model'
wget https://huggingface.co/zlucia/custom-legalbert/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'zlucia_custom-legalbert/model/model/cp-vanilla.model'
wget https://huggingface.co/nlpaueb/legal-bert-base-uncased/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'nlpaueb_legal-bert-base-uncased/model/model/cp-vanilla.model'
wget https://huggingface.co/facebook/bart-large/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'facebook_bart-large/model/model/cp-vanilla.model'
wget https://huggingface.co/albert-xxlarge-v2/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'albert-xxlarge-v2/model/model/cp-vanilla.model'
wget https://huggingface.co/roberta-large/resolve/main/pytorch_model.bin
mv 'pytorch_model.bin' 'roberta-large/model/model/cp-vanilla.model'
cp '../DocNLI/Code/DocNLI/DocNLI_dataset/allenai_longformer-base-4096_docNLI_Longformer_epoch_2.pt' 'allenai_longformer-base-4096/model/model/cp-DocNLI.model'
cp '../DocNLI/Code/DocNLI/DocNLI_dataset/google_bigbird-roberta-base_docNLI_Longformer_epoch_2.pt' 'google_bigbird-roberta-base/model/model/cp-DocNLI.model'
cp '../DocNLI/Code/DocNLI/DocNLI_dataset/zlucia_custom-legalbert_docNLI_Longformer_epoch_2.pt' 'zlucia_custom-legalbert/model/model/cp-DocNLI.model'
cp '../DocNLI/Code/DocNLI/DocNLI_dataset/nlpaueb_legal-bert-base-uncased_docNLI_Longformer_epoch_2.pt' 'nlpaueb_legal-bert-base-uncased/model/model/cp-DocNLI.model'
cp '../DocNLI/Code/DocNLI/DocNLI_dataset/facebook_bart-large_docNLI_Longformer_epoch_2.pt' 'facebook_bart-large/model/model/cp-DocNLI.model'
cp '../DocNLI/Code/DocNLI/DocNLI_dataset/albert-xxlarge-v2_docNLI_Longformer_epoch_2.pt' 'albert-xxlarge-v2/model/model/cp-DocNLI.model'
cp '../DocNLI/Code/DocNLI/DocNLI_dataset/roberta-large_docNLI_Longformer_epoch_2.pt' 'roberta-large/model/model/cp-DocNLI.model'
cp '../ConTRoL-dataset/basline/saved_models/allenai_longformer-base-4096_model.pt' 'allenai_longformer-base-4096/model/model/cp-ConTRoL-dataset.model'
cp '../ConTRoL-dataset/basline/saved_models/google_bigbird-roberta-base_model.pt' 'google_bigbird-roberta-base/model/model/cp-ConTRoL-dataset.model'
cp '../ConTRoL-dataset/basline/saved_models/zlucia_custom-legalbert_model.pt' 'zlucia_custom-legalbert/model/model/cp-ConTRoL-dataset.model'
cp '../ConTRoL-dataset/basline/saved_models/nlpaueb_legal-bert-base-uncased_model.pt' 'nlpaueb_legal-bert-base-uncased/model/model/cp-ConTRoL-dataset.model'
cp '../ConTRoL-dataset/basline/saved_models/facebook_bart-large_model.pt' 'facebook_bart-large/model/model/cp-ConTRoL-dataset.model'
cp '../ConTRoL-dataset/basline/saved_models/albert-xxlarge-v2_model.pt' 'albert-xxlarge-v2/model/model/cp-ConTRoL-dataset.model'
cp '../ConTRoL-dataset/basline/saved_models/roberta-large_model.pt' 'roberta-large/model/model/cp-ConTRoL-dataset.model'
cd ..

#Retrieval model

cd anli/src

python training_albert.py --learning_rate 1e-6 --model_class_name 'albert-xxlarge-v2' -n 1 -g 1 --single_gpu -nr 0 --max_length 512 --gradient_accumulation_steps 4 --eval_frequency 0 --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 16  --train_data LawngNLI_retrieval_train:none  --train_weights 1  --eval_data LawngNLI_retrieval_dev:none  --experiment_name "albert-xxlarge-v2|LawngNLI_retrieval" --seed 42 --fp16

cd ../..