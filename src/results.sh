git clone https://github.com/facebookresearch/anli.git
cd anli/
git reset --hard 4a37569
cd ..
cd anli/
source setup.sh
cd $DIR_TMP
bash script/download_data.sh
python src/dataset_tools/build_data.py

cd ..
cd anli/src/
mv nli/training.py training.py

sed -i "s|RobertaTokenizer|AutoTokenizer.from_pretrained('@model_name')|" training.py
sed -i "s|RobertaForSequenceClassification|AutoModelForSequenceClassification.from_pretrained('@model_name', gradient_checkpointing=True)|" training.py
sed -i "s|AutoTokenizer.from_pretrained('@model_name'), AutoModelForSequenceClassification.from_pretrained('@model_name', gradient_checkpointing=True)|AutoTokenizer, AutoModelForSequenceClassification|" training.py
sed -i "s|roberta-large|@model_name|g" training.py
sed -i "s| and args.total_step <= 0| and is_finished|" training.py
mkdir -p data/build/LawngNLI_retrieval/
sed -i 's|snli_|LawngNLI_retrieval_|g' training.py
sed -i 's|snli/|LawngNLI_retrieval/|g' training.py

cp training.py training_allenai_longformer-base-4096.py
sed -i "s|for i in range(len(eval_data_name))|if False|" training_allenai_longformer-base-4096.py
sed -i "s|@model_name|allenai/longformer-base-4096|g" training_allenai_longformer-base-4096.py
cp training.py training_albert.py
sed -i "s/'do_lower_case' in model_class_item else False/'do_lower_case' in model_class_item else True/" training_albert.py
sed -i "s|@model_name|ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli|g" training_albert.py
cp training.py training_google_bigbird-roberta-base.py
sed -i "s|truncation=True|truncation=True, padding=True|" training_google_bigbird-roberta-base.py
sed -i "s|@model_name|google/bigbird-roberta-base|g" training_google_bigbird-roberta-base.py
cp training.py training_zlucia_custom-legalbert.py
sed -i "s/'do_lower_case' in model_class_item else False/'do_lower_case' in model_class_item else True/" training_zlucia_custom-legalbert.py
cp training_zlucia_custom-legalbert.py training_nlpaueb_legal-bert-base-uncased.py
sed -i "s|@model_name|zlucia/custom-legalbert|g" training_zlucia_custom-legalbert.py
sed -i "s|@model_name|nlpaueb/legal-bert-base-uncased|g" training_nlpaueb_legal-bert-base-uncased.py

cd ../..

git clone https://github.com/salesforce/DocNLI.git
cd DocNLI/
git reset --hard 31b3404
cd ..
cd DocNLI/Code/DocNLI/

conda activate LawngNLI3
gdown 16TZBTZcb9laNKxIvgbs5nOBgq3MhND5s
conda activate LawngNLI

unzip DocNLI_dataset.zip
rm -rf DocNLI_dataset.zip

sed -i "s/\/export\/home\/Dataset\/para_entail_datasets\//DocNLI_dataset\//" ../load_data.py
sed -i "s|'/export/home/Dataset/BERT_pretrained_mine/paragraph_entail/2021'|'DocNLI_dataset'|" train_docNLI_Longformer_storeModel.py
sed -i "s|'docNLI_Longformer_epoch_'|pretrain_model_dir.replace('/','_')+'_'+'docNLI_Longformer_epoch_'|" train_docNLI_Longformer_storeModel.py

sed -i "s/from transformers.models.longformer.tokenization_longformer import LongformerTokenizer as RobertaTokenizer/from transformers import AutoTokenizer as RobertaTokenizer/" train_docNLI_Longformer_storeModel.py
sed -i "s/from transformers.models.longformer.modeling_longformer import LongformerModel as RobertaModel/from transformers import AutoModel as RobertaModel/" train_docNLI_Longformer_storeModel.py
sed -i "s/from transformers.optimization import AdamW/from transformers.optimization import AdamW\nfrom transformers import AutoConfig/" train_docNLI_Longformer_storeModel.py
sed -i "s|bert_hidden_dim = 768|bert_hidden_dim = AutoConfig.from_pretrained('allenai/longformer-base-4096').hidden_size|" train_docNLI_Longformer_storeModel.py

sed -i "s/(pretrain_model_dir)/(pretrain_model_dir, gradient_checkpointing=True)/" train_docNLI_Longformer_storeModel.py
sed -i "s|allenai/longformer-base-4096|args.model|" train_docNLI_Longformer_storeModel.py
sed -i 's|to train.")|to train.")\n    parser.add_argument("--task_name",default=None,type=str,required=True)|' train_docNLI_Longformer_storeModel.py
cp train_docNLI_Longformer_storeModel.py train_docNLI_Longformer_storeModel2.py
sed -i "s|outputs_single\[1\]|outputs_single[2][:,0,:]|" train_docNLI_Longformer_storeModel2.py
sed -i "s|input_ids, input_mask, None|input_ids, input_mask, None, output_hidden_states=True|" train_docNLI_Longformer_storeModel2.py
cp train_docNLI_Longformer_storeModel.py train_docNLI_Longformer_storeModel3.py
sed -i "s|pretrain_model_dir, gradient_checkpointing=True|pretrain_model_dir|" train_docNLI_Longformer_storeModel3.py

cd ../../..

git clone https://github.com/csitfun/ConTRoL-dataset.git
cd ConTRoL-dataset/
git reset --hard 2805fa5
cd ..
cd ConTRoL-dataset/basline/src/
mv ../../data ../

sed -i "s|flint.data_utils.fields|data_utils.fields|" data_utils/batchbuilder.py
sed -i "s|RobertaTokenizer|AutoTokenizer.from_pretrained('@model_name')|" training.py
sed -i "s|RobertaForSequenceClassification|AutoModelForSequenceClassification.from_pretrained('@model_name')|" training.py
sed -i "s|AutoTokenizer.from_pretrained('@model_name'), AutoModelForSequenceClassification.from_pretrained('@model_name')|AutoTokenizer, AutoModelForSequenceClassification|" training.py
sed -i "s|AutoModelForSequenceClassification.from_pretrained('@model_name')|AutoModelForSequenceClassification.from_pretrained('@model_name', gradient_checkpointing=True)|" training.py
sed -i "s|roberta-large|@model_name|g" training.py
sed -i 's|"data/build/pnli|"data|' training.py
sed -i "s|'pnli_|'ConTRoL_|" training.py
sed -i "s|saved_models/transfer/checkpoints/eval/model.pt|../../../anli/saved_models/'+args.model_class_name.replace('/','_')+'_model.pt|" training.py
sed -i 's|@model_name|args.model_class_name|g' training.py
cp training.py training2.py
sed -i "s|truncation=True|truncation=True, padding='max_length'|" training2.py
cp training.py training3.py
sed -i "s/'do_lower_case' in model_class_item else False/'do_lower_case' in model_class_item else True/" training3.py
cp training.py training4.py
sed -i 's|\["distilbert", "bart-large"\]|["distilbert", "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"]|' training4.py
cp training3.py training5.py
sed -i "s|'@model_name', gradient_checkpointing=True|'@model_name'|" training5.py

cd ../../..

git clone https://github.com/mlpen/Nystromformer.git
cd Nystromformer
git reset --hard 6539b89
cd ..
cd Nystromformer
mkdir glue/mnli
sed -i 's|            data_loader = DataLoader(train_dataset|            train_dataset.examples=random.sample(train_dataset.examples,len(train_dataset.examples))\n            data_loader = DataLoader(train_dataset|' code/run_glue.py
sed -i 's|import numpy as np|import numpy as np\nimport pandas as pd\nfrom transformers.data.processors.glue import MnliProcessor|' code/run_glue.py 
sed -i 's|shuffle = True|shuffle = False, pin_memory = True|' code/run_glue.py 
sed -i 's|/model/config.json|model/config.json|' code/run_glue.py 
sed -i 's|dev_datasets = .*|dev_datasets = {}|' code/run_glue.py 
sed -i 's|if args.task.lower() == "mnli"|if False|' code/run_glue.py 
sed -i 's|model_config\["num_classes"\] = 2|model_config["num_classes"] = 3|' code/run_glue.py 
sed -i 's|/code/roberta-base|../code/roberta-base|' code/utils.py 
sed -i 's|:04||' code/run_glue.py 
sed -i 's|\.module||' code/run_glue.py 
sed -i "s|\['model_state_dict'\]||" code/run_glue.py 
sed -i 's|\(.*best_scores\[partition_name\], sort_keys = True), flush = True).*\)|\1\nsave_path = os.path.join(checkpoint_dir, f"cp-@d.model")\ntorch.save({"model_state_dict":model.module.state_dict()}, save_path)|' code/run_glue.py
sed -i 's|from torch.utils.data import DataLoader|from torch.utils.data import DataLoader, Dataset|' code/run_glue.py
sed -i 's|GlueDataset(data_args, tokenizer = tokenizer)|Data(MnliProcessor().get_train_examples(data_args.data_dir), tokenizer, data_args)\ncollator = DataCollatorWithPadding(tokenizer = tokenizer, padding = "max_length", max_length = model_config["max_seq_len"])|' code/run_glue.py
sed -i "s|import utils|import utils\nimport sys, csv, random\ncsv.field_size_limit(sys.maxsize)\n\nfrom transformers.trainer_utils import set_seed\nset_seed(42)\nclass Data(Dataset):\n    def __init__(self, examples, tokenizer, data_args):\n        self.len=len(examples)\n        self.examples=examples\n        self.tokenizer=tokenizer\n        self.data_args=data_args\n\n    def __getitem__(self, i):\n        return {**{k: batch_encoding[k][0] for batch_encoding in [self.tokenizer([(example.text_a, example.text_b) for example in [self.examples[i]]],max_length=data_args.max_seq_length,truncation=True)] for k in batch_encoding}, **{'labels': int(self.examples[i].label)}}\n\n    def __len__(self):\n        return self.len|" code/run_glue.py
sed -i 's|collate_fn = default_data_collator|collate_fn = collator|' code/run_glue.py
sed -i 's|import default_data_collator|import DataCollatorWithPadding|' code/run_glue.py
sed -i "s|example.text_a, example.text_b|example.text_a.replace('@n@','\\\n'), example.text_b.replace('@n@','\\\n')|" code/run_glue.py
cp code/run_glue.py code/run_glue2.py
sed -i 's|int(self.examples\[i\].label)|int(float(self.examples[i].label))|' code/run_glue2.py
sed -i 's|"--checkpoint", type = int|"--checkpoint", type = str|' code/run_glue2.py
sed -i 's|f"cp-@d.model"|f"cp-"+args.checkpoint+f"-@d-finetune2-"+str(epoch_idx)+f"-.model"|' code/run_glue2.py
sed -i 's|save_path = |        save_path = |' code/run_glue2.py
sed -i 's|torch.save|        torch.save|' code/run_glue2.py
sed -i "s|if False|val=pd.read_csv(data_args.data_dir+'/val.tsv',lineterminator='\\\r',sep='\\\t', quoting=csv.QUOTE_NONE).iloc[:,8:].replace('@n@','\\\n')\nif False|" code/run_glue2.py
sed -i 's|model_config\["num_classes"\] = 3|model_config["num_classes"] = 2 if "DocNLI" in args.checkpoint else 3|' code/run_glue2.py
sed -i 's|model_config = config\["model"\]|model_config = config["model"]; model_config["max_seq_len"] = 4096|' code/run_glue2.py
for model_name in 'allenai_longformer-base-4096' 'google_bigbird-roberta-base' 'zlucia_custom-legalbert' 'nlpaueb_legal-bert-base-uncased' 'facebook_bart-large' 'albert-xxlarge-v2' 'roberta-large' 'sentence-transformers_all-distilroberta-v1' 'sentence-transformers_all-mpnet-base-v2' 'sentence-transformers_nli-distilroberta-base-v2' 'sentence-transformers_msmarco-distilroberta-base-v2'
do
	mkdir -p "$model_name"/model/model
done

sed -i 's|if batch_idx % 10 == 0:|if batch_idx % 100 == 0:\n                model.eval()\n                with torch.no_grad():\n                    summary["accuracy"] = np.mean(np.asarray(labels[-100*args.batch_size:]) == np.asarray(predictions[-100*args.batch_size:]))\n                    l = val.iloc[:,2].tolist()\n                    a=val.iloc[0:5*gpu_config["inst_per_gpu"]]\n                    c=0\n                    p=[]\n                    while a.empty is False:\n                        input = tokenizer(a.iloc[:,0].tolist(), a.iloc[:,1].tolist(), padding=True, truncation=True, return_tensors="pt")\n                        p.extend(model(**input)["sent_scores"].argmax(-1).cpu().data.tolist())\n                        c+=1\n                        a=val.iloc[c*5*gpu_config["inst_per_gpu"]:(c+1)*5*gpu_config["inst_per_gpu"]]\n                    summary["val_accuracy"] = np.mean(np.asarray(l) == np.asarray(p))\n                model.train()|' code/run_glue2.py
sed -i 's|model.load_state_dict|model.module.load_state_dict|' code/run_glue2.py
sed -i 's|ArgumentParser()|ArgumentParser()\nparser.add_argument("--dataset", type = str, dest = "dataset", required = True)|g' code/run_glue2.py
sed -i 's|@d|args.dataset|g' code/run_glue2.py
sed -i "s|'/val.tsv'|args.dataset+'/val.tsv'|g" code/run_glue2.py
sed -i "s|get_train_examples(data_args.data_dir)|get_train_examples(data_args.data_dir+'/'+args.dataset)|g" code/run_glue2.py
cp code/run_glue2.py code/run_glue3.py
sed -i "s|padding=True|padding='max_length'|" code/run_glue2.py
sed -i 's|parser = argparse.ArgumentParser()|parser = argparse.ArgumentParser()\nparser.add_argument("--model_name", type = str, help = "model_name", dest = "model_name", required = True)|' code/run_glue3.py
sed -i 's|from model_wrapper import ModelForSequenceClassification|from transformers import AutoModelForSequenceClassification|' code/run_glue3.py
sed -i 's|model = ModelForSequenceClassification(model_config)|if args.model_name=="albert-xxlarge-v2":\n    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3)\nelse:\n    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=3, gradient_checkpointing=True)|' code/run_glue3.py
sed -i 's|RobertaTokenizerFast|AutoTokenizer|' code/run_glue3.py
sed -i 's|tokenizer = utils.get_tokenizer(model_config\["max_seq_len"\])|tokenizer = AutoTokenizer.from_pretrained(args.model_name)\ntokenizer.model_max_length = tokenizer.model_max_length if tokenizer.model_max_length<=4096 else 512|' code/run_glue3.py
sed -i 's|model/config.json|../softmax/model/config.json|' code/run_glue3.py
sed -i 's|args.model_name|args.model_name.replace("_","/")|' code/run_glue3.py
sed -i 's|max_length = model_config\["max_seq_len"\]|max_length = tokenizer.model_max_length|' code/run_glue3.py
sed -i 's|max_length=data_args.max_seq_length|max_length = self.tokenizer.model_max_length|' code/run_glue3.py
sed -i 's|del outputs\["sent_scores"\]|outputs={"loss":outputs["loss"]}|' code/run_glue3.py
sed -i 's|sent_scores|logits|' code/run_glue3.py
sed -i 's|join(checkpoint_dir,|join("model/model/",|' code/run_glue3.py
sed -i 's|len(device_ids), gpu_config\["inst_per_gpu"\]|len(device_ids), 1 if args.model_name=="albert-xxlarge-v2" else 2|' code/run_glue3.py
cp code/run_glue2.py code/run_glue4.py
sed -i 's|int(float(self.examples\[i\].label))|0 if int(float(self.examples[i].label))==0 else 1|' code/run_glue4.py
sed -i 's|self.len=len(examples)|self.len=len(examples+[a for a in examples if int(float(a.label))==0])|' code/run_glue4.py
sed -i 's|self.examples=examples|self.examples=examples+[a for a in examples if int(float(a.label))==0]|' code/run_glue4.py
sed -i 's|l = val.iloc\[:,2\].tolist()|l = val.iloc[:,2].replace(2,1).tolist()|' code/run_glue4.py
sed -i 's|num_labels=3|num_labels=2|' code/run_glue4.py
cp code/run_glue3.py code/run_glue5.py
sed -i 's|int(float(self.examples\[i\].label))|0 if int(float(self.examples[i].label))==0 else 1|' code/run_glue5.py
sed -i 's|self.len=len(examples)|self.len=len(examples+[a for a in examples if int(float(a.label))==0])|' code/run_glue5.py
sed -i 's|self.examples=examples|self.examples=examples+[a for a in examples if int(float(a.label))==0]|' code/run_glue5.py
sed -i 's|l = val.iloc\[:,2\].tolist()|l = val.iloc[:,2].replace(2,1).tolist()|' code/run_glue5.py
sed -i 's|num_labels=3|num_labels=2|' code/run_glue5.py
sed -i 's|import json|import json, re\nfrom collections import OrderedDict|' code/run_glue5.py
sed -i 's|load_state_dict(checkpoint|load_state_dict(OrderedDict((a[re.sub("^[^\.]*\.","",k)], v) for a in [{re.sub("^[^\.]*\.","",b): b for b in model.module.state_dict().keys()}] for (k, v) in checkpoint.items() if re.sub("^[^\.]*\.","",k) in a.keys())|' code/run_glue5.py
