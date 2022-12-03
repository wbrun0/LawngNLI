from collections import defaultdict
from collections import OrderedDict
from math import ceil
import pandas as pd
import numpy as np
import lzma, ujson, os, nltk, pickle, multiprocessing, re, gc
nltk.download('punkt')
import random, torch, statistics, sentence_transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, RobertaTokenizerFast
import sys
sys.path.append('./code')
import utils, csv, json, os
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pd.set_option('display.max_columns',None)

results=pd.DataFrame()
ref={}
probability3={}
for dataset in ['LawngNLI_hypothesis_only','LawngNLI_long_premise_only','LawngNLI_long_premise_filtered_BM25_only','LawngNLI_short_premise_only','LawngNLI_short_premise_filtered_BM25_only']:
	dataset2=dataset
	dataset=re.sub('base_','',dataset)
	dataset=re.sub('_negation','',dataset)
	dataset=re.sub('_length','',dataset)
	if re.search('_short_LawngNLI',dataset):
		dataset=re.sub('.*_short_LawngNLI','LawngNLI',dataset2)
		data=pd.read_csv('../'+dataset+'_test.tsv',lineterminator='\r',sep='\t', quoting=csv.QUOTE_NONE).iloc[:,6:]
		dataset=re.sub('_short_LawngNLI.*','',dataset2)
	elif re.search('short_..._LawngNLI',dataset):
		dataset=re.sub('short_..._LawngNLI','LawngNLI',dataset)
		data=pd.read_csv('../'+dataset+'_test.tsv',lineterminator='\r',sep='\t', quoting=csv.QUOTE_NONE).iloc[:,6:]
		dataset='LawngNLI_short_premise'
	else:
		data=pd.read_csv('../'+dataset+'_test.tsv',lineterminator='\r',sep='\t', quoting=csv.QUOTE_NONE).iloc[:,6:]
	if re.search('negation',dataset2):
		data=data[data['negation']=='negation'].copy()
	if re.search('length',dataset2):
		data=data[data['length']==True].copy()
	data['premise']=data['premise'].str.replace('@n@','\n')
	data['hypothesis']=data['hypothesis'].str.replace('@n@','\n')
	for finetune in ['-finetune2-3-']:
		data2=data.copy()
		for model_name in [
		'facebook_bart-large/model/model/cp-vanilla.model',
		'albert-xxlarge-v2/model/model/cp-vanilla.model',
		'facebook_bart-large/model/model/cp-anli.model',
		'albert-xxlarge-v2/model/model/cp-anli.model',
		'facebook_bart-large/model/model/cp-ConTRoL-dataset.model',
		'albert-xxlarge-v2/model/model/cp-ConTRoL-dataset.model',
		]:
			data=data2.copy()
			l=2 if re.search('DocNLI',model_name) else 3
			model_name3=model_name
			model_name=re.sub('/model.*','',model_name)
			os.chdir(model_name)
			try:
				model_name2=model_name.replace('_','/')
				model_name=re.sub('\.model','',re.sub('/model/model/cp-','_',model_name3))
				model_name3=re.sub('\.model','-'+dataset+('-finetune2-1-' if re.search('DocNLI',model_name) else finetune)+'.model' if not re.search('base_',dataset2) else '.model',model_name3)
				checkpoint = torch.load('../'+model_name3, map_location='cpu')
				if 'model_state_dict' in checkpoint.keys():
					checkpoint = checkpoint['model_state_dict']
				if re.search('DocNLI',model_name):
					data.loc[data['label']==2,'label']=1
					data=pd.concat([data[data['label']==0],data[data['label']==0],data[data['label']==1]])
					tokenizer = AutoTokenizer.from_pretrained(model_name2)
					model = AutoModelForSequenceClassification.from_pretrained(model_name2, num_labels=l)
					missing_keys, unexpected_keys = model.load_state_dict(OrderedDict((a[re.sub("^[^\.]*\.","",k)], v) for a in [{re.sub("^[^\.]*\.","",b): b for b in model.state_dict().keys()}] for (k, v) in checkpoint.items() if re.sub("^[^\.]*\.","",k) in a.keys()), strict = False)
					print(f"{missing_keys}")
					print(f"{unexpected_keys}")
					os.chdir('..')
				elif re.search('ConTRoL-dataset',model_name):
					tokenizer = AutoTokenizer.from_pretrained(model_name2)
					model = AutoModelForSequenceClassification.from_pretrained(None, config=AutoConfig.from_pretrained(model_name2, num_labels=l), state_dict=checkpoint)
					os.chdir('..')
				elif re.search('anli',model_name) or re.search('vanilla',model_name):	
					tokenizer = AutoTokenizer.from_pretrained(model_name2)
					model = AutoModelForSequenceClassification.from_pretrained(None, config=AutoConfig.from_pretrained(model_name2, num_labels=l), state_dict=checkpoint)
					os.chdir('..')
				print(model_name)
				model = model.to(device)
				model.eval()
				label=torch.Tensor(data['label'].to_numpy())
				for hypothesis in ['hypothesis']:
					for premise in ['premise']: 
								data[premise+'_2']=data[premise].copy()
								for hypothesis_only in ['premise_included']:
									batch_size=24
									probability=[]
									a=data.iloc[0:batch_size]
									c=0
									temp=[]
									while a.empty is False:
										if re.search('bigbird-roberta-base',model_name) and not re.search('short_..._LawngNLI',dataset2):
											input = tokenizer(a[premise+'_2'].tolist() if hypothesis_only=='premise_included' else ['']*(len(a[hypothesis].tolist())), a[hypothesis].tolist(), padding='max_length', truncation=True, return_tensors="pt")						
										elif re.search('custom-legalbert',model_name) or re.search('legal-bert-base-uncased',model_name) or re.search('short_..._LawngNLI',dataset2):
											input = tokenizer(a[premise+'_2'].tolist() if hypothesis_only=='premise_included' else ['']*(len(a[hypothesis].tolist())), a[hypothesis].tolist(), padding='max_length', truncation=True,  max_length=512, return_tensors="pt")						
										else:
											input = tokenizer(a[premise+'_2'].tolist() if hypothesis_only=='premise_included' else ['']*(len(a[hypothesis].tolist())), a[hypothesis].tolist(), padding=True, truncation=True, return_tensors="pt")
										input.to(device)
										with torch.no_grad():
											outputs = model(**input)
										probability += (outputs["logits"]).argmax(-1).cpu()					
										c+=1
										a=data.iloc[c*batch_size:(c+1)*batch_size]
									print('probability')
									print(probability)
									print('label')
									print(label)
									results.at[model_name,hypothesis_only+'_'+dataset2+'_'+finetune]=str(round(torch.mean(((label)==(torch.Tensor(probability))).float()).item(),3))+'+/-'+str(round(max([abs(a-torch.mean(((label)==(torch.Tensor(probability))).float()).item()) for a in proportion_confint(torch.sum(((label)==(torch.Tensor(probability))).float()).item(),((label)==(torch.Tensor(probability))).shape[0],alpha=0.05,method='beta')]),3))
									results.at['N',hypothesis_only+'_'+dataset2+'_'+finetune]=len(data2)
			except:
				os.chdir('..')
				print(model_name)
	print(results)
results = results.reindex(sorted(results.columns), axis=1)
results.to_csv('_all_test_only_all_test.csv')
