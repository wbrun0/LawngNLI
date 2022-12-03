from collections import defaultdict
from collections import OrderedDict
from eyecite import clean_text
from math import ceil
from gensim.summarization.bm25 import BM25
from sentence_transformers import SentenceTransformer, util, InputExample
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pandas as pd
import numpy as np
import lzma, ujson, os, nltk, pickle, multiprocessing, re, gc, random, torch, faiss, statistics, csv, json
nltk.download('punkt')
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
from torch.nn.functional import softmax 
from torch.utils.data import DataLoader
from sentence_transformers.util import batch_to_device

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pd.set_option('display.max_columns',None)

results=pd.DataFrame()
ref={}
for finetune in ['base_']:
	for dataset in ['LawngNLI_long_premise_filtered_BM25']:
		dataset2=dataset
		for retrieval in [0,2]:
			dataset=dataset2
			data=pd.read_pickle('../data_long_BM25_100_'+str(retrieval)+'.xz')
			data2=data.copy()
			data['probability']=[b for a in np.array_split(data,100) for b in BM25([c.split() for c in a['text']]).get_scores(a['hypothesis'].iloc[0].split())]
			data=data.sort_values(['index','probability'],ascending=False)
			data['rank']=data.groupby('index')['label'].transform(lambda a: a.reset_index(drop=True).loc[a.reset_index(drop=True)==1].index.item()+1 if 1 in set(a.tolist()) else 1e10)
			for t in [1,10,100]:
				results.at['BM25',str(retrieval)+'_'+dataset2+'_'+dataset+'_'+str(t)]=str(round(int(len(data)/100)/1322*torch.mean((torch.Tensor(data.loc[(([True]+[False]*99)*int(len(data)/100)),'rank'].to_numpy())<=t).float()).item(),3))+'+/-'+str(round(int(len(data)/100)/1322*max([abs(a-torch.mean((torch.Tensor(data.loc[(([True]+[False]*99)*int(len(data)/100)),'rank'].to_numpy())<=t).float()).item()) for a in proportion_confint(torch.sum((torch.Tensor(data.loc[(([True]+[False]*99)*int(len(data)/100)),'rank'].to_numpy())<=t).float()).item(),(torch.Tensor(data.loc[(([True]+[False]*99)*int(len(data)/100)),'rank'].to_numpy())<=t).shape[0],alpha=0.05,method='beta')]),3))
			for model_name in [
			'all-mpnet-base-v2',
			'albert-xxlarge-v2/model/model/cp-anli-LawngNLI-retrieval.model',
			'cross-encoder/ms-marco-MiniLM-L-6-v2',
			'albert-xxlarge-v2/model/model/cp-anli.model',
			'albert-xxlarge-v2/model/model/cp-DocNLI.model',
			'albert-xxlarge-v2/model/model/cp-ConTRoL-dataset.model',
			]:
				data=data2.copy()
				l=2 if re.search('DocNLI',model_name) else 3
				model_name3=model_name
				model_name=re.sub('/model.*','',model_name)
				if model_name=='cross-encoder/ms-marco-MiniLM-L-6-v2':
					tokenizer = AutoTokenizer.from_pretrained(model_name)
					model = AutoModelForSequenceClassification.from_pretrained(model_name)						
				elif model_name=='all-mpnet-base-v2':
						model_name3=model_name
						model_name=re.sub('/model.*','',model_name)
						model_name2=model_name.replace('_','/')
						model_name=re.sub('\.model','',re.sub('/model/model/cp-','_',model_name3))
						model_name3=re.sub('\.model','-'+dataset+finetune+'_2.model' if not re.search('base_',finetune) else '.model',model_name3)
						model = SentenceTransformer(model_name2) if re.search('base_',finetune) else SentenceTransformer(model_name3)
				else:
					os.chdir(model_name)
					model_name2=model_name.replace('_','/')
					model_name=re.sub('\.model','',re.sub('/model/model/cp-','_',model_name3))
					checkpoint = torch.load('../'+model_name3, map_location='cpu')
					if 'model_state_dict' in checkpoint.keys():
						checkpoint = checkpoint['model_state_dict']
					if re.search('DocNLI',model_name):
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
				if model_name=='all-mpnet-base-v2':
						probability=[]
						if True:
							dataloader = DataLoader(list(zip(data['text'],data['hypothesis'])), shuffle=False, batch_size=100)
							for batch in dataloader:
								A=model.encode(batch[0], convert_to_numpy=True)
								B=model.encode(batch[1], convert_to_numpy=True)
								probability += [util.dot_score(a,b) for a, b in zip(A,B)]
						data['probability']=[b if not (a.isspace() or a=='') else -np.inf for a, b in zip(data['text'],torch.Tensor(probability).tolist())]
				else:
					for hypothesis in ['hypothesis']:
						for premise in ['text']: 
									data[premise+'_2']=data[premise].copy()
									for hypothesis_only in ['premise_included']:
										batch_size=24
										probability=[]
										a=data.iloc[0:batch_size]
										c=0
										temp=[]
										while a.empty is False:
											input = tokenizer(a[premise+'_2'].tolist() if hypothesis_only=='premise_included' else ['']*(len(a[hypothesis].tolist())), a[hypothesis].tolist(), padding=True, truncation=True, return_tensors="pt")
											input.to(device)
											with torch.no_grad():
												outputs = model(**input)
											if model_name=='cross-encoder/ms-marco-MiniLM-L-6-v2':
												probability += outputs["logits"].cpu()													
											else:
												probability += outputs["logits"][...,0 if re.search('DocNLI',model_name) else retrieval].cpu()					
											c+=1
											a=data.iloc[c*batch_size:(c+1)*batch_size]
										data['probability']=probability
				data=data.sort_values(['index','probability'],ascending=False)
				data['rank']=data.groupby('index')['label'].transform(lambda a: a.reset_index(drop=True).loc[a.reset_index(drop=True)==1].index.item()+1 if 1 in set(a.tolist()) else 1e10)
				for t in [1,10,100]:
					results.at[model_name,str(retrieval)+'_'+dataset2+'_'+dataset+'_'+str(t)]=str(round(int(len(data)/100)/1322*torch.mean((torch.Tensor(data.loc[(([True]+[False]*99)*int(len(data)/100)),'rank'].to_numpy())<=t).float()).item(),3))+'+/-'+str(round(int(len(data)/100)/1322*max([abs(a-torch.mean((torch.Tensor(data.loc[(([True]+[False]*99)*int(len(data)/100)),'rank'].to_numpy())<=t).float()).item()) for a in proportion_confint(torch.sum((torch.Tensor(data.loc[(([True]+[False]*99)*int(len(data)/100)),'rank'].to_numpy())<=t).float()).item(),(torch.Tensor(data.loc[(([True]+[False]*99)*int(len(data)/100)),'rank'].to_numpy())<=t).shape[0],alpha=0.05,method='beta')]),3))
					results.at['N',str(retrieval)+'_'+dataset2+'_'+dataset+'_'+str(t)]=len(data.loc[(([True]+[False]*99)*int(len(data)/100))])
				print(results)
results = results.reindex(sorted(results.columns), axis=1)
results.to_csv('_all_test_retrieval.csv',mode='a')