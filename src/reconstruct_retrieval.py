from eyecite import clean_text
from math import ceil
from gensim.summarization.bm25 import BM25
from sentence_transformers import SentenceTransformer, util, InputExample
import pandas as pd
import numpy as np
import lzma, ujson, nltk, pickle, multiprocessing, re, gc, random, torch, faiss, statistics, csv, json
nltk.download('punkt')
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
from torch.nn.functional import softmax 
from torch.utils.data import DataLoader
from sentence_transformers.util import batch_to_device

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pd.set_option('display.max_columns',None)

def reinflate(file_num):
	data2={}
	for h in file_num:
		if h==0:
			i='data.jsonl.xz~'
		else:
			i='data.jsonl.xz~'+str(h)			
		with lzma.open(i) as f:
			for count, line in enumerate(f):
				a=ujson.loads(str(line, 'utf8'))
				data2[a['id']]=b[0] if len(b:=[clean_text(re.sub('<[^>]*>','',re.sub('<page-number label=\"([0-9]*)[^>]*>[^>]*>','@\\1@',c[(first:=c.find('</'))+c[first+1:].find('<p ')+1:].replace('&amp;','&'))), ['inline_whitespace', 'underscores']) for b in [a['casebody']['data']] for c in b[b.find('<opinion type=')+b[b.find('<opinion type='):].find('"')+1:b.find('</casebody>\n')].split('</opinion>\n  <opinion type="') if a['casebody']['status']=='ok' and c[:c.find('"')]=='majority'])==1 else None
	if data2=={}:
		return None
	else:
		data3=pd.DataFrame(pd.Series(data2,name='long_premise_with_citations'))
		data3=data3[~pd.isnull(data3['long_premise_with_citations'])]
		data3=data3[((data3['long_premise_with_citations']!='') & (~data3['long_premise_with_citations'].fillna(value='').str.isspace()))]
		return data3

data=pd.read_pickle('LawngNLI.xz')
data=data[data['split']!='train']	
data=data[['cited_case','short_premise','long_premise','hypothesis','label','cited_pages','sample','split','negation']]	
data.columns=['cited_case','short_premise','text','hypothesis','label','cited_pages','sample','split','negation']	
data=data[data['sample']==1]	
data=data.reset_index(drop=True)
data=data[data['split']=='test'].copy()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data=data.reset_index(drop=False)
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device, cache_folder='./')
temp2=[]
for file_num in [48]+list(range(0,48))+list(range(49,57)):
	print(file_num); temp=reinflate([file_num])
	temp3=temp['long_premise_with_citations'].to_dict()
	lookup=model.encode(temp['long_premise_with_citations'].tolist(), batch_size=100, show_progress_bar=True)			
	if torch.cuda.is_available():
		index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), lookup.shape[1], faiss.GpuIndexFlatConfig())
	else:
		index = faiss.IndexFlatIP(lookup.shape[1])
	index.add(lookup)
	batch_size=800
	hypotheses=data.set_index('cited_case').loc[[(b in a) for a in [set(temp.index)] for b in data['cited_case']],['index','hypothesis','label']].copy()
	a=hypotheses.iloc[0:batch_size]
	c=0
	I=[]
	t=1000
	while a.empty is False:
		query=model.encode(a['hypothesis'].tolist())
		query=np.ascontiguousarray(query, dtype=np.float32)
		_, i = index.search(query, t)
		I+=i.reshape(-1).tolist()
		c+=1
		a=hypotheses.iloc[c*batch_size:(c+1)*batch_size]
	temp=temp.append(pd.Series(name='last'))
	temp.loc['last']=''
	temp=temp.iloc[I]
	hypotheses=pd.DataFrame(np.repeat(hypotheses.values, t, axis=0), columns=hypotheses.columns, index=np.repeat(hypotheses.index.values, t, axis=0))
	hypotheses['nli']=hypotheses['label'].copy()
	hypotheses['label']=(temp.index==hypotheses.index).astype(int)
	temp=pd.concat([temp.reset_index(drop=True),hypotheses.reset_index(drop=True)],axis=1)
	temp['retrieved']=temp.groupby('index')['label'].transform(sum)
	temp.loc[(temp['retrieved']==0) & (([False]*999+[True])*int(len(temp)/1000)),'long_premise_with_citations']=pd.Series(hypotheses.index).astype(int).loc[(temp['retrieved']==0) & (([False]*999+[True])*int(len(temp)/1000))].map(temp3).tolist()
	temp.loc[(temp['retrieved']==0) & (([False]*999+[True])*int(len(temp)/1000)),'label']=1
	temp=temp.set_index('index',drop=True)
	temp2+=[temp]
temp2=pd.concat(temp2).rename({'long_premise_with_citations':'text'},axis=1).reset_index(drop=False)
temp2['text']=temp2['text'].fillna(value='')
print(round(temp2.loc[temp2['retrieved']==1,'label'].sum()/(len(temp)/1000),3))
temp2.to_pickle('data_long.xz')
temp2['text']=['\n'.join([c[l] for l in np.sort(np.argpartition(BM25([h.split() for h in c]).get_scores(b.split()),-min(len(c),5))[-min(len(c),5):]).tolist()]) if not (a.isspace() or a=='') else a for a, b in zip(temp2['text'],temp2['hypothesis']) for c in [a.split('\n')]]
temp2.to_pickle('data_long_BM25.xz')

for split in ['train','dev']:
	data=pd.read_pickle('LawngNLI.xz')
	data=data[['cited_case','short_premise','long_premise','hypothesis','label','cited_pages','sample','split','negation']]	
	data.columns=['cited_case','short_premise','text','hypothesis','label','cited_pages','sample','split','negation']	
	data=data[data['sample']==1]	
	data=data.reset_index(drop=True)
	data=data[data['split']==('train' if split=='train' else 'val')].copy()
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	data=data.reset_index(drop=False)
	data2=data.rename({'long_premise_with_citations':'text'},axis=1).copy()
	data=data[data['label']==1].copy()
	model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device, cache_folder='./')
	temp2=[]
	for file_num in [48]+list(range(0,48))+list(range(49,57)):
		print(file_num); temp=reinflate([file_num])
		temp3=temp['long_premise_with_citations'].to_dict()
		lookup=model.encode(temp['long_premise_with_citations'].tolist(), batch_size=100, show_progress_bar=True)			
		if torch.cuda.is_available():
			index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), lookup.shape[1], faiss.GpuIndexFlatConfig())
		else:
			index = faiss.IndexFlatIP(lookup.shape[1])
		index.add(lookup)
		batch_size=800
		hypotheses=data.set_index('cited_case').loc[[(b in a) for a in [set(temp.index)] for b in data['cited_case']],['index','hypothesis','label']].copy()
		a=hypotheses.iloc[0:batch_size]
		c=0
		I=[]
		t=10
		while a.empty is False:
			query=model.encode(a['hypothesis'].tolist())
			query=np.ascontiguousarray(query, dtype=np.float32)
			_, i = index.search(query, t)
			I+=i.reshape(-1).tolist()
			c+=1
			a=hypotheses.iloc[c*batch_size:(c+1)*batch_size]
		temp=temp.append(pd.Series(name='last'))
		temp.loc['last']=''
		temp=temp.iloc[I]
		hypotheses=pd.DataFrame(np.repeat(hypotheses.values, t, axis=0), columns=hypotheses.columns, index=np.repeat(hypotheses.index.values, t, axis=0))
		hypotheses['min']=np.array(list(range(t))*int(len(hypotheses)/t))+(temp.index==hypotheses.index).astype(int)*t
		temp=pd.concat([temp.reset_index(drop=True),hypotheses.reset_index(drop=True)],axis=1)
		temp['c']=[a//t for a in list(range(len(temp)))]
		temp=temp[temp['min']==temp.groupby('c')['min'].transform(min)].copy()
		temp=temp.set_index('index',drop=True)
		temp2+=[temp]
	temp2=pd.concat(temp2).rename({'long_premise_with_citations':'text'},axis=1).reset_index(drop=False)
	temp2['text']=temp2['text'].fillna(value='')
	data2.loc[data2['label']==1,'text']=temp2['text'].tolist()
	data2['text']=['\n'.join([c[l] for l in np.sort(np.argpartition(BM25([h.split() for h in c]).get_scores(b.split()),-min(len(c),5))[-min(len(c),5):]).tolist()]) if not (a.isspace() or a=='') else a for a, b in zip(data2['text'],data2['hypothesis']) for c in [a.split('\n')]]
	data2=data2.reset_index()[['level_0','text','hypothesis','label']].copy()
	data2['label']=data2['label'].map({0:'e',1:'n',2:'c'})
	data2.columns=['uid','premise','hypothesis','label']
	data2.to_json(split+'.jsonl',orient='records',lines=True)