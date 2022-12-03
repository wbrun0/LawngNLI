import csv
import sys
import pandas as pd
from nltk.tokenize import word_tokenize

results=pd.DataFrame()
negation=['no','not','never','none','nobody','nothing','neither','nor','cannot']
nli_label2index = {
    'e': 0,
    'n': 1,
    'c': 2,
    'h': -1,
}

d='ContractNLI'

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pandas as pd
import numpy as np
import nltk, lzma, ujson, re, random, torch, statistics
from anli.src.utils import common
nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained('roberta-large')

with open("contract-nli/train.json") as f:
	data=ujson.load(f)
data=pd.DataFrame([(a["text"],data["labels"][b]["hypothesis"],a["annotation_sets"][0]["annotations"][b]["choice"]) for a in data["documents"] for b in data["labels"].keys()], columns=["premise","hypothesis","label"])
data["label"]=data["label"].map({"Entailment":0,"NotMentioned":1,"Contradiction":2})
input = tokenizer(data['premise'].tolist())
l=[len(input[a]) for a in range(data.shape[0])]
results.at["Percentiles of premises' token lengths [10, 25, 50, 75, 90]",d]=str(np.percentile(l, [10, 25, 50, 75, 90]))
input = tokenizer(data['hypothesis'].tolist())
l=[len(input[a]) for a in range(data.shape[0])]
results.at["Average of hypotheses' token lengths",d]=np.mean(l)
results.at["Proportion of hypotheses containing a negation phrase",d]=[round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==0,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==1,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==2,'hypothesis']]),3)]
results.at["Number of unique training premises",d]=len(data["premise"].unique())
results.at["Number of unique training hypotheses",d]=len(data["hypothesis"].unique())
results.at["Number of training examples",d]=len(l)
results.at["Text sources",d]='Contracts'

d='anli'

data=[]
for data2 in ["anli/data/build/snli/train.jsonl","anli/data/build/mnli/train.jsonl","anli/data/build/fever_nli/train.jsonl"]:
	data+=common.load_jsonl(data2)
for data2 in ["anli/data/build/anli/r1/train.jsonl","anli/data/build/anli/r2/train.jsonl","anli/data/build/anli/r2/train.jsonl", "anli/data/build/anli/r3/train.jsonl"]:
	data+=common.load_jsonl(data2)*10
data=pd.DataFrame(data)
input = tokenizer(data['premise'].tolist())
l=[len(input[a]) for a in range(data.shape[0])]
results.at["Percentiles of premises' token lengths [10, 25, 50, 75, 90]",d]=str(np.percentile(l, [10, 25, 50, 75, 90]))
data['label']=data['label'].map(nli_label2index)
input = tokenizer(data['hypothesis'].tolist())
l=[len(input[a]) for a in range(data.shape[0])]
results.at["Average of hypotheses' token lengths",d]=np.mean(l)
results.at["Proportion of hypotheses containing a negation phrase",d]=[round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==0,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==1,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==2,'hypothesis']]),3)]
results.at["Number of unique training premises",d]=len(data["premise"].unique())
results.at["Number of unique training hypotheses",d]=len(data["hypothesis"].unique())
results.at["Number of training examples",d]=len(l)
results.at["Text sources",d]='Wikipedia, news, etc. SNLI, MNLI, NLI version of FEVER'
data['premise']=data['premise'].str.replace('\n','@n@')
data['hypothesis']=data['hypothesis'].str.replace('\n','@n@')
data.reset_index(drop=True)[['label']*7+['premise','hypothesis','label']].to_csv(d+'.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
d='DocNLI'

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pandas as pd
import numpy as np
import json_lines
import codecs
import random
import json
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('roberta-large')

readfile = codecs.open('DocNLI/Code/DocNLI/DocNLI_dataset/train.json', 'r', 'utf-8')
data = json.load(readfile)
examples = []
for a in data:
	premise = a.get('premise')
	hypothesis = a.get('hypothesis')
	label = a.get('label')
	examples+=[[premise,hypothesis,label]]
data=pd.DataFrame(examples, columns=['premise','hypothesis','label'])
data['label']=data['label'].map({'entailment': 0,'not_entailment': 1})
input = tokenizer(data['premise'].tolist())
l=[len(input[a]) for a in range(data.shape[0])]
results.at["Percentiles of premises' token lengths [10, 25, 50, 75, 90]",d]=str(np.percentile(l, [10, 25, 50, 75, 90]))
input = tokenizer(data['hypothesis'].tolist())
l=[len(input[a]) for a in range(data.shape[0])]
results.at["Average of hypotheses' token lengths",d]=np.mean(l)
results.at["Proportion of hypotheses containing a negation phrase",d]=[round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==0,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==1,'hypothesis']]),3)]
results.at["Number of unique training premises",d]=len(data["premise"].unique())
results.at["Number of unique training hypotheses",d]=len(data["hypothesis"].unique())
results.at["Number of training examples",d]=len(l)
results.at["Text sources",d]='ANLI, SQuAD, CNN/DailyMail, DUC (2001), Curation'
data['premise']=data['premise'].str.replace('\n','@n@')
data['hypothesis']=data['hypothesis'].str.replace('\n','@n@')
data.reset_index(drop=True)[['label']*7+['premise','hypothesis','label']].to_csv(d+'.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
d='ConTRoL-dataset'

sys.path.append('./ConTRoL-dataset/basline/src/')

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from utils import common
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('roberta-large')

data=pd.DataFrame(common.load_jsonl('ConTRoL-dataset/basline/data/train.jsonl'))
input = tokenizer(data['premise'].tolist())
l=[len(input[a]) for a in range(data.shape[0])]
results.at["Percentiles of premises' token lengths [10, 25, 50, 75, 90]",d]=str(np.percentile(l, [10, 25, 50, 75, 90]))
data['label']=data['label'].map(nli_label2index)
input = tokenizer(data['hypothesis'].tolist())
l=[len(input[a]) for a in range(data.shape[0])]
results.at["Average of hypotheses' token lengths",d]=np.mean(l)
results.at["Proportion of hypotheses containing a negation phrase",d]=[round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==0,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==1,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in data.loc[data['label']==2,'hypothesis']]),3)]
results.at["Number of unique training premises",d]=len(data["premise"].unique())
results.at["Number of unique training hypotheses",d]=len(data["hypothesis"].unique())
results.at["Number of training examples",d]=len(l)
results.at["Text sources",d]='Various civil service exams'
data['premise']=data['premise'].str.replace('\n','@n@')
data['hypothesis']=data['hypothesis'].str.replace('\n','@n@')
data.reset_index(drop=True)[['label']*7+['premise','hypothesis','label']].to_csv(d+'.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pandas as pd
import numpy as np
import nltk, lzma, ujson, re, random, torch, statistics
nltk.download('punkt')

tokenizer = AutoTokenizer.from_pretrained('roberta-large')

d='LawngNLI_long_premise'

data=pd.read_pickle('LawngNLI.xz')
data=data[data['split']=='train']	
data=data[['short_premise','long_premise','hypothesis','label','cited_pages','sample','split','negation']]	
data.columns=['short_premise','premise','hypothesis','label','cited_pages','sample','split','negation']	
for sample in [0,1,2]:
	d+=str(sample); temp=data.copy().reset_index(drop=True)
	if sample<2:
			temp=temp[temp['sample']==sample].copy().reset_index(drop=True)
	input = tokenizer(temp['premise'].tolist())
	l=[len(input[a]) for a in range(temp.shape[0])]
	results.at["Percentiles of premises' token lengths [10, 25, 50, 75, 90]",d]=str(np.percentile(l, [10, 25, 50, 75, 90]))
	input = tokenizer(temp['hypothesis'].tolist())
	l=[len(input[a]) for a in range(temp.shape[0])]
	results.at["Average of hypotheses' token lengths",d]=np.mean(l)
	results.at["Proportion of hypotheses containing a negation phrase",d]=[round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in temp.loc[temp['label']==0,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in temp.loc[temp['label']==1,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in temp.loc[temp['label']==2,'hypothesis']]),3)]
	results.at["Number of unique training premises",d]=len(temp["premise"].unique())
	results.at["Number of unique training hypotheses",d]=len(temp["hypothesis"].unique())
	results.at["Number of training examples",d]=len(l)
	results.at["Text sources",d]='Legal case opinions'
d=d[:-3]
data=data[data['sample']==1]	
input = tokenizer(data['premise'].tolist())
from gensim.summarization.bm25 import BM25
data['_premise']=['@n@'.join([c[l] for l in np.sort(np.argpartition(BM25([h.split() for h in c]).get_scores(b.split()),-min(len(c),5))[-min(len(c),5):]).tolist()]) for a, b in zip(data['premise'],data['hypothesis']) for c in [a.split('\n')]]
data['premise'] = ['\n'.join(c[-l:]+c) for a, b in zip(data['premise'],input['input_ids']) for c in [a.split('\n')] for l in [sum([(h==50118) for h in b[-513:]])+1]]
data['premise']=data['premise'].str.replace('\n','@n@')
data['hypothesis']=data['hypothesis'].str.replace('\n','@n@')
data=data.reset_index(drop=True)
data[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data.rename({'premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_filtered_BM25.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data['temp']=' '
data.rename({'hypothesis':'l','temp':'hypothesis'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_only.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data.rename({'hypothesis':'l','temp':'hypothesis','premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_filtered_BM25_only.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data['premise']=data['premise'].str.replace('@n@','\n')
data['hypothesis']=data['hypothesis'].str.replace('@n@','\n')
data['premise']=[a+'\n'+('\n').join([c for c in b.split('\n') if c.strip() not in h]) for a, b in zip(data['short_premise'],data['premise']) for h in [[q.strip() for q in a.split('\n')]]]
data['pages']=[((int(max(pages))-int(min(pages)))==len(pages)-1) if ((pages!=frozenset()) & (not pd.isnull(pages))) else None for pages in data['cited_pages']]
data['premise']=data['premise'].str.replace('\n','@n@')
data['hypothesis']=data['hypothesis'].str.replace('\n','@n@')
data[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_short.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data=pd.read_pickle('LawngNLI.xz')
data=data[data['split']!='train']	
data=data[['short_premise','long_premise','hypothesis','label','cited_pages','sample','split','negation']]	
data.columns=['short_premise','premise','hypothesis','label','cited_pages','sample','split','negation']	
data=data[data['sample']==1]	
input = tokenizer(data['premise'].tolist())
from gensim.summarization.bm25 import BM25
data['_premise']=['@n@'.join([c[l] for l in np.sort(np.argpartition(BM25([h.split() for h in c]).get_scores(b.split()),-min(len(c),5))[-min(len(c),5):]).tolist()]) for a, b in zip(data['premise'],data['hypothesis']) for c in [a.split('\n')]]
data['premise'] = ['\n'.join(c[-l:]+c) for a, b in zip(data['premise'],input['input_ids']) for c in [a.split('\n')] for l in [sum([(h==50118) for h in b[-513:]])+1]]
data['premise']=data['premise'].str.replace('\n','@n@')
data['hypothesis']=data['hypothesis'].str.replace('\n','@n@')
data=data.reset_index(drop=True)
data[data['split']=='val'][['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
l=[len(input[a]) for a in range(data.shape[0])]
data['length']=[(a>2339) for a in l]
data[data['split']=='test'][['sample']*5+['length','negation','premise','hypothesis','label']].to_csv(d+'_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='val'].rename({'premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_filtered_BM25_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='test'].rename({'premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_filtered_BM25_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data['temp']=' '
data[data['split']=='val'].rename({'hypothesis':'l','temp':'hypothesis'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_only_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='test'].rename({'hypothesis':'l','temp':'hypothesis'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_only_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='val'].rename({'hypothesis':'l','temp':'hypothesis','premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_filtered_BM25_only_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='test'].rename({'hypothesis':'l','temp':'hypothesis','premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_filtered_BM25_only_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data['premise']=data['premise'].str.replace('@n@','\n')
data['hypothesis']=data['hypothesis'].str.replace('@n@','\n')
data['premise']=[a+'\n'+('\n').join([c for c in b.split('\n') if c.strip() not in h]) for a, b in zip(data['short_premise'],data['premise']) for h in [[q.strip() for q in a.split('\n')]]]
data['pages']=[((int(max(pages))-int(min(pages)))==len(pages)-1) if ((pages!=frozenset()) & (not pd.isnull(pages))) else None for pages in data['cited_pages']]
data['premise']=data['premise'].str.replace('\n','@n@')
data['hypothesis']=data['hypothesis'].str.replace('\n','@n@')
data[data['split']=='val'][['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_short_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[(data['split']=='test') & (data['pages']==True)][['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_short1_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[(data['split']=='test') & (data['pages']==False)][['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_short2_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
d='LawngNLI_short_premise'

data=pd.read_pickle('LawngNLI.xz')
data=data[data['split']=='train']	
data=data[['short_premise','long_premise','hypothesis','label','sample','split','negation']]	
data.columns=['premise','long_premise','hypothesis','label','sample','split','negation']	
for sample in [0,1,2]:
	d+=str(sample); temp=data.copy().reset_index(drop=True)
	if sample<2:
			temp=temp[temp['sample']==sample].copy().reset_index(drop=True)
	input = tokenizer(temp['premise'].tolist())
	l=[len(input[a]) for a in range(temp.shape[0])]
	results.at["Percentiles of premises' token lengths [10, 25, 50, 75, 90]",d]=str(np.percentile(l, [10, 25, 50, 75, 90]))
	input = tokenizer(temp['hypothesis'].tolist())
	l=[len(input[a]) for a in range(temp.shape[0])]
	results.at["Average of hypotheses' token lengths",d]=np.mean(l)
	results.at["Proportion of hypotheses containing a negation phrase",d]=[round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in temp.loc[temp['label']==0,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in temp.loc[temp['label']==1,'hypothesis']]),3),round(np.mean([any([b in negation for b in word_tokenize(a)]+[("n't" in a)]) for a in temp.loc[temp['label']==2,'hypothesis']]),3)]
	results.at["Number of unique training premises",d]=len(temp["premise"].unique())
	results.at["Number of unique training hypotheses",d]=len(temp["hypothesis"].unique())
	results.at["Number of training examples",d]=len(l)
	results.at["Text sources",d]='Legal case opinions'
d=d[:-3]
data=data[data['sample']==1]	
from gensim.summarization.bm25 import BM25
data['_premise']=['@n@'.join([c[l] for l in np.sort(np.argpartition(BM25([h.split() for h in c]).get_scores(b.split()),-min(len(c),5))[-min(len(c),5):]).tolist()]) for a, b in zip(data['premise'],data['hypothesis']) for c in [a.split('\n')]]
data['premise']=data['premise'].str.replace('\n','@n@')
data['hypothesis']=data['hypothesis'].str.replace('\n','@n@')
data=data.reset_index(drop=True)
data[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data.rename({'premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_filtered_BM25.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data['temp']=' '
data.rename({'hypothesis':'l','temp':'hypothesis'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_only.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data.rename({'hypothesis':'l','temp':'hypothesis','premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_filtered_BM25_only.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
d='LawngNLI_hypothesis_only'
data['premise']=' '
data[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
d='LawngNLI_short_premise'
data=pd.read_pickle('LawngNLI.xz')
data=data[data['split']!='train']	
data=data[['short_premise','long_premise','hypothesis','label','sample','split','negation']]	
data.columns=['premise','long_premise','hypothesis','label','sample','split','negation']	
data=data[data['sample']==1]	
input = tokenizer(data['long_premise'].tolist())
from gensim.summarization.bm25 import BM25
data['_premise']=['@n@'.join([c[l] for l in np.sort(np.argpartition(BM25([h.split() for h in c]).get_scores(b.split()),-min(len(c),5))[-min(len(c),5):]).tolist()]) for a, b in zip(data['premise'],data['hypothesis']) for c in [a.split('\n')]]
data['premise']=data['premise'].str.replace('\n','@n@')
data['hypothesis']=data['hypothesis'].str.replace('\n','@n@')
data=data.reset_index(drop=True)
data[data['split']=='val'][['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
l=[len(input[a]) for a in range(data.shape[0])]
data['length']=[(a>2339) for a in l]
data[data['split']=='test'][['sample']*5+['length','negation','premise','hypothesis','label']].to_csv(d+'_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='val'].rename({'premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_filtered_BM25_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='test'].rename({'premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_filtered_BM25_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data['temp']=' '
data[data['split']=='val'].rename({'hypothesis':'l','temp':'hypothesis'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_only_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='test'].rename({'hypothesis':'l','temp':'hypothesis'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_only_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='val'].rename({'hypothesis':'l','temp':'hypothesis','premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_filtered_BM25_only_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
data[data['split']=='test'].rename({'hypothesis':'l','temp':'hypothesis','premise':'long_premise','_premise':'premise'}, axis='columns')[['sample']*6+['negation','premise','hypothesis','label']].to_csv(d+'_filtered_BM25_only_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
d='LawngNLI_hypothesis_only'
data['premise']=' '
data[data['split']=='val'][['sample']*6+['negation','premise','hypothesis','label']].groupby(['label','negation']).sample(84, random_state=1).sample(500, random_state=0).to_csv(d+'_val.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)
input = tokenizer(data['long_premise'].tolist())
l=[len(input[a]) for a in range(data.shape[0])]
data['length']=[(a>2339) for a in l]
data[data['split']=='test'][['sample']*5+['length','negation','premise','hypothesis','label']].to_csv(d+'_test.tsv',line_terminator='\r',sep='\t', quoting=csv.QUOTE_NONE)

results.to_csv('_stats.csv')
