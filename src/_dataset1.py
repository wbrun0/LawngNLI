from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pattern.en import conjugate

from eyecite import clean_text, get_citations, resolve_citations
from eyecite.helpers import add_defendant
from eyecite.tokenizers import default_tokenizer as tokenizer
from collections import defaultdict
from math import ceil
import pandas as pd
import numpy as np
import lzma, ujson, argparse, os, nltk, pickle, multiprocessing, re, gc, random
nltk.download('punkt')
	
def rangef(pincite):
	if pincite is not None:
		e=pincite.replace("at ","").replace("§","").replace("*","").replace("pp","p").replace("p.","").replace(" ","").split(",")
		e=",".join([",".join([str(h) for h in range(int(g.split("-")[0]),int(g.split("-")[1] if int(g.split("-")[0])<int(g.split("-")[1]) else g.split("-")[0][:-len(g.split("-")[1])]+g.split("-")[1])+1)]) if g.find("-")>=0 else g for g in e]).split(",")
		return frozenset(e)
	else:
		return frozenset()
	
def citation_span(cite, text):
	if cite.__class__.__name__=='SupraCitation' or cite.__class__.__name__=='ShortCaseCitation':
		words, citation_tokens = tokenizer.tokenize(text)
		add_defendant(cite, words)
	a='span'
	a_val=0
	after=text[cite.span()[1]+1:]
	#find last metadata field
	for b in ['parenthetical','pin_cite','year','extra']:
		if (b!='pin_cite' or (cite.__class__.__name__=='FullCaseCitation' and hasattr(cite.metadata,'pin_cite'))) and (b!='year' or hasattr(cite.metadata,'year')) and (b!='extra' or hasattr(cite.metadata,'extra')):
			if a_val<(candidate_val:=(after.find(getattr(cite.metadata,b))+len(getattr(cite.metadata,b))+(b=='parenthetical')+(b=='year')+1 if getattr(cite.metadata,b) is not None else 0)):
				a=b
				a_val=candidate_val
	c=[max(min(((text[:cite.span()[0]].rindex(cite.metadata.plaintiff) if cite.metadata.plaintiff!=cite.metadata.defendant else text[:text[:cite.span()[0]].rindex(cite.metadata.plaintiff)].rindex(cite.metadata.plaintiff)) if cite.metadata.plaintiff!=None else cite.span()[0]) if hasattr(cite.metadata, 'plaintiff') else cite.span()[0],
	(text[:cite.span()[0]].rindex(cite.metadata.antecedent_guess) if cite.metadata.antecedent_guess!=None else cite.span()[0]) if hasattr(cite.metadata, 'antecedent_guess') else cite.span()[0])
	-1,0),cite.span()[1]+a_val]
	d=text[:c[0]].removesuffix(', e.g.,').removesuffix(' e.g.,').removesuffix(', e.g.').removesuffix(' e.g.')
	#https://www.law.cornell.edu/citation/6-300
	for e in [' E.g.',' accord',' Accord',' see also',' See also',' see generally',' See generally',' see',' See',' but cf.',' But cf.',' cf.',' Cf.',' contra',' Contra',' but see',' But see']:
		d=d.removesuffix(e)
	#subset of https://www.thesaurus.com/e/grammar/when-do-you-use-punctuation-marks/
	if text[len(d)-1] in ['.','?','!',';',',']:
		c[0]=len(d)
	else:
		#party names are part of sentence
		c[0]=max(cite.span()[0]-1,0)
	return c

def negate2(h, w):
	if (h._.labels[0]=='ADJP' if len(h._.labels)==1 else False):
		n=[k for k, m in enumerate(h) if (m.tag_ in ['JJ','VBN']) or ((m.tag_=='') & (len(m.text)>1))]
		if len(n)==0:
			return ['@ERROR@']
		else:
			n=min(n)
			if (h[n-1].text in ['no','not','never'] if n>=1 else False):
				return list(h[:n-1])+list(h[n:])
			elif (h[n+1].text in ['no','not','never'] if n+1<=len(h)-1 else False):
				return list(h[:n+1])+list(h[n+2:])
			else:	
				return list(h[:n])+["not "]+list(h[n:])
	n=[k for k, m in enumerate(h) if m.tag_ in ['MD','VBP','VBZ','VBD','VBN','VBG']]
	if len(n)==0:
		return ['@ERROR@']
	else:
		n=min(n)
		if h[n].text=="can" and (h[n+1].text=="not" if n+1<=len(h)-1 else False):
			return list(h[:n+1])+[' ']+list(h[n+2:])
		elif (h[n-1].text in ['no','not','never'] if n>=1 else False):
			return list(h[:n-1])+list(h[n:])
		elif (h[n+1].text in ['no','not','never'] if n+1<=len(h)-1 else False):
			if h[n].text in ["do","does","did"] and (h[n+2].tag_=='VB' if n+2<=len(h)-1 else False):
				return list(h[:n])+[conjugate(h[n+2].text,'pl' if h[n].text=="do" else ('3sg' if h[n].text=="does" else 'p'))]+['' if h[n+2].text==h[n+2].text_with_ws else ' ']+list(h[n+3:])
			else:
				return list(h[:n+1])+list(h[n+2:])
		elif h[n].tag_=='MD' or (h[n].text in ['am','is','are','was','were']) or (h[n].text in ['has','have','had'] and (h[n+1].tag_=='VBN' if n+1<=len(h)-1 else False)) or (h[n].text in ["do","does","did"] and (h[n+1].tag_=='VB' if n+1<=len(h)-1 else False)):
			return list(h[:n])+[h[n].text]+[" not"]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
		elif h[n].tag_=='VBP':
			return list(h[:n])+["do not "+w.lemmatize(h[n].text,'v')]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
		elif h[n].tag_=='VBZ':
			return list(h[:n])+["does not "+w.lemmatize(h[n].text,'v')]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
		elif h[n].tag_=='VBD':
			return list(h[:n])+["did not "+w.lemmatize(h[n].text,'v')]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
		elif h[n].tag_=='VBN' or h[n].tag_=='VBG':
			return list(h[:n])+["not "+h[n].text]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
		return ['@ERROR@']

#inverting the adjectives
def negate3(h, w):
	if (h._.labels[0]=='ADJP' if len(h._.labels)==1 else False):
		n=[k for k, m in enumerate(h) if (m.tag_ in ['JJ','VBN']) or ((m.tag_=='') & (len(m.text)>1))]
		if len(n)==0:
			return ['@ERROR@']
		else:
			n=min(n)
			invert2=invert[invert['POS']=='a']
			if (h[n-1].text in ['no','not','never'] if n>=1 else False):
				return list(h[:n-1])+list(h[n:])
			elif (h[n+1].text in ['no','not','never'] if n+1<=len(h)-1 else False):
				return list(h[:n+1])+list(h[n+2:])
			elif h[n].text in invert2['pos_element'].values:
				return list(h[:n])+[invert2[invert2['pos_element']==h[n].text]['neg_element'].values[0]]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
			elif h[n].text in invert2['neg_element'].values:
				return list(h[:n])+[invert2[invert2['neg_element']==h[n].text]['pos_element'].values[0]]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
			else:	
				return list(h[:n])+["not "]+list(h[n:])
	n=[k for k, m in enumerate(h) if m.tag_ in ['MD','VBP','VBZ','VBD','VBN','VBG']]
	if len(n)==0:
		return ['@ERROR@']
	else:
		n=min(n)
		if h[n].tag_=='MD' or (h[n].text in ['am','is','are','was','were']):
			invert2=invert[invert['POS']=='a']
			if ((h[n+1].tag_=='JJ' and h[n+1].text in invert2['pos_element'].values) if n+1<=len(h)-1 else False):
				return list(h[:n+1])+[invert2[invert2['pos_element']==h[n+1].text]['neg_element'].values[0]]+['' if h[n+1].text==h[n+1].text_with_ws else ' ']+list(h[n+2:])
			elif ((h[n+1].tag_=='JJ' and h[n+1].text in invert2['neg_element'].values) if n+1<=len(h)-1 else False):
				return list(h[:n+1])+[invert2[invert2['neg_element']==h[n+1].text]['pos_element'].values[0]]+['' if h[n+1].text==h[n+1].text_with_ws else ' ']+list(h[n+2:])
			else:	
				if (h[n+1].text in ['no','not','never'] if n+1<=len(h)-1 else False):
					if h[n].text in ["do","does","did"] and (h[n+2].tag_=='VB' if n+2<=len(h)-1 else False):
						return list(h[:n])+[conjugate(h[n+2].text,'pl' if h[n].text=="do" else ('3sg' if h[n].text=="does" else 'p'))]+['' if h[n+2].text==h[n+2].text_with_ws else ' ']+list(h[n+3:])
					elif h[n].text=="can" and (h[n+1].text=="not" if n+1<=len(h)-1 else False):
						return list(h[:n+1])+[' ']+list(h[n+2:])
					else:
						return list(h[:n+1])+list(h[n+2:])
				else:		
					return list(h[:n+1])+["not"]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
		else:
			return negate2(h, w)

#inverting the verbs as well as the adjectives
def negate4(h, w):
	if (h._.labels[0]=='ADJP' if len(h._.labels)==1 else False):
		n=[k for k, m in enumerate(h) if (m.tag_ in ['JJ','VBN']) or ((m.tag_=='') & (len(m.text)>1))]
		if len(n)==0:
			return ['@ERROR@']
		else:
			n=min(n)
			invert2=invert[invert['POS']=='a']
			if (h[n-1].text in ['no','not','never'] if n>=1 else False):
				return list(h[:n-1])+list(h[n:])
			elif (h[n+1].text in ['no','not','never'] if n+1<=len(h)-1 else False):
				return list(h[:n+1])+list(h[n+2:])
			elif h[n].text in invert2['pos_element'].values:
				return list(h[:n])+[invert2[invert2['pos_element']==h[n].text]['neg_element'].values[0]]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
			elif h[n].text in invert2['neg_element'].values:
				return list(h[:n])+[invert2[invert2['neg_element']==h[n].text]['pos_element'].values[0]]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
			else:	
				return list(h[:n])+["not "]+list(h[n:])
	n=[k for k, m in enumerate(h) if m.tag_ in ['MD','VBP','VBZ','VBD','VBN','VBG']]
	if len(n)==0:
		return ['@ERROR@']
	else:
		n=min(n)
		if h[n].tag_=='MD' or (h[n].text in ['am','is','are','was','were']):
			invert2=invert[invert['POS']=='a']
			if ((h[n+1].tag_=='JJ' and h[n+1].text in invert2['pos_element'].values) if n+1<=len(h)-1 else False):
				return list(h[:n+1])+[invert2[invert2['pos_element']==h[n+1].text]['neg_element'].values[0]]+['' if h[n+1].text==h[n+1].text_with_ws else ' ']+list(h[n+2:])
			elif ((h[n+1].tag_=='JJ' and h[n+1].text in invert2['neg_element'].values) if n+1<=len(h)-1 else False):
				return list(h[:n+1])+[invert2[invert2['neg_element']==h[n+1].text]['pos_element'].values[0]]+['' if h[n+1].text==h[n+1].text_with_ws else ' ']+list(h[n+2:])
			else:			
				if (h[n+1].text in ['no','not','never'] if n+1<=len(h)-1 else False):
					if h[n].text in ["do","does","did"] and (h[n+2].tag_=='VB' if n+2<=len(h)-1 else False):
						return list(h[:n])+[conjugate(h[n+2].text,'pl' if h[n].text=="do" else ('3sg' if h[n].text=="does" else 'p'))]+['' if h[n+2].text==h[n+2].text_with_ws else ' ']+list(h[n+3:])
					elif h[n].text=="can" and (h[n+1].text=="not" if n+1<=len(h)-1 else False):
						return list(h[:n+1])+[' ']+list(h[n+2:])
					else:
						return list(h[:n+1])+list(h[n+2:])
				else:		
					return list(h[:n+1])+["not"]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
		inverted=''
		if h[n].tag_[:2]=='VB' and h[n].text not in ['has','have','had']:
			inverted2=''
			invert2=invert[invert['POS']=='v']
			if w.lemmatize(h[n].text,'v') in invert2['pos_element'].values:
				inverted2=invert2[invert2['pos_element']==w.lemmatize(h[n].text,'v')]['neg_element'].values[0]
			elif w.lemmatize(h[n].text,'v') in invert2['neg_element'].values:
				inverted2=invert2[invert2['neg_element']==w.lemmatize(h[n].text,'v')]['pos_element'].values[0]
			if inverted2!='':
				if h[n].tag_=='VBP':
					inverted=conjugate(inverted2, 'pl')
				elif h[n].tag_=='VBZ':
					inverted=conjugate(inverted2, '3sg')
				elif h[n].tag_=='VBD':
					inverted=conjugate(inverted2, 'p')
				elif h[n].tag_=='VBN':
					inverted=conjugate(inverted2, 'ppart')
				elif h[n].tag_=='VBG':
					inverted=conjugate(inverted2, 'part')
		if inverted!='':
			return list(h[:n])+[inverted]+['' if h[n].text==h[n].text_with_ws else ' ']+list(h[n+1:])
		else:
			return negate2(h, w)

def contradict(b, w, nlp):
	c=b
	l=['^']
	l1=''
	l2=''
	while True:
		d=list(c._.children)
		f=[j for j, e in enumerate(d) if (e._.labels[0]=='VP' if len(e._.labels)==1 else False)]
		f2=[j for j, e in enumerate(d) if (e._.labels[0]=='NP' if len(e._.labels)==1 else False)]	
		f3=[j for j, e in enumerate(d) if (e._.labels[0]=='ADJP' if len(e._.labels)==1 else False)]	
		if len(f2)==1:
			#NP
			g=d[:f2[0]]
			h=d[f2[0]]
			if h[0].text in ['no','not','never','No','Not','Never']:
				return '@ERROR@'
			l1=('').join([a if isinstance(a, str) else a.text_with_ws for a in l+list(g)+list(h)])
			i=h.text_with_ws.replace("no one", "someone", 1) 
			if i!=h.text_with_ws and l2=='':
				l2=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+list(i)])
			i=h.text_with_ws.replace("No one", "Someone", 1) 
			if i!=h.text_with_ws and l2=='':
				l2=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+list(i)])
			r=pd.DataFrame([['none','some'],['nobody','somebody'],['nothing','something'],['any','none'],['anyone','no one'],['anybody','nobody'],['anything','nothing'],['some','none'],['someone','no one'],['somebody','nobody'],['something','nothing'],['None','Some'],['Nobody','Somebody'],['Nothing','Something'],['Any','None'],['Anyone','No one'],['Anybody','Nobody'],['Anything','Nothing'],['Some','None'],['Someone','No one'],['Somebody','Nobody'],['Something','Nothing']])
			f4=[j for j, o in enumerate(h) if o.text in r.iloc[:,0]]
			if len(f4)>0 and l2=='':
				f4=min(f4)						
				i=list(h[:f4])+[r.at[r.iloc[:,0]==h[f4],1][0]]+list(h[f4+1:])
				l2=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+list(i)])
			r=pd.DataFrame([['neither','either'],['Neither','Either']])
			f4=[j for j, o in enumerate(h) if o.text in r.iloc[:,0]]
			if len(f4)>0 and l2=='':
				f4=min(f4)						
				i=list(h[:f4])+[r.at[r.iloc[:,0]==h[f4],1][0]]+list(h[f4+1])+list(h[f4+2].text_with_ws.replace(' nor ',' or '))+list(h[f4+3:])
				l2=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+list(i)])
			r=pd.DataFrame([['either','neither'],['Either','Neither']])
			f4=[j for j, o in enumerate(h) if o.text in r.iloc[:,0]]
			if len(f4)>0 and l2=='':
				f4=min(f4)						
				i=list(h[:f4])+[r.at[r.iloc[:,0]==h[f4],1][0]]+list(h[f4+1])+list(h[f4+2].text_with_ws.replace(' or ',' nor '))+list(h[f4+3:])
				l2=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+list(i)])
			#l3=l2
			#l4=l2
			if len(f)==1 and l2=='':
				#VP
				g=d[:f[0]]
				h=d[f[0]]
				if d[f[0]-1]._.labels in [(),('ADVP')] and f[0]>=1:
					g=d[:f[0]-1]
					h=list(nlp(('').join([a if isinstance(a, str) else a.text_with_ws for a in list(d[f[0]-1])+list(h)])).sents)[0]
				l1=('').join([a if isinstance(a, str) else a.text_with_ws for a in l+list(g)+list(h)])
				l2=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+negate2(h, w)]).replace('can not','cannot')
				#l3=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+negate3(h, w)]).replace('can not','cannot')
				#l4=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+negate4(h, w)]).replace('can not','cannot')
			if len(f3)==1 and l2=='':
				#ADJP
				g=d[:f3[0]]
				h=d[f3[0]]
				if d[f3[0]-1]._.labels in [(),('ADVP')] and f3[0]>=1:
					g=d[:f3[0]-1]
					h=list(nlp(('').join([a if isinstance(a, str) else a.text_with_ws for a in list(d[f3[0]-1])+list(h)])).sents)[0]
				l1=('').join([a if isinstance(a, str) else a.text_with_ws for a in l+list(g)+list(h)])
				l2=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+negate2(h, w)]).replace('can not','cannot')
				#l3=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+negate3(h, w)]).replace('can not','cannot')
				#l4=('').join([a if isinstance(a, str) else a.text_with_ws for a in l[1:]+list(g)+negate4(h, w)]).replace('can not','cannot')
		if l2!='':
			return ('^'+b.text_with_ws).replace(l1,l2)
		else:
			f=[j for j, e in enumerate(d) if (e._.labels[0]=='S' if len(e._.labels)==1 else False)]
			if len(f)>0:
				c=random.choice(f)
				l+=d[:c]
				c=d[c]
			else:
				return '@ERROR@'

def keep(b):
	for w in ['plaintiff','defendant','petitioner','respondent','appellant','appellee']:
		b=re.sub('[, \-]*'+w+'[s]*[,\-]*','',b)
	return re.sub(',$','',b)

def func(exclude, data3, k, v):
	def test():
		try:
			conjugate('try','past')
		except:
			pass
	test()
	w=WordNetLemmatizer() 
	random.seed(k)
	#citedid keeps only citations referring to a single case, and conditional statement excludes citations to footnotes, etc.
	with open('citations.pkl', 'rb') as f:
	    citations = pickle.load(f)
	#https://www.law.cornell.edu/citation/4-200
	temp=[[case['case'], case['stance'], citedid := (citation[0] if len(pd.Series(citation:=citations.get(resource.citation.corrected_citation().replace(",","")), dtype=object))==1 else None), pagerange := ((rangef(cite.metadata.pin_cite) if (not re.search('[a-zA-Z]',cite.metadata.pin_cite)) and ('¶' not in cite.metadata.pin_cite) and (':' not in cite.metadata.pin_cite) and ('86-8790115759' not in cite.metadata.pin_cite) else rangef(None)) if cite.metadata.pin_cite is not None else None), re.sub("@[0-9]*@", "", cite.metadata.parenthetical),(sum([(case['text'][:c[0]].endswith(', '+a+',') | case['text'][c[1]+1:].startswith(', '+a+',')) for c in [citation_span(cite,case['text'])] for a in ["acq.","aff'd","aff'g","cert.","juris.","mem.","nonacq.","prob.","reh'g","rev'd","rev'g"]])>0)] if cite.metadata.parenthetical is not None else [case['case'], (citation[0] if len(pd.Series(citation:=citations.get(resource.citation.corrected_citation().replace(",","")), dtype=object))==1 else None)] for index, case in v.iterrows() for resource, cites in resolve_citations(get_citations(case['text'], remove_ambiguous=True)).items() for cite in cites]
	b=dict()
	for (key, value) in [a if len(a)==2 else (a[0],a[2]) for a in temp]:
		if value!=None:
			if key not in b.keys():
				b[key]=[]
			b[key].append(value)			
	exclude.update(b)
	del citations
	import spacy, benepar
	benepar.download('benepar_en3')
	nlp = spacy.load("en_core_web_lg")
	nlp.add_pipe("benepar", config={"model": "benepar_en3"})
	data3[k]=[c[0:4]+[e]+c[5:]+[(' ').join([contradict(b, w, nlp) if i==random.randint(0,len(a)-1) else b.text for a in [list(nlp(re.sub('^[a-z]*ing that ','',e)).sents)] for i, b in enumerate(a)])] for c in temp for e in [(' ').join([''.join([a.text_with_ws if a.orth_.lower()==a.norm_ or a.is_punct else (' '+a.norm_.title() if a.is_title else ' '+a.norm_)+a.whitespace_ for a in h]).replace('  ',' ').replace('can not','cannot').replace(" \([‘’'][sd][^a-zA-Z0-9]\)","\1").strip() for h in list(nlp(re.sub('^[a-z]*ing that ','',c[4])).sents)]) if len(c)>2 else ''] if len(c)>2]
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--cpu", type = int, dest = "cpu", required = True)
	args = parser.parse_args()
	multiprocessing.set_start_method('forkserver')
	num_processes=min(multiprocessing.cpu_count(),args.cpu)
	pool = multiprocessing.Pool(num_processes, maxtasksperchild=1)
	data3 = multiprocessing.Manager().dict()
	exclude = multiprocessing.Manager().dict()

	if not os.path.isfile('citations.pkl'):
		citations = defaultdict(list)
		with lzma.open('data.jsonl.xz~') as f:
			cite=[[b['cite'],ujson.loads(str(line, 'utf8'))['id']] for line in f for b in ujson.loads(str(line, 'utf8'))['citations']]
			for (key, value) in cite:
				citations[key].append(value)	
		for h in range(1,57):
			h='data.jsonl.xz~'+str(h)
			with lzma.open(h) as f:
				cite=[[b['cite'],ujson.loads(str(line, 'utf8'))['id']] for line in f for b in ujson.loads(str(line, 'utf8'))['citations']]
				for (key, value) in cite:
					citations[key].append(value)			
		citations=dict(citations)
		with open('citations.pkl', 'wb') as f:
			pickle.dump(citations, f)

	with open('keep.csv', 'w'):
	    pass
	with open('unfiltered-LawngNLI2.csv', 'w'):
	    pass		

	for h in range(0,57):
		if h==0:
			i='data.jsonl.xz~'
		else:
			i='data.jsonl.xz~'+str(h)			
		with lzma.open(i) as f:
			data3.clear()	
			data2=[]
			c=0
			for j, line in enumerate(f):
				if ((j % (num_processes*10000))==0) & (j!=0):
					data2+=[ujson.loads(str(line, 'utf8'))]
					pd.DataFrame([[a['id'],[keep(b) for b in a['name'].lower().replace(' v. ','|').replace(' vs. ','|').replace(' against ','|').split('|')],a['decision_date']] for a in data2 if a['casebody']['status']=='ok']).to_csv('keep.csv',index=False,header=False,mode='a')
					data2=pd.DataFrame([[a['id'],c[:c.find('"')],c[(first:=c.find('</'))+c[first+1:].find('<p ')+1:]] for a in data2 for b in [a['casebody']['data']] for c in b[b.find('<opinion type=')+b[b.find('<opinion type='):].find('"')+1:b.find('</casebody>\n')].split('</opinion>\n  <opinion type="') if a['casebody']['status']=='ok'], columns=['case', 'stance', 'text'])
					data2['text']=[clean_text(re.sub('<[^>]*>','',re.sub('<page-number label=\"([0-9]*)[^>]*>[^>]*>','@\\1@',text.replace('&amp;','&'))), ['inline_whitespace', 'underscores']) for text in data2['text']]
					w=pool.starmap(func, [(exclude, data3, k, v) for k, v in enumerate(np.array_split(data2,10*num_processes), 10000*num_processes*h+100*num_processes*c)], chunksize=1)
					pd.DataFrame([b for data4 in [dict(data3).copy()] for k in sorted(data4.keys()) for b in data4[k]]).to_csv('unfiltered-LawngNLI2.csv',index=False,header=False,mode='a')
					del data2
					gc.collect()
					c+=1
					data2=[]
					data3.clear()
				else:
					data2+=[ujson.loads(str(line, 'utf8'))]
		if data2!=[]:
			pd.DataFrame([[a['id'],[keep(b) for b in a['name'].lower().replace(' v. ','|').replace(' vs. ','|').replace(' against ','|').split('|')],a['decision_date']] for a in data2 if a['casebody']['status']=='ok']).to_csv('keep.csv',index=False,header=False,mode='a')
			data2=pd.DataFrame([[a['id'],c[:c.find('"')],c[(first:=c.find('</'))+c[first+1:].find('<p ')+1:]] for a in data2 for b in [a['casebody']['data']] for c in b[b.find('<opinion type=')+b[b.find('<opinion type='):].find('"')+1:b.find('</casebody>\n')].split('</opinion>\n  <opinion type="') if a['casebody']['status']=='ok'], columns=['case', 'stance', 'text'])
			data2['text']=[clean_text(re.sub('<[^>]*>','',re.sub('<page-number label=\"([0-9]*)[^>]*>[^>]*>','@\\1@',text.replace('&amp;','&'))), ['inline_whitespace', 'underscores']) for text in data2['text']]
			w=pool.starmap(func, [(exclude, data3, k, v) for k, v in enumerate(np.array_split(data2,10*num_processes), 10000*num_processes*h+100*num_processes*c)], chunksize=1)
			pd.DataFrame([b for data4 in [dict(data3).copy()] for k in sorted(data4.keys()) for b in data4[k]]).to_csv('unfiltered-LawngNLI2.csv',index=False,header=False,mode='a')
			del data2
			gc.collect()
		print(i)
	with open('exclude.pkl', 'wb') as f:
		pickle.dump(dict(exclude).copy(), f)
	 
if __name__ == '__main__':
    main()
