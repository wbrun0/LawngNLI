from itertools import compress
from collections import deque
from nltk.stem import WordNetLemmatizer
from pattern.en import conjugate
import pandas as pd
import numpy as np
import spacy, benepar, lzma, ujson, random, re, pickle, base64
benepar.download('benepar_en3')
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})
from eyecite import clean_text, get_citations, resolve_citations
from eyecite.helpers import add_defendant
from eyecite.tokenizers import default_tokenizer as tokenizer

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

def contradict(b, w, nlp, clause_id):
	c=b
	l=['^']
	l1=''
	l2=''
	t=0
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
				c=f[int(clause_id[t])]
				l+=d[:c]
				c=d[c]
				t+=1
			else:
				return '@ERROR@'

def pagelookup(cited, pages):
	pages=sorted(pages)
	paragraphs=[i+2 for i, l in enumerate(cited) if l == "\n"]+[0]+[len(cited)+1]
	portion=[]
	for i in pages:
		if cited.find("@"+i+"@")==-1:				
			start=0
		else:
			start=max([j for j in paragraphs if j<=cited.find("@"+i+"@")+len("@"+i+"@")])
		if cited.find("@"+str(int(i)+1)+"@")==-1:
			end=len(cited)
		else:	
			end=max(min([j for j in paragraphs if j>=cited.find("@"+str(int(i)+1)+"@")])-2,0)
		if start==0 and end==len(cited):
			return None
		portion+=[cited[start:end+1]]
	portion='\n'.join(portion)
	return portion
	
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

def remove_citations(cited_text):
	if pd.isnull(cited_text):
		return cited_text
	else:
		text=cited_text
		for a in reversed(get_citations(cited_text, remove_ambiguous=False)):
			b=citation_span(a, cited_text)
			text=text[:b[0]]+'|'*(b[1]-b[0])+text[b[1]:]
		#subset of https://www.thesaurus.com/e/grammar/when-do-you-use-punctuation-marks/
		return re.sub('([^|])([,;:])([|]+[\.?!,;])*([|]+)([\.?!,;:])([^|])','\\1\\5\\6',
		re.sub('([^|])([\.?!"“”‘’\n])([|]+[\.?!,;])*([|]+)([\.?!,;:])([^|])','\\1\\2\\6',text)	
		).replace('|','')

def reinflate(data,file_num):
	subset=set(data['cited_case'])
	data2={}
	for h in file_num:
		if h==0:
			i='data.jsonl.xz~'
		else:
			i='data.jsonl.xz~'+str(h)			
		with lzma.open(i) as f:
			for count, line in enumerate(f):
				a=ujson.loads(str(line, 'utf8'))
				if a['id'] in subset:
					data2[a['id']]=b[0] if len(b:=[clean_text(re.sub('<[^>]*>','',re.sub('<page-number label=\"([0-9]*)[^>]*>[^>]*>','@\\1@',c[(first:=c.find('</'))+c[first+1:].find('<p ')+1:].replace('&amp;','&'))), ['inline_whitespace', 'underscores']) for b in [a['casebody']['data']] for c in b[b.find('<opinion type=')+b[b.find('<opinion type='):].find('"')+1:b.find('</casebody>\n')].split('</opinion>\n  <opinion type="') if a['casebody']['status']=='ok' and c[:c.find('"')]=='majority'])==1 else None
	if data2=={}:
		return None
	else:
		return pd.Series(data2)

data=pd.read_pickle('LawngNLI_minus_case_text.xz')
data2=[]
for file_num in [48]+list(range(0,48))+list(range(49,57)):
	print(file_num); temp=reinflate(data,[file_num])		
	if temp is not None:
		if len(temp)>0:
			data2+=[temp]
data2=pd.concat(data2).to_dict()
data['long_premise_with_citations']=data['cited_case'].map(data2)
data['short_premise_with_citations']=[pagelookup(premise,pages) if ((pages!=frozenset()) & (not pd.isnull(pages))) else None for premise, pages in zip(data['long_premise_with_citations'],data['cited_pages'])]
data['long_premise_with_citations']=data['long_premise_with_citations'].str.replace("@[0-9]*@","", regex=True)
data['short_premise_with_citations']=data['short_premise_with_citations'].str.replace("@[0-9]*@","", regex=True)
data.loc[~((data['short_premise_with_citations']!='') & (~data['short_premise_with_citations'].fillna(value='').str.isspace())),'short_premise_with_citations']=None
with open('LawngNLI_filter.xz','rb') as f:
	h=pickle.load(f)
for l in ['long','short']:
	c=deque([int(a) for a in base64.b64decode(h[l]).decode()])
	data[l+'_premise']=[''.join(list(compress(b:=list(a.replace('|','')),[c.popleft() for _ in range(len(b))]))) for a in data[l+'_premise_with_citations']]
cases=set(data['caseid'])
data2=[]
data3=[]
for h in range(0,57):
	if h==0:
		i='data.jsonl.xz~'
	else:
		i='data.jsonl.xz~'+str(h)			
	with lzma.open(i) as f:
		for j, line in enumerate(f):
			a=ujson.loads(str(line, 'utf8'))
			if a['id'] in cases:
				if ((len(data3) % (10000))==0) & (j!=0):
					data3+=[a]
					data3=pd.DataFrame([[a['id'],c[:c.find('"')],c[(first:=c.find('</'))+c[first+1:].find('<p ')+1:],i] for a in data3 for b in [a['casebody']['data']] for i, c in enumerate(b[b.find('<opinion type=')+b[b.find('<opinion type='):].find('"')+1:b.find('</casebody>\n')].split('</opinion>\n  <opinion type="')) if a['casebody']['status']=='ok'], columns=['caseid', 'stance', 'text', 'i'])
					data3['text']=[clean_text(re.sub('<[^>]*>','',re.sub('<page-number label=\"([0-9]*)[^>]*>[^>]*>','@\\1@',text.replace('&amp;','&'))), ['inline_whitespace', 'underscores']) for text in data3['text']]
					data2+=[data3]
					data3=[]
				else:
					data3+=[a]
				data3=pd.DataFrame([[a['id'],c[:c.find('"')],c[(first:=c.find('</'))+c[first+1:].find('<p ')+1:],i] for a in data3 for b in [a['casebody']['data']] for i, c in enumerate(b[b.find('<opinion type=')+b[b.find('<opinion type='):].find('"')+1:b.find('</casebody>\n')].split('</opinion>\n  <opinion type="')) if a['casebody']['status']=='ok'], columns=['caseid', 'stance', 'text', 'i'])
				data3['text']=[clean_text(re.sub('<[^>]*>','',re.sub('<page-number label=\"([0-9]*)[^>]*>[^>]*>','@\\1@',text.replace('&amp;','&'))), ['inline_whitespace', 'underscores']) for text in data3['text']]
				data2+=[data3]
				data3=[]
data2=pd.concat(data2)
data=data.merge(data2,on=['caseid','i'],how='left')
data['citing_parenthetical']=[a[b:c] for a, b, c in zip(data['text'].str.replace("@[0-9]*@", "", regex=True).str.replace("n't"," not",regex=False).str.replace("can not","cannot",regex=False),data['start_index'].astype(int),data['end_index'].astype(int))]
data['citing_parenthetical']=data['citing_parenthetical'].str.replace("@[0-9]*@", "", regex=True)
def test():
	try:
		conjugate('try','past')
	except:
		pass
test()
w=WordNetLemmatizer()
data.loc[data['h']==True,'citing_parenthetical']=[e for c in data.loc[data['h']==True,'citing_parenthetical'] for e in [(' ').join([''.join([a.text_with_ws if a.orth_.lower()==a.norm_ or a.is_punct else (' '+a.norm_.title() if a.is_title else ' '+a.norm_)+a.whitespace_ for a in h]).replace('  ',' ').replace('can not','cannot').replace(" \([‘’'][sd][^a-zA-Z0-9]\)","\1").strip() for h in list(nlp(re.sub('^[a-z]*ing that ','',c)).sents)])]]
data.loc[data['index']==6941,'citing_parenthetical']=data.loc[data['index']==6941,'citing_parenthetical'].iloc[0].replace('did not','do not')
data['contradicted_parenthetical']=data['citing_parenthetical'].tolist()
data.loc[data['e']==False,'contradicted_parenthetical']=[(' ').join([contradict(b, w, nlp, clause_id) if i==sent_id else b.text for a in [list(nlp(re.sub('^[a-z]*ing that ','',e)).sents)] for i, b in enumerate(a)]) if len(e)>0 else '' for clause_id, sent_id, c in zip(data.loc[data['e']==False,'clause_id'],data.loc[data['e']==False,'sent_id'],data.loc[data['e']==False,'citing_parenthetical']) for e in [(' ').join([''.join([a.text_with_ws if a.orth_.lower()==a.norm_ or a.is_punct else (' '+a.norm_.title() if a.is_title else ' '+a.norm_)+a.whitespace_ for a in h]).replace('  ',' ').replace('can not','cannot').replace(" \([‘’'][sd][^a-zA-Z0-9]\)","\1").strip() for h in list(nlp(re.sub('^[a-z]*ing that ','',c)).sents)])]]
data.loc[data['index']==67431,'contradicted_parenthetical']=data.loc[data['index']==67431,'contradicted_parenthetical'].iloc[0][:130]+'not '+data.loc[data['index']==67431,'contradicted_parenthetical'].iloc[0][130:]
data.loc[data['index']==54396,'contradicted_parenthetical']=data.loc[data['index']==54396,'contradicted_parenthetical'].iloc[0][:27]+'not '+data.loc[data['index']==54396,'contradicted_parenthetical'].iloc[0][27:]
data.loc[data['index']==61876,'contradicted_parenthetical']=data.loc[data['index']==61876,'contradicted_parenthetical'].iloc[0][:104]+'did not '+data.loc[data['index']==61876,'contradicted_parenthetical'].iloc[0][104:111]+data.loc[data['index']==61876,'contradicted_parenthetical'].iloc[0][112:]
data.loc[data['index']==46095,'contradicted_parenthetical']=data.loc[data['index']==46095,'contradicted_parenthetical'].iloc[0][:86]+'not '+data.loc[data['index']==46095,'contradicted_parenthetical'].iloc[0][86:]
data.loc[data['index']==78001,'contradicted_parenthetical']=data.loc[data['index']==78001,'contradicted_parenthetical'].iloc[0][:118]+'not '+data.loc[data['index']==78001,'contradicted_parenthetical'].iloc[0][118:]
data.loc[data['index']==79478,'contradicted_parenthetical']=data.loc[data['index']==79478,'contradicted_parenthetical'].iloc[0][:81]+'not '+data.loc[data['index']==79478,'contradicted_parenthetical'].iloc[0][81:]
data.loc[data['index']==90479,'contradicted_parenthetical']=data.loc[data['index']==90479,'contradicted_parenthetical'].iloc[0][:77]+'not '+data.loc[data['index']==90479,'contradicted_parenthetical'].iloc[0][77:]
data.loc[data['index']==7559,'contradicted_parenthetical']=data.loc[data['index']==7559,'contradicted_parenthetical'].iloc[0][:26]+'not '+data.loc[data['index']==7559,'contradicted_parenthetical'].iloc[0][26:]
data.loc[data['index']==70471,'contradicted_parenthetical']=data.loc[data['index']==70471,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==163463,'contradicted_parenthetical']=data.loc[data['index']==163463,'contradicted_parenthetical'].iloc[0][:90]+'did not '+data.loc[data['index']==163463,'contradicted_parenthetical'].iloc[0][90:97]+data.loc[data['index']==163463,'contradicted_parenthetical'].iloc[0][98:]
data.loc[data['index']==14186,'contradicted_parenthetical']=data.loc[data['index']==14186,'contradicted_parenthetical'].iloc[0][:49]+'did not '+data.loc[data['index']==14186,'contradicted_parenthetical'].iloc[0][49:59]+data.loc[data['index']==14186,'contradicted_parenthetical'].iloc[0][60:]
data.loc[data['index']==31166,'contradicted_parenthetical']=data.loc[data['index']==31166,'contradicted_parenthetical'].iloc[0][:116]+data.loc[data['index']==31166,'contradicted_parenthetical'].iloc[0][120:]
data.loc[data['index']==70192,'contradicted_parenthetical']=data.loc[data['index']==70192,'contradicted_parenthetical'].iloc[0][:204]+'do not '+data.loc[data['index']==70192,'contradicted_parenthetical'].iloc[0][204:]
data.loc[data['index']==59255,'contradicted_parenthetical']=data.loc[data['index']==59255,'contradicted_parenthetical'].iloc[0][:88]+'did not '+data.loc[data['index']==59255,'contradicted_parenthetical'].iloc[0][88:95]+data.loc[data['index']==59255,'contradicted_parenthetical'].iloc[0][97:]
data.loc[data['index']==9072,'contradicted_parenthetical']=data.loc[data['index']==9072,'contradicted_parenthetical'].iloc[0][:92]+'not '+data.loc[data['index']==9072,'contradicted_parenthetical'].iloc[0][92:]
data.loc[data['index']==107994,'contradicted_parenthetical']=data.loc[data['index']==107994,'contradicted_parenthetical'].iloc[0][:106]+'did not '+data.loc[data['index']==107994,'contradicted_parenthetical'].iloc[0][106:112]+data.loc[data['index']==107994,'contradicted_parenthetical'].iloc[0][114:]
data.loc[data['index']==99911,'contradicted_parenthetical']=data.loc[data['index']==99911,'contradicted_parenthetical'].iloc[0][:167]+'did not '+data.loc[data['index']==99911,'contradicted_parenthetical'].iloc[0][167:173]+data.loc[data['index']==99911,'contradicted_parenthetical'].iloc[0][175:]
data.loc[data['index']==160219,'contradicted_parenthetical']=data.loc[data['index']==160219,'contradicted_parenthetical'].iloc[0][:129]+'not '+data.loc[data['index']==160219,'contradicted_parenthetical'].iloc[0][129:]
data.loc[data['index']==116962,'contradicted_parenthetical']=data.loc[data['index']==116962,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==110523,'contradicted_parenthetical']=data.loc[data['index']==110523,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==113965,'contradicted_parenthetical']=data.loc[data['index']==113965,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==56187,'contradicted_parenthetical']=data.loc[data['index']==56187,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==9547,'contradicted_parenthetical']=data.loc[data['index']==9547,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==10391,'contradicted_parenthetical']=data.loc[data['index']==10391,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==156446,'contradicted_parenthetical']=data.loc[data['index']==156446,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==115932,'contradicted_parenthetical']=data.loc[data['index']==115932,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==43467,'contradicted_parenthetical']=data.loc[data['index']==43467,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==59814,'contradicted_parenthetical']=data.loc[data['index']==59814,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==125557,'contradicted_parenthetical']=data.loc[data['index']==125557,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==164813,'contradicted_parenthetical']=data.loc[data['index']==164813,'contradicted_parenthetical'].iloc[0]+'not'
data.loc[data['index']==66345,'contradicted_parenthetical']=data.loc[data['index']==66345,'contradicted_parenthetical'].iloc[0][4:]
data.loc[data['index']==76630,'contradicted_parenthetical']=data.loc[data['index']==76630,'contradicted_parenthetical'].iloc[0]+'no'
data.loc[data['index']==25766,'contradicted_parenthetical']=data.loc[data['index']==25766,'contradicted_parenthetical'].iloc[0][:27]+'not '+data.loc[data['index']==25766,'contradicted_parenthetical'].iloc[0][27:]
data.loc[data['index']==43174,'contradicted_parenthetical']=data.loc[data['index']==43174,'contradicted_parenthetical'].iloc[0][:17]+'not '+data.loc[data['index']==43174,'contradicted_parenthetical'].iloc[0][17:]
data.loc[data['index']==153151,'contradicted_parenthetical']=data.loc[data['index']==153151,'contradicted_parenthetical'].iloc[0][:8]+'not '+data.loc[data['index']==153151,'contradicted_parenthetical'].iloc[0][8:]
data.loc[data['index']==89968,'contradicted_parenthetical']=data.loc[data['index']==89968,'contradicted_parenthetical'].iloc[0][:18]+'did not '+data.loc[data['index']==89968,'contradicted_parenthetical'].iloc[0][18:22]+'y'+data.loc[data['index']==89968,'contradicted_parenthetical'].iloc[0][25:]
data['citing_parenthetical2']=data['citing_parenthetical'].copy()
data.loc[data['flip']==True,'citing_parenthetical']=data.loc[data['flip']==True,'contradicted_parenthetical'].copy()
data.loc[data['flip']==True,'contradicted_parenthetical']=data.loc[data['flip']==True,'citing_parenthetical2'].copy()
data=data.drop(['citing_parenthetical2','end_index','flip','i','start_index'],axis=1)
data=data.rename(columns={'citing_parenthetical': 'hypothesis'})
data=data.set_index('index',drop=False)
data.index.name=None
data=data[['index', 'caseid', 'stance', 'cited_case', 'cited_pages', 'hypothesis', 'case_history_flag', 'contradicted_parenthetical', 'negation', 'long_premise_with_citations', 'short_premise_with_citations', 'portion', 'label', 'similarity', 'long_premise', 'short_premise', 'sample', 'a', 'split']].copy()
data.to_pickle('LawngNLI.xz')
