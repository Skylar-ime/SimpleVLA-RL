import requests,xml.etree.ElementTree as E, concurrent.futures,json,time
from pathlib import Path
queries={
'direct':'(all:"vision-language-action" OR all:VLA OR all:robot) AND (all:"self-distillation" OR all:"self distillation" OR all:SDPO)',
'feedback':'(all:"vision-language-action" OR all:VLA) AND (all:"rich feedback" OR all:"language feedback" OR all:"hindsight" OR all:"retrospective")',
'near':'ti:GRAPE OR ti:"RoboMonkey" OR ti:"RoboCLIP" OR ti:"RoboReward" OR ti:"RECAP" OR ti:"Reinforcement Learning via Self-Distillation"',
'language':'(ti:"Learning from Language Feedback" OR ti:"Learning from Corrections" OR ti:"Reinforcement Learning with Human" OR ti:"Reflect") AND (all:robot OR all:robotic)',
}
ns={'a':'http://www.w3.org/2005/Atom'}
def f(item):
 k,q=item
 try:
  r=requests.get('https://export.arxiv.org/api/query',params={'search_query':q,'start':0,'max_results':45,'sortBy':'relevance'},timeout=55)
  r.raise_for_status(); root=E.fromstring(r.text)
  rows=[{tag:' '.join(e.findtext('a:'+tag,default='',namespaces=ns).split()) for tag in ['id','title','published','summary']} for e in root.findall('a:entry',ns)]
  Path('/home/skylar/@lab/sdpo_vla_research/'+k+'.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
  return k,rows
 except Exception as e:return k,str(e)
for k,rows in concurrent.futures.ThreadPoolExecutor(2).map(f,queries.items()):
 print('\nQUERY',k)
 if isinstance(rows,list):
  for x in rows:print(x['id'],x['title'],'\n',x['summary'][:1500])
 else:print(rows)
