from pathlib import Path
import re, unicodedata, sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; RES=ROOT/'results'; FIG=RES/'figures'; FIG.mkdir(parents=True,exist_ok=True)
GEN=RES/'gemma_generations_checkpoint.csv'; EVAL=RES/'gemma_evaluation_per_example.csv'
for p in (GEN,EVAL):
    if not p.exists(): sys.exit(f'Missing required file: {p}')
def read(p): return pd.read_csv(p,encoding='utf-8-sig',keep_default_na=False)
g,e=read(GEN),read(EVAL)
def col(df,opts,required=True):
    low={str(c).lower().strip():c for c in df.columns}
    for x in opts:
        if x.lower() in low:return low[x.lower()]
    for c in df.columns:
        if any(x.lower() in str(c).lower() for x in opts):return c
    if required:raise ValueError(f'Cannot find {opts}. Available columns: {list(df.columns)}')
    return None
gl=col(g,['language','lang']); gi=col(g,['id','example_id','article_id'],False)
gr=col(g,['human_summary','reference','summary']); ga=col(g,['gemma_summary','generated_summary','prediction','gemma_prediction']); gs=col(g,['status'],False)
el=col(e,['language','lang']); ei=col(e,['id','example_id','article_id'],False)
if gs:g=g[g[gs].astype(str).str.lower().isin(['success','ok','successful',''])].copy()
g['_lang']=g[gl].astype(str).str.strip().str.lower(); e['_lang']=e[el].astype(str).str.strip().str.lower()
g['_id']=g[gi].astype(str).str.strip() if gi else g.groupby('_lang').cumcount().astype(str)
e['_id']=e[ei].astype(str).str.strip() if ei else e.groupby('_lang').cumcount().astype(str)
g['_human']=g[gr].astype(str); g['_gemma']=g[ga].astype(str)
g=g[(g['_human'].str.strip()!='')&(g['_gemma'].str.strip()!='')].copy()
metrics=[]
for c in e.columns:
    s=str(c).lower()
    if any(k in s for k in ['rouge','bertscore','bert_score']) and pd.to_numeric(e[c],errors='coerce').notna().any():metrics.append(c)
if not metrics:raise ValueError(f'No numeric ROUGE/BERTScore columns found. Evaluation columns: {list(e.columns)}')
m=g.merge(e[['_lang','_id']+metrics],on=['_lang','_id'],how='left',suffixes=('','_eval'))
if m[metrics].isna().all(axis=None):
    g['_order']=g.groupby('_lang').cumcount(); e['_order']=e.groupby('_lang').cumcount()
    m=g.merge(e[['_lang','_order']+metrics],on=['_lang','_order'],how='left',suffixes=('','_eval'))
if m[metrics].isna().all(axis=None):raise ValueError('Could not match metric scores to generations. Check language/ID columns.')
def tokens(x):return re.findall(r'[\w]+',unicodedata.normalize('NFC',str(x)).lower(),flags=re.UNICODE)
m['human_tokens']=m['_human'].map(lambda x:len(tokens(x))); m['gemma_tokens']=m['_gemma'].map(lambda x:len(tokens(x)))
m['length_difference_gemma_minus_human']=m['gemma_tokens']-m['human_tokens']
m['length_ratio_gemma_to_human']=m['gemma_tokens']/m['human_tokens'].replace(0,pd.NA)
article=col(g,['article','text','source_article'],False)
if article:
    src=g[['_lang','_id',article]].drop_duplicates(['_lang','_id']).rename(columns={article:'_article'})
    m=m.merge(src,on=['_lang','_id'],how='left'); m['article_tokens']=m['_article'].astype(str).map(lambda x:len(tokens(x)))
    m['human_compression_ratio']=m['human_tokens']/m['article_tokens'].replace(0,pd.NA); m['gemma_compression_ratio']=m['gemma_tokens']/m['article_tokens'].replace(0,pd.NA)
m.to_csv(RES/'human_vs_gemma_per_example_analysis.csv',index=False,encoding='utf-8-sig')
cols=['human_tokens','gemma_tokens','length_difference_gemma_minus_human']+[c for c in ['human_compression_ratio','gemma_compression_ratio'] if c in m]+metrics
rows=[]
for lang,grp in m.groupby('_lang'):
    for c in cols:
        x=pd.to_numeric(grp[c],errors='coerce').dropna(); rows.append({'language':lang,'measure':c,'n':len(x),'mean':x.mean(),'sd':x.std(ddof=1),'median':x.median(),'q1':x.quantile(.25),'q3':x.quantile(.75)})
pd.DataFrame(rows).to_csv(RES/'human_vs_gemma_descriptive_stats.csv',index=False,encoding='utf-8-sig')
rows=[]
for lang,grp in m.groupby('_lang'):
    r={'language':lang,'n':len(grp)}
    for c in cols:
        x=pd.to_numeric(grp[c],errors='coerce').dropna()
        if len(x):r[c+'_mean']=x.mean();r[c+'_sd']=x.std(ddof=1);r[c+'_median']=x.median()
    rows.append(r)
pd.DataFrame(rows).to_csv(RES/'human_vs_gemma_language_summary.csv',index=False,encoding='utf-8-sig')
langs=list(m['_lang'].dropna().unique())
# Figure 1: paired lengths
fig,axes=plt.subplots(1,len(langs),figsize=(6*len(langs),5),squeeze=False)
for ax,lang in zip(axes[0],langs):
    z=m[m['_lang']==lang]; ax.scatter(z['human_tokens'],z['gemma_tokens'],alpha=.65)
    lim=max(1,int(max(z['human_tokens'].max(),z['gemma_tokens'].max())*1.05)); ax.plot([0,lim],[0,lim],linestyle='--')
    ax.set(xlabel='Human reference length (tokens)',ylabel='Gemma summary length (tokens)',title=f'{lang.title()} (n={len(z)})',xlim=(0,lim),ylim=(0,lim))
fig.suptitle('Human reference vs Gemma summary length');fig.tight_layout();fig.savefig(FIG/'figure1_summary_length_scatter.png',dpi=300,bbox_inches='tight');plt.close(fig)
# Figure 2: ROUGE-L distribution
rl=[c for c in metrics if 'rougel' in c.lower() or 'rouge_l' in c.lower()]
if rl:
    c=rl[0]; vals=[pd.to_numeric(m.loc[m['_lang']==l,c],errors='coerce').dropna().values for l in langs]
    fig,ax=plt.subplots(figsize=(7,5));ax.boxplot(vals,tick_labels=[l.title() for l in langs],showmeans=True);ax.set(ylabel=c,title='ROUGE-L distribution by language',ylim=(0,1));fig.tight_layout();fig.savefig(FIG/'figure2_rougel_by_language.png',dpi=300,bbox_inches='tight');plt.close(fig)
# Figure 3: metric means by language
means=m.groupby('_lang')[metrics].mean(numeric_only=True);fig,ax=plt.subplots(figsize=(max(9,1.4*len(metrics)),5));width=.8/max(1,len(langs));xs=list(range(len(metrics)))
for i,lang in enumerate(langs):ax.bar([x-.4+width/2+i*width for x in xs],[means.loc[lang,c] if lang in means.index else float('nan') for c in metrics],width=width,label=lang.title())
ax.set_xticks(xs);ax.set_xticklabels(metrics,rotation=25,ha='right');ax.set(ylabel='Mean score',ylim=(0,1),title='Mean automatic evaluation scores by language');ax.legend();fig.tight_layout();fig.savefig(FIG/'figure3_metric_means_by_language.png',dpi=300,bbox_inches='tight');plt.close(fig)
# Figure 4: length difference
fig,ax=plt.subplots(figsize=(7,5));vals=[m.loc[m['_lang']==l,'length_difference_gemma_minus_human'].dropna().values for l in langs];ax.boxplot(vals,tick_labels=[l.title() for l in langs],showmeans=True);ax.axhline(0,linestyle='--');ax.set(ylabel='Gemma length − human reference length (tokens)',title='Paired summary-length differences');fig.tight_layout();fig.savefig(FIG/'figure4_length_difference.png',dpi=300,bbox_inches='tight');plt.close(fig)
print(f'Analysed {len(m)} paired examples.');print(m.groupby('_lang').size().to_string());print('Metric columns:',', '.join(metrics));print('Tables saved: human_vs_gemma_per_example_analysis.csv, human_vs_gemma_descriptive_stats.csv, human_vs_gemma_language_summary.csv');print('Figures saved under results/figures/.')
print('Caution: XL-Sum references are benchmark summaries, not independent human participants.')
