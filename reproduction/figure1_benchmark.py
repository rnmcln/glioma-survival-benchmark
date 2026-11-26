"""Plot 24-month paired contrasts under both tuning criteria."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

src, dest = map(Path,sys.argv[1:])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.5,
                     'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
                     'pdf.fonttype':42,'xtick.major.width':.5,'ytick.major.size':0})
fig, axs = plt.subplots(1,2,figsize=(7.15,4.45),sharey=True)
fig.subplots_adjust(left=.36,right=.98,top=.94,bottom=.20,wspace=.22)
conditions=[
    ('TCGA internal',None,None,None),
    ('Gravendeel, mode imputation','gravendeel','mode','all'),
    ('Gravendeel, model-based imputation','gravendeel','model_imputed','all'),
    ('Gravendeel, recorded IDH only','gravendeel','mode','idh_recorded'),
    ('REMBRANDT','rembrandt','mode','all'),
]
comps=['cox_mlp - cox_linear','cox_mlp - cox_flexible']
colors=['#396D7C','#363B42']
labels=[]
positions=[]
for j,comp in enumerate(comps):
    for i,(label,tar,imp,sub) in enumerate(conditions):
        y=12-6*j-i
        labels.append(label); positions.append(y)
        for k,s in enumerate(('', '_pll')):
            for ax,metric in zip(axs,('td_auc_24','brier_24')):
                kind='primary' if tar is None else 'external'
                data=pd.read_csv(src/f'survival_{kind}_differences{s}.csv')
                rows=data.loc[(data.comparison==comp)&(data.metric==metric)]
                if tar:
                    rows=rows.loc[(rows.target==tar)&(rows.idh_handling==imp)&(rows.subset==sub)]
                assert len(rows)==1,(comp, tar, imp, sub, metric)
                r=rows.iloc[0]
                yy=y+(.16 if k==0 else -.16)
                ax.hlines(yy,r.lo,r.hi,color=colors[k],lw=.75)
                ax.plot(r.delta,yy,'o',markersize=3.5,mec=colors[k],mew=.7,
                        mfc=colors[k] if k==0 else 'white')
for i,(ax,label) in enumerate(zip(axs,('Difference in time-dependent AUROC','Difference in IPCW Brier score'))):
    ax.axvline(0,color='.63',lw=.6,ls='--',zorder=0)
    ax.set_xlabel(label)
    ax.set_ylim(1.3,13.55)
    ax.text(0,1.01,'AB'[i],transform=ax.transAxes,fontsize=10,fontweight='bold')
axs[0].set_yticks(positions,labels)
axs[0].tick_params(axis='y',pad=6)
for y,label in [(13.04,'Network vs linear Cox'),(7.05,'Network vs flexible Cox')]:
    axs[0].text(-.03,y,label,transform=axs[0].get_yaxis_transform(),ha='right',va='center',weight='medium')
fig.legend(handles=[
    Line2D([],[],ls='none',marker='o',color=c,markerfacecolor=c if i==0 else 'white',
           markersize=4,label=t)
    for i,(c,t) in enumerate(zip(colors,('Concordance tuning','Partial-likelihood tuning')))],
    frameon=False,loc='lower center',bbox_to_anchor=(.59,.02),ncol=2,fontsize=8)
fig.savefig(dest/'Figure1.pdf')
fig.savefig(dest/'Figure1.tif',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
