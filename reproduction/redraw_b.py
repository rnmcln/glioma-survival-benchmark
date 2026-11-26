"""Render the class-weighting figure with the displayed prevalence defined correctly."""
from pathlib import Path
import sys

repo,target=map(Path,sys.argv[1:])
sys.path[:0]=[str(repo/'scripts'),str(repo)]
source=repo/'scripts/08_figures.py'
code=source.read_text()
assert 'Logit of event rate in training data' in code
code=code.replace('Logit of event rate in training data','Logit of event rate in the setting')
code=code.replace('"font.family": "Inter"','"font.family": "DejaVu Sans"')
namespace={'__name__':'reviewed_benchmark_weighting','__file__':str(source)}
exec(compile(code,str(source),'exec'),namespace)
namespace['config'].FIGURES=target
namespace['figure3_weighting'](stem='Figure2')
