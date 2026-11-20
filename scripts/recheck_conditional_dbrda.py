#!/usr/bin/env python3
"""Rerun the archived conditional dbRDA tests using private prepared inputs.

This is an independently reconstructed check, not the original runtime log.
Statistical calculations are delegated to the archived environment module.
All generated results must remain in a private output directory.
"""
from pathlib import Path
import argparse, os, importlib.util, sys, json, hashlib, platform
import pandas as pd, numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=Path(os.environ.get('CEPHALOPOD_PRIVATE_DATA_DIR','data_private'))/'02_analysis_data_strict_with_anomalies.csv')
parser.add_argument('--source-module', type=Path, default=Path(os.environ.get('CEPHALOPOD_ENVIRONMENT_MODULE',str(Path(__file__).with_name('step6_environment_models.py')))))
parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get('CEPHALOPOD_FIG9_OUTPUT_DIR','outputs/figure9')))
args=parser.parse_args()
src=args.source_module.resolve()
inp=args.input.resolve()
out=args.output_dir.resolve()
for path in (src,inp):
 if not path.is_file(): parser.error(f'Required private input or source module not found: {path}')
out.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('revised_step6',src)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
d=pd.read_csv(inp);d=d.loc[d.Cephalopod_Presence.eq(1)].copy()
tax=pd.DataFrame({'Species_Code':[f'Species_{i:02d}' for i in range(1,26)]})
rng=np.random.default_rng(m.RANDOM_SEED+6100);rows=[]
for month in (4,11):
 print('START',month,flush=True)
 o=m.ordination_for_season(d,tax,month,rng);rows.extend(o[0]);print('END',month,flush=True)
df=pd.DataFrame(rows)
df['P_FDR_within_analysis_season']=df.groupby(['Analysis','Season'],observed=True).P_value.transform(lambda x:m.bh_fdr(x.to_numpy()))
df.to_csv(out/'conditional_dbrda_recomputed.csv',index=False,encoding='utf-8-sig')
meta={'status':'new independent recomputation, not original runtime log','source_script':str(src),'source_script_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'input':str(inp),'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'Python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':m.__import__('scipy').__version__ if hasattr(m,'__import__') else __import__('scipy').__version__,'seed':m.RANDOM_SEED,'permutations':m.PERMUTATIONS,'ordination_rng_initial_seed':m.RANDOM_SEED+6100,'check_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'result_scope':'Independent recomputation of conditional tests; not an original full pipeline runtime log'}
(out/'recomputation_provenance.json').write_text(json.dumps(meta,indent=2))
print(df[['Analysis','Month','Term','F','TermFractionTotal','P_value','P_FDR_within_analysis_season']].to_string(index=False),flush=True)
