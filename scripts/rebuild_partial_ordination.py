#!/usr/bin/env python3
"""Reconstruct partial dbRDA display coordinates; no permutations are rerun.

Responses and predictors are both projected off the identical station/year
nuisance space used by the archived conditional tests. SVD is performed only
on their conditional fitted multivariate response, never on the old
unconditional four-variable ordination. The first three axes are retained.

Raw residual-response scores are Y_residual @ V and fitted (LC) scores are
Y_fitted @ V. Their coordinate units are the original sqrt(Bray-Curtis) PCoA
units. Arrow correlations are provided for both choices, without arbitrary
plot expansion. Plotting must declare the selected scores and arrow scaling.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import json, sys, hashlib, importlib.util, platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy import linalg
from scipy.spatial.distance import pdist, squareform
from threadpoolctl import threadpool_limits
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=Path(os.environ.get('CEPHALOPOD_PRIVATE_DATA_DIR','data_private'))/'02_analysis_data_strict_with_anomalies.csv')
parser.add_argument('--source-module', type=Path, default=Path(os.environ.get('CEPHALOPOD_ENVIRONMENT_MODULE',str(Path(__file__).with_name('step6_environment_models.py')))))
parser.add_argument('--output-dir', type=Path, default=Path(os.environ.get('CEPHALOPOD_FIG9_OUTPUT_DIR','outputs/figure9')))
parser.add_argument('--reference-tests', type=Path, default=None, help='Private test-result CSV; default is conditional_dbrda_recomputed.csv in output-dir.')
args=parser.parse_args()
OUT=args.output_dir.resolve()
SOURCE=args.source_module.resolve()
INPUT=args.input.resolve()
REF=(args.reference_tests or OUT/'conditional_dbrda_recomputed.csv').resolve()
for path in (SOURCE,INPUT,REF):
 if not path.is_file(): parser.error(f'Required private input or source module not found: {path}')
OUT.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('archived_environment',SOURCE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
variables=['SST_Anomaly','SSS_Anomaly','log1p_Chla_0125_Anomaly']
biomass=[f'B_Species_{i:02d}_kg_km2' for i in range(1,26)]
all_data=pd.read_csv(INPUT)
reference=pd.read_csv(REF)
score_frames=[];arrow_rows=[];axis_rows=[];month_meta=[]
with threadpool_limits(limits=1):
 for month in (4,11):
  d=all_data.loc[(all_data.Month.eq(month))&(all_data.Cephalopod_Presence.eq(1))].dropna(subset=['Depth_m',*variables,*biomass]).copy().reset_index(drop=True)
  abundance=np.sqrt(d[biomass].to_numpy(float))
  assert np.all(abundance.sum(axis=1)>0)
  distance=squareform(pdist(abundance,metric='braycurtis'))
  pcoa=m.lingoes_pcoa(np.sqrt(distance))
  y=pcoa.coordinates
  env,scale=m.standardized_linear_design(d,variables)
  station=pd.get_dummies(d.LocationID.astype(str),prefix='LocationID',drop_first=True,dtype=float)
  year=pd.get_dummies(d.Year.astype(str),prefix='Year',drop_first=True,dtype=float)
  nuisance=np.column_stack([np.ones(len(d)),station.to_numpy(),year.to_numpy()])
  qc,rankc=m.qr_projection(nuisance)
  yr=y-m.fitted_by_q(qc,y)
  xr=env[:,1:]-m.fitted_by_q(qc,env[:,1:])
  qx,rankx=m.qr_projection(xr)
  # Small rank x PCoA-dimension SVD is exactly the nonzero SVD of fitted Y.
  small=qx.T@yr
  u,s,vt=linalg.svd(small,full_matrices=False,check_finite=False)
  fitted=qx@small
  lc=(qx@u)*s
  response=yr@vt.T
  # Deterministic display signs: strongest environmental correlation is positive.
  for j in range(rankx):
   corr=np.array([np.corrcoef(xr[:,i],lc[:,j])[0,1]for i in range(3)])
   sign=1. if corr[np.argmax(np.abs(corr))]>=0 else -1.
   lc[:,j]*=sign;response[:,j]*=sign;vt[j]*=sign
  ev=s*s
  total=float(np.square(y-y.mean(axis=0)).sum())
  remaining=float(np.square(yr).sum())
  joint=float(np.square(fitted).sum())
  qf,rankf=m.qr_projection(np.column_stack([nuisance,env[:,1:]]))
  full_rss=m.rss_by_q(qf,y)
  joint_by_nested_rss=m.rss_by_q(qc,y)-full_rss
  observed_f=(joint/rankx)/(full_rss/(len(d)-rankf))
  ref=reference.loc[(reference.Month==month)&(reference.Analysis=='Conditional_LocationID_Year')&(reference.Term=='AllTemporalEnvironment')].iloc[0]
  checks={
   'rank_nuisance':rankc,'rank_environment_residual':rankx,'rank_full':rankf,
   'condition_response_max_absolute':float(np.abs(qc.T@yr).max()),
   'condition_predictor_max_absolute':float(np.abs(qc.T@xr).max()),
   'condition_fitted_max_absolute':float(np.abs(qc.T@fitted).max()),
   'fitted_ss_minus_nested_model_ss':joint-joint_by_nested_rss,
   'axis_eigenvalue_sum_minus_joint_ss':float(ev.sum()-joint),
   'response_to_LC_projection_max_absolute':float(np.abs(qx@(qx.T@response)-lc).max()),
   'joint_fraction_minus_reference':float(joint/total-ref.TermFractionTotal),
   'joint_F_minus_reference':float(observed_f-ref.F),
  }
  assert rankx==3 and rankf-rankc==3
  for k,v in checks.items():
   if k not in ['rank_nuisance','rank_environment_residual','rank_full']:assert abs(v)<1e-9,(month,k,v)
  coef=linalg.lstsq(xr,lc)[0]
  output=d[['EventID','Year','Month','LocationID','Lon','Lat','Depth_m']].copy()
  for j in range(3):
   output[f'ResidualResponse_dbRDA{j+1}']=response[:,j]
   output[f'FittedLC_dbRDA{j+1}']=lc[:,j]
   axis_rows.append(dict(Month=month,Axis=f'dbRDA{j+1}',EigenvalueSS=float(ev[j]),PercentConstrained=float(100*ev[j]/joint),PercentTotal=float(100*ev[j]/total),PercentAfterCondition=float(100*ev[j]/remaining)))
  for i,v in enumerate(variables):
   output[f'{v}_conditional_residual_z']=xr[:,i]
   arrow=dict(Month=month,Variable=v,OriginalMean=scale[v][0],OriginalSD=scale[v][1])
   for j in range(3):
    arrow[f'Correlation_FittedLC_dbRDA{j+1}']=float(np.corrcoef(xr[:,i],lc[:,j])[0,1])
    arrow[f'Correlation_ResidualResponse_dbRDA{j+1}']=float(np.corrcoef(xr[:,i],response[:,j])[0,1])
    arrow[f'RegressionCoefficient_FittedLC_dbRDA{j+1}']=float(coef[i,j])
   arrow_rows.append(arrow)
  score_frames.append(output)
  month_meta.append(dict(Month=month,N=len(d),N_Locations=int(d.LocationID.nunique()),PCoAAxes=pcoa.axes_retained,LingoesConstant=pcoa.lingoes_constant,TotalSS=total,ConditionedRemainingSS=remaining,JointConstrainedSS=joint,JointFractionTotal=joint/total,JointFractionRemaining=joint/remaining,F=observed_f,DF_Term=rankx,DF_Residual=len(d)-rankf,FirstTwoAxesFractionConstrained=float(ev[:2].sum()/joint),FirstTwoAxesFractionTotal=float(ev[:2].sum()/total),NumericalChecks=checks))
pd.concat(score_frames,ignore_index=True).to_csv(OUT/'partial_dbrda_event_scores.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(arrow_rows).to_csv(OUT/'partial_dbrda_environment_vectors.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(axis_rows).to_csv(OUT/'partial_dbrda_axes.csv',index=False,encoding='utf-8-sig')
meta=dict(Description='New reconstruction of conditional ordination, consistent with existing all-positive-axis conditional dbRDA effect sizes; no permutation rerun.',InputPath=str(INPUT),InputSHA256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),ArchivedSourcePath=str(SOURCE),ArchivedSourceSHA256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),ReconstructionScriptSHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),Packages=dict(Python=platform.python_version(),NumPy=np.__version__,pandas=pd.__version__,SciPy=scipy.__version__),ScoreConvention='Unscaled conditional PCoA coordinates projected on right singular vectors; provided both residual-response and environment-fitted LC scores.',ArrowConvention='Pearson correlations of station/year-residualized standardized environmental predictors with the specified axis scores; regression coefficients separately provided.',DisplaySignConvention='Largest absolute environmental correlation with each fitted LC axis is positive.',InferenceCaution='Axis directions and correlations are descriptive; they do not test individual predictors and do not imply causal effects. This operation does not produce new p values.',Months=month_meta)
(OUT/'partial_dbrda_reconstruction.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(json.dumps(month_meta,ensure_ascii=False,indent=2))
