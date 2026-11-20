"""Run the Chapter 2 statistical analyses (no stand-alone figure-rendering stages).
The fisheries input is private; gridded public environmental files may be under
  data/environment/.
python run_pipeline.py --config config.local.json --stages all
python run_pipeline.py --config config.local.json --stages models
"""
from pathlib import Path
import argparse, os, sys, json, subprocess, hashlib, time
SCRIPTS=Path(__file__).resolve().parent/'scripts'
STAGES={'clean':'step1_clean.py','environment':'step2_environment.py','master':'step3_master.py',
        'resources':'step4_resources.py','composition':'step4b_composition.py',
        'alpha':'step4c_alpha.py','beta':'step5_beta.py','spatial':'section33_spatial.py',
        'models':'step6_environment_models.py'}

def configure(config):
    root=Path(config['output_root']).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    # Avoid cross-stage module replacement and nested BLAS oversubscription.
    env.update(OPENBLAS_NUM_THREADS=str(config.get('blas_threads',2)),OMP_NUM_THREADS=str(config.get('blas_threads',2)),
               MKL_NUM_THREADS=str(config.get('blas_threads',2)),PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    raw=Path(config['fishery_csv']).expanduser().resolve();eroot=Path(config['environment_root']).expanduser().resolve()
    values={'CEPHALOPOD_DEPTH_NC':eroot/'Depth_0.005°_2024.nc','CEPHALOPOD_INPUT_CSV':raw,'CEPHALOPOD_EVENT_SOURCE':raw,'CEPHALOPOD_ENV_ROOT':eroot,
            'CEPHALOPOD_OUTPUT_DIR':root/'cephalopod_step1_cleaned','CEPHALOPOD_STEP1_DIR':root/'cephalopod_step1_cleaned',
            'CEPHALOPOD_STEP2_OUTPUT_DIR':root/'cephalopod_step2_environment_matched','CEPHALOPOD_STEP2_DIR':root/'cephalopod_step2_environment_matched',
            'CEPHALOPOD_STEP3_OUTPUT_DIR':root/'cephalopod_step3_analysis_master','CEPHALOPOD_STEP3_ZIP':root/'cephalopod_step3_analysis_master.zip',
            'CEPHALOPOD_STEP3_SOURCE':root/'cephalopod_step3_analysis_master.zip','CEPHALOPOD_STEP4_OUTPUT_DIR':root/'cephalopod_step4_resource_iri_alpha',
            'CEPHALOPOD_STEP4_ZIP':root/'cephalopod_step4_resource_iri_alpha.zip','CEPHALOPOD_STEP4B_OUTPUT_ROOT':root,'CEPHALOPOD_STEP4C_OUTPUT_ROOT':root,
            'CEPHALOPOD_STEP5_OUTPUT_DIR':root/'cephalopod_step5_beta_diversity_region','CEPHALOPOD_STEP5B_OUTPUT_DIR':root/'cephalopod_step5b_region_detailed_analysis',
            'CEPHALOPOD_STEP5_SOURCE':root/'cephalopod_step5_beta_diversity_region.zip','CEPHALOPOD_STEP6_OUTPUT_ROOT':root,
            'CEPHALOPOD_SECTION33_OUTPUT_DIR':root/'cephalopod_section3_3_spatial_shifts_hotspots','CEPHALOPOD_FIGURE1_OUTPUT_ROOT':root,
            'CEPHALOPOD_REGION_MAPPING_SOURCE':root/'00_cephalopod_region_mapping.csv','CEPHALOPOD_REGION_MAPPING':root/'00_cephalopod_region_mapping.csv'}
    env.update({k:str(v) for k,v in values.items()})
    defaults={'CEPHALOPOD_REQUIRE_REGION_MAPPING':'1','CEPHALOPOD_STEP6_RUN_CV':'1','CEPHALOPOD_STEP6_RUN_ORDINATION':'1',
              'CEPHALOPOD_STEP4B_RUN_GEE':'0','CEPHALOPOD_PERMUTATIONS':'999','CEPHALOPOD_PAIRWISE_PERMUTATIONS':'999',
              'CEPHALOPOD_NMDS_RANDOM_STARTS':'8','CEPHALOPOD_NMDS_MAX_ITER':'1000',
              'CEPHALOPOD_NMDS_EPS':'1e-6','CEPHALOPOD_NMDS_VALIDATION_ITER':'100',
              'CEPHALOPOD_NMDS_MAX_VALIDATION_ROUNDS':'3','CEPHALOPOD_MAX_SIMPER_CONTRASTS':'0'}
    env.update(defaults);env.update({str(k):str(v) for k,v in config.get('environment_overrides',{}).items()})
    return root,env

def build_mapping(root):
    import numpy as np,pandas as pd
    d=pd.read_csv(root/'cephalopod_step1_cleaned/09_location_dictionary.csv')
    if d.LocationID.duplicated().any():raise ValueError('Duplicate location in Step 1 station dictionary')
    if not d.Lat.between(27,31).all():raise ValueError('Station outside latitude bounds')
    d['Region']=np.select([d.Lat<28.5,d.Lat<30],['South','Central'],default='North')
    d['Region_Order']=d.Region.map({'South':1,'Central':2,'North':3})
    d['Region_Definition']=d.Region.map({'South':'27 <= latitude < 28.5','Central':'28.5 <= latitude < 30','North':'30 <= latitude <= 31'})
    d.to_csv(root/'00_cephalopod_region_mapping.csv',index=False,encoding='utf-8-sig')
    counts=d.Region.value_counts().to_dict()
    if counts!={'North':47,'Central':41,'South':36}:raise ValueError(f'Region counts differ from manuscript reference: {counts}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));ap.add_argument('--stages',default='all');args=ap.parse_args()
    config=json.loads(args.config.read_text());root,env=configure(config)
    stages=list(STAGES) if args.stages=='all' else args.stages.split(',')
    invalid=set(stages)-set(STAGES)
    if invalid:raise ValueError(f'Unknown stages: {invalid}')
    audit=root/'run_audit';audit.mkdir(exist_ok=True)
    (audit/'configuration.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    (audit/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    inputs=[]
    paths=[Path(config['fishery_csv']),*Path(config['environment_root']).glob('*.nc')]
    for p in paths:
        if p.exists():inputs.append({'Path':str(p.resolve()),'Bytes':p.stat().st_size,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (audit/'input_hashes.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
    for stage in stages:
        if stage in {'beta','spatial'} and not (root/'00_cephalopod_region_mapping.csv').exists():build_mapping(root)
        cmd=[sys.executable,str(SCRIPTS/STAGES[stage])]
        print(f'Running {stage}; full output: {audit}/{stage}.log',flush=True)
        start=time.time()
        with (audit/f'{stage}.log').open('w',encoding='utf-8') as log:
            process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        with (audit/'stage_status.jsonl').open('a') as f:
            f.write(json.dumps({'Stage':stage,'ReturnCode':process.returncode,'Seconds':time.time()-start,'Command':cmd,'CodeSHA256':hashlib.sha256((SCRIPTS/STAGES[stage]).read_bytes()).hexdigest()})+'\n')
        if process.returncode:
            print((audit/f'{stage}.log').read_text()[-6000:])
            raise RuntimeError(f'{stage} failed; inspect its log before continuing')
        if stage=='clean':build_mapping(root)
        print(f'Completed {stage} in {time.time()-start:.1f}s',flush=True)
if __name__=='__main__':main()