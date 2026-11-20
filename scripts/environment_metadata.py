"""Export metadata read from supplied NetCDFs; do not infer native resolution from filenames."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import xarray as xr

def audit_environment_sources(paths,output_dir):
    output_dir=Path(output_dir);rows=[];full=[]
    def encode(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        return str(x)
    for file in map(Path,paths):
        with xr.open_dataset(file) as ds:
            coords={name:{'Values':(ds[name].values.astype(str).tolist() if np.issubdtype(ds[name].dtype,np.datetime64) else ds[name].values),'Attributes':dict(ds[name].attrs)} for name in ds.coords}
            full.append({'File':str(file),'SHA256':hashlib.sha256(file.read_bytes()).hexdigest(),
                         'GlobalAttributes':dict(ds.attrs),'Variables':{name:dict(ds[name].attrs) for name in ds.data_vars},
                         'Coordinates':coords})
            lat=next((x for x in ['lat','latitude'] if x in ds.coords),None)
            lon=next((x for x in ['lon','longitude'] if x in ds.coords),None)
            for name in ds.data_vars:
                rows.append({'File':file.name,'Variable':name,'Units':ds[name].attrs.get('units',''),
                             'ProductID_Metadata':ds.attrs.get('subset:productId',''),
                             'DatasetID_Metadata':ds.attrs.get('subset:datasetId',''),
                             'DownloadDate_Metadata':ds.attrs.get('subset:date',''),
                             'VersionEvidence':ds.attrs.get('subset:datasetId',ds.attrs.get('title','')),
                             'LatitudeSpacing_deg_measured':float(np.median(np.abs(np.diff(ds[lat].values)))) if lat else np.nan,
                             'LongitudeSpacing_deg_measured':float(np.median(np.abs(np.diff(ds[lon].values)))) if lon else np.nan,
                             'TimeFirst':str(ds.time.min().values) if 'time' in ds else '',
                             'TimeLast':str(ds.time.max().values) if 'time' in ds else '',
                             'N_TimeLayers':ds.sizes.get('time',0),
                             'NativeResolutionStatus':'metadata only; measured spacing describes supplied file',
                             'UpstreamProcessing':ds.attrs.get('processing',ds.attrs.get('temporal_processing','not documented in supplied file')),
                             'SecondarySourceDownloadDate':'not independently established' if 'source_file_2' in ds.attrs else '',
                             'SHA256':hashlib.sha256(file.read_bytes()).hexdigest()})
    a=output_dir/'17a_source_product_metadata_audit.csv';b=output_dir/'17b_source_netcdf_attributes.json'
    pd.DataFrame(rows).to_csv(a,index=False,encoding='utf-8-sig')
    b.write_text(json.dumps(full,default=encode,ensure_ascii=False,indent=2),encoding='utf-8')
    return [a,b]