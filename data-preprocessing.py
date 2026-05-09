import os
import pandas as pd
import numpy as np
import warnings
import requests
import tarfile
import io
import glob

from nilearn import maskers
from joblib import Parallel, delayed
from tqdm import tqdm

DATASET_ROOT = r'C:\Users\T2430431\Downloads\RawDataBIDS'
OUTPUT_FILE = r'C:\Users\T2430431\Downloads\RawDataBIDS\adhd_deepfmri_aal.npy'
ATLAS_DIR = r'C:\Users\T2430431\Downloads\Atlas_Cache' 
N_JOBS = -1 
TARGET_LENGTH = 172


if not os.path.exists(ATLAS_DIR):
    os.makedirs(ATLAS_DIR)


found_atlas = glob.glob(os.path.join(ATLAS_DIR, '**', 'ROI_MNI_V4.nii'), recursive=True)

if not found_atlas:
    print("Atlas not found. Downloading Classic AAL (SSL Verify=False)...")
    url = "https://www.gin.cnrs.fr/wp-content/uploads/aal_for_SPM12.tar.gz"
    try:
        response = requests.get(url, verify=False, stream=True)
        response.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            tar.extractall(path=ATLAS_DIR)
        print("Download successful!")
        found_atlas = glob.glob(os.path.join(ATLAS_DIR, '**', 'ROI_MNI_V4.nii'), recursive=True)
    except Exception as e:
        print(f"Download failed: {e}")
        exit()

if not found_atlas:
    print("CRITICAL ERROR: ROI_MNI_V4.nii still not found after download.")
    exit()

aal_nii_path = found_atlas[0]
print(f"Using Atlas: {aal_nii_path}")


masker = maskers.NiftiLabelsMasker(
    labels_img=aal_nii_path,
    standardize="zscore_sample", 
    smoothing_fwhm=6,          
    low_pass=0.08,             
    high_pass=0.009,           
    t_r=2.0,                   
    detrend=True,
    verbose=0
)


def process_subject(row_data):
    site_path, participant_id, dx_raw, gender_raw, age_raw, iq_raw, tr_val = row_data
    
    try:
        sub_id = str(participant_id)
        
        formatted_sub_id = f"sub-{sub_id.zfill(7)}" if not sub_id.startswith('sub-') else sub_id
        
        
        subject_dir = os.path.join(site_path, formatted_sub_id)
        
        fmri_path = None
        if os.path.exists(subject_dir):
            
            bold_files = glob.glob(os.path.join(subject_dir, '**', '*bold.nii.gz'), recursive=True)
            if bold_files:
                fmri_path = bold_files[0] 
        
        if not fmri_path: return None

        
        label = 0
        val = str(dx_raw).lower()
        if any(x in val for x in ["typically", "control", "td", "0"]):
            label = 0
        elif any(x in val for x in ["adhd", "1", "2", "3"]):
            label = 1
        else:
            return None

        
        masker.t_r = tr_val 
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
           
            full_time_series = masker.fit_transform(fmri_path)

   
        n_time_points, n_regions = full_time_series.shape
        
        if n_time_points < TARGET_LENGTH:
            return None 
        
        final_time_series = full_time_series[:TARGET_LENGTH, :]

        
        if n_regions >= 90:
            final_time_series = final_time_series[:, :90]
        else:
            return None

        
        gender = 0 if str(gender_raw).lower() == 'male' else 1
        try: age = float(age_raw)
        except: age = 0.0
        try: iq = float(iq_raw)
        except: iq = 100.0
        if np.isnan(iq): iq = 100.0

        return {
            'time_series': final_time_series.astype(np.float32), 
            'pheno': [age, gender, iq], 
            'label': label,
            'site': os.path.basename(site_path)
        }

    except Exception as e:
        return None

if __name__ == "__main__":
    
    site_config = {
        'NYU': 2.0, 'NeuroIMAGE': 1.96, 'Peking_1': 2.0, 'Peking_2': 2.0, 'Peking_3': 2.0,
        'KKI': 2.5, 'OHSU': 2.5, 'WashU': 2.5, 'Pittsburgh': 1.5 
    }
    
    tasks = []
    
    print("Scanning dataset directories...")
    for site, tr in site_config.items():
        site_path = os.path.join(DATASET_ROOT, site)
        
        possible_phenos = [
            os.path.join(site_path, 'participants.tsv'),
            os.path.join(DATASET_ROOT, f"{site}_phenotypic.csv")
        ]
        
        pheno_file = None
        for p in possible_phenos:
            if os.path.exists(p):
                pheno_file = p
                break
        
        if not pheno_file: 
            print(f"Skipping {site}: No phenotypic file found.")
            continue
        
        try:
          
            if pheno_file.endswith('.tsv'):
                df = pd.read_csv(pheno_file, sep='\t', encoding_errors='replace')
            else:
                df = pd.read_csv(pheno_file, encoding_errors='replace')
                
            df.columns = [c.lower() for c in df.columns]
            
           
            target_col = next((c for c in ['dx', 'diagnosis', 'adhd_status', 'group'] if c in df.columns), None)
            # Find subject ID column (sometimes it's 'scanDir_ID' or 'Subject')
            id_col = next((c for c in ['participant_id', 'scandir_id', 'subject'] if c in df.columns), None)
            
            if not target_col or not id_col: 
                print(f"Skipping {site}: Columns missing in {os.path.basename(pheno_file)}")
                continue

            for _, row in df.iterrows():
                
                iq_val = np.nan 
                if 'iq' in df.columns: iq_val = row['iq']
                elif 'verbal_iq' in df.columns: iq_val = row['verbal_iq']

                
                gender_val = row['gender'] if 'gender' in df.columns else '0'

                tasks.append((
                    site_path, 
                    row[id_col], 
                    row[target_col], 
                    gender_val, 
                    row['age'] if 'age' in df.columns else 0, 
                    iq_val, 
                    tr
                ))
        except Exception as e:
            print(f"Error preparing {site}: {e}")

    print(f"Found {len(tasks)} potential scans.")
    print(f"Starting Processing (Target Length: {TARGET_LENGTH}, Atlas: AAL 90)...")
    
    results = Parallel(n_jobs=N_JOBS, backend='loky')(
        delayed(process_subject)(t) for t in tqdm(tasks, unit="scan")
    )

    valid_results = [r for r in results if r is not None]
    
    print(f"\nProcessing Complete!")
    print(f"Successfully processed: {len(valid_results)} subjects")
    
    np.save(OUTPUT_FILE, valid_results)
    print(f"Saved to {OUTPUT_FILE}")
    
    if len(valid_results) > 0:
        print(f"Shape check: {valid_results[0]['time_series'].shape}")
