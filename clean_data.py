import pandas as pd
import numpy as np
import dill

def clean_proc(proc):
    '''
    Clean 'procedures' data
    '''
    res = proc.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'])
    res = res.drop(columns=['ROW_ID', 'SEQ_NUM'])
    res = res.drop_duplicates().reset_index(drop=True)
    return res

def clean_medi(medi):
    '''
    Clean 'prescriptions' data
    '''
    res = medi[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE', 'NDC']]
    res = res[res.NDC != '0']
    res = res.fillna(method='pad').dropna().drop_duplicates()
    res['ICUSTAY_ID'] = res['ICUSTAY_ID'].astype('int64')
    res['STARTDATE'] = pd.to_datetime(res['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    res = res.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
    res = res.reset_index(drop=True)
    
    ## filter to first instance of presciption (by start date)
    groups = res.drop(columns=['NDC'])
    groups = groups.groupby(by=['SUBJECT_ID','HADM_ID','ICUSTAY_ID']).head(1).reset_index(drop=True)
    res = pd.merge(res, groups, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
    res = res.drop(columns=['STARTDATE', 'ICUSTAY_ID'])
    res = res.drop_duplicates().reset_index(drop=True)
    
    # filter to where subject visits > 1
    res = res.loc[res.groupby('SUBJECT_ID')['HADM_ID'].transform('nunique') > 1]
    res = res.reset_index(drop=True)
    return res

def clean_diag(diag):
    '''
    Clean 'diagnoses' data
    '''
    res = diag.dropna()
    res = res.drop(columns=['SEQ_NUM','ROW_ID'])
    res = res.drop_duplicates()
    res = res.sort_values(by=['SUBJECT_ID','HADM_ID']).reset_index(drop=True)
    return res

def ndc2atc4(medi, ndc2rxnorm, ndc2atc):
    '''
    Remap NDC -> RXCUI -> ATC4 (renamed to NDC)
    '''
    res = medi.copy()
    res['RXCUI'] = medi['NDC'].map(ndc2rxnorm)
    res = res.dropna()

    rx = ndc2atc.drop(columns=['YEAR','MONTH','NDC'])
    rx.drop_duplicates(subset=['RXCUI'], inplace=True)
    res = res[res['RXCUI'] != '']
    res['RXCUI'] = res['RXCUI'].astype('int64')
    
    res = pd.merge(res, rx, on=['RXCUI'])
    res = res.drop(columns=['NDC', 'RXCUI'])
    res = res.rename(columns={'ATC4':'NDC'})
    
    res['NDC'] = res['NDC'].map(lambda x: x[:4])
    res = res.drop_duplicates().reset_index(drop=True)  
    return res

def process_all(medi, diag, proc, ndc2rxnorm, ndc2atc):
    '''
    Function to clean all data
    '''
    ## initial cleaning
    medi_clean = clean_medi(medi)
    medi_clean = ndc2atc4(medi_clean, ndc2rxnorm, ndc2atc)
    diag_clean = clean_diag(diag)
    proc_clean = clean_proc(proc)
    
    ## filter to first 2k diagnoses
    codes = diag_clean.groupby('ICD9_CODE').size().reset_index(name='count')
    codes = codes.sort_values(by=['count'], ascending=False).reset_index(drop=True)
    codes_2k = codes.head(2000).ICD9_CODE
    diag_clean = diag_clean[diag_clean['ICD9_CODE'].isin(codes_2k)].reset_index(drop=True)
    
    ## create subjectid + admissionid key for all 3 datasets
    medi_key = medi_clean[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_key = diag_clean[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    proc_key = proc_clean[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    
    ## combine them (inner join) to get keys present in all datasets
    combined_key = pd.merge(medi_key, diag_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = pd.merge(combined_key, proc_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
    ## merge keys onto cleaned data
    diag_clean = diag_clean.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    medi_clean = medi_clean.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    proc_clean = proc_clean.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    
    ## flatten and merge
    diag_clean = diag_clean.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index()  
    medi_clean = medi_clean.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    proc_clean = proc_clean.groupby(by=['SUBJECT_ID','HADM_ID'])['ICD9_CODE'].unique().reset_index().rename(columns={'ICD9_CODE':'PRO_CODE'})  
    
    medi_clean['NDC'] = medi_clean['NDC'].map(lambda x: list(x))
    proc_clean['PRO_CODE'] = proc_clean['PRO_CODE'].map(lambda x: list(x))

    data = pd.merge(diag_clean, medi_clean, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = pd.merge(data, proc_clean, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    return data


def main():
    DATADIR = "data/"

    ## data files
    print("Reading in files...")
    medi = pd.read_csv(DATADIR + "PRESCRIPTIONS.csv", low_memory=False, dtype={'NDC':'category'})
    diag = pd.read_csv(DATADIR + "DIAGNOSES_ICD.csv", low_memory=False)
    proc = pd.read_csv(DATADIR + "PROCEDURES_ICD.csv", low_memory=False, dtype={'ICD9_CODE':'category'})

    ## mapping files
    with open(DATADIR + "ndc2rxnorm_mapping.txt", 'r') as f:
        ndc2rxnorm = eval(f.read())
    ndc2atc = pd.read_csv(DATADIR + "ndc2atc_level4.csv")

    ## process all data, save to pkl file
    print("Cleaning data...")
    data = process_all(medi, diag, proc, ndc2rxnorm, ndc2atc)
    print("Saving to data directory...")
    data.to_pickle(DATADIR + 'data_final.pkl')
    print("Done!")

if __name__ == '__main__':
    main()