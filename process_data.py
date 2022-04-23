import pandas as pd
import numpy as np
import dill
from collections import defaultdict

### Functions for cleaning MIMIC data 

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



### Functions for creating Vocab for medical codes, as well as to create patient records

class Voc(object):
    '''
    Vocabulary Class
    idx2word: all unique words (medical codes) with corresponding unique IDs
    word2idx: reverse of idx2word, unqiue IDs as keys with medical codes as values
    '''
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_code(self, code):
        if code not in self.word2idx:
            self.idx2word[len(self.word2idx)] = code
            self.word2idx[code] = len(self.word2idx)
                
def create_vocab(df, data_path ="data/", filename="voc_final.pkl"):
    '''
    Create vocabulary from cleaned mimic3 medical codes
    '''
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()
    
    ## get unique codes for each type of code
    icd9_codes = df['ICD9_CODE'].explode().unique()
    ndc_codes = df['NDC'].explode().unique()
    pro_codes = df['PRO_CODE'].explode().unique()
    
    ## create code mapping
    for icd9 in icd9_codes: 
        diag_voc.add_code(icd9)
    for ndc in ndc_codes: 
        med_voc.add_code(ndc)    
    for pro in pro_codes: 
        pro_voc.add_code(pro)
    
    dill.dump(obj={'diag_voc':diag_voc, 'med_voc':med_voc ,'pro_voc':pro_voc}, 
              file=open(data_path + filename, 'wb'))
    return diag_voc, med_voc, pro_voc

def create_patient_record(df, diag_voc, med_voc, pro_voc, data_path ="data/", filename="records_final.pkl"):
    '''
    Create a list of patient records, where each entry contains
    medical codes corresponding to each patient for each visit
    '''
    records = []
    ## loop through each patient
    for subject_id in df['SUBJECT_ID'].unique():
        
        ## filter to rows with that patient
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        
        ## each row correspond to an admission record, append each type of code to admission list
        for index, row in item_df.iterrows():
            admission = []
            admission.append([diag_voc.word2idx[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2idx[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2idx[i] for i in row['NDC']])
            
            ## append list of codes per admission to patient list
            patient.append(admission)
        ## append patient list to records list
        records.append(patient) 
    dill.dump(obj=records, file=open(data_path + 'records_final.pkl', 'wb'))
    return records


### Functions for creating adjacency matrices

def create_atc_map(med_voc):
    '''
    Create ATC map from medical code vocab
    '''
    med_voc_size = len(med_voc.idx2word)
    med_unique_word = [med_voc.idx2word[i] for i in range(med_voc_size)]
    atc_map = defaultdict(set)
    for item in med_unique_word:
        atc_map[item[:4]].add(item)
    return atc_map

def create_cid2atc_map(datafile, atc_map):
    '''
    Create cid code -> atc mapping from drug cid file and atc mapping
    '''
    cid2atc = defaultdict(set)
    with open(datafile, 'r') as f:
        for line in f:
            line_ls = line[:-1].split(',')
            cid = line_ls[0]
            atcs = line_ls[1:]
            for atc in atcs:
                if len(atc_map[atc[:4]]) != 0:
                    cid2atc[cid].add(atc[:4])
    return cid2atc

def clean_ddi(ddi, topk=40):
    '''
    Clean DDI (drug-drug interactions) file, filter to top K ddi
    '''
    # fliter to severe side effects 
    ddi_most_pd = ddi.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index(name='count')
    ddi_most_pd = ddi_most_pd.sort_values(by=['count'], ascending=False).reset_index(drop=True)
    ddi_most_pd = ddi_most_pd.iloc[-topk:, :]

    fliter_ddi_df = ddi.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
    ddi_df = fliter_ddi_df[['STITCH 1','STITCH 2']].drop_duplicates().reset_index(drop=True)
    return ddi_df


def normalize(adj):
    '''
    Row-normalization of adjacency matrix (ehr/ddi)
    corresponding to individual normalization of the feature vector of each sample
    Keeps activations on similar levels, helps optimization.
    https://github.com/tkipf/gcn/issues/35
    '''
    rowsum = np.array(adj.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diagflat(r_inv)
    adj = r_mat_inv.dot(adj)
    return adj

def create_ehr_adj(records, size, savepath):
    '''
    Create weighted EHR adjacency matrix
    '''
    ehr_adj = np.zeros((size, size))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    ehr_adj[med_i, med_j] = 1
                    ehr_adj[med_j, med_i] = 1

    ## add self connection, normalize row
    ehr_adj += np.eye(ehr_adj.shape[0])
    ehr_adj = normalize(ehr_adj)

    dill.dump(ehr_adj, open(savepath+"ehr_adj_normalized.pkl", 'wb'))
    return ehr_adj

def create_ddi_adj(ddi_df, med_voc, cid2atc, atc_map, size, savepath):
    '''
    Create DDI adjacency matrix
    '''
    ddi_adj = np.zeros((size, size))
    for index, row in ddi_df.iterrows():
        # ddi
        cid1 = row['STITCH 1']
        cid2 = row['STITCH 2']
        
        # cid -> atc_level3
        for atc_i in cid2atc[cid1]:
            for atc_j in cid2atc[cid2]:
                
                # atc_level3 -> atc_level4
                for i in atc_map[atc_i]:
                    for j in atc_map[atc_j]:
                        if med_voc.word2idx[i] != med_voc.word2idx[j]:
                            ddi_adj[med_voc.word2idx[i], med_voc.word2idx[j]] = 1
                            ddi_adj[med_voc.word2idx[j], med_voc.word2idx[i]] = 1
    ## add self connection, normalize rows
    dill.dump(ddi_adj, open(savepath+"ddi_A.pkl", 'wb')) 
    ddi_adj += np.eye(ddi_adj.shape[0])
    ddi_adj = normalize(ddi_adj)

    dill.dump(ddi_adj, open(savepath+"ddi_A_normalized.pkl", 'wb')) 
    return ddi_adj



def main():
    DATADIR = "data/"

    ## data files
    print("Reading in MIMIC and mapping files...")
    medi = pd.read_csv(DATADIR + "PRESCRIPTIONS.csv", low_memory=False, dtype={'NDC':'category'})
    diag = pd.read_csv(DATADIR + "DIAGNOSES_ICD.csv", low_memory=False)
    proc = pd.read_csv(DATADIR + "PROCEDURES_ICD.csv", low_memory=False, dtype={'ICD9_CODE':'category'})

    ## mapping files
    with open(DATADIR + "ndc2rxnorm_mapping.txt", 'r') as f:
        ndc2rxnorm = eval(f.read())
    ndc2atc = pd.read_csv(DATADIR + "ndc2atc_level4.csv")

    ## process all data, save to pkl file
    print("Processing MIMIC data...")
    data = process_all(medi, diag, proc, ndc2rxnorm, ndc2atc)
    print("Saving cleaned MIMIC data to data directory...")
    data.to_pickle(DATADIR + 'data_final.pkl')
    
    ## create vocab and patient records
    print("Creating vocab and patient records and saving to data directory...")
    diag_voc, med_voc, pro_voc = create_vocab(data)
    records = create_patient_record(data, diag_voc, med_voc, pro_voc)

    ## Creating atc code mappings and cid -> atc code mappings
    print("Creating atc code mappings and cid -> atc code mappings")
    atc_map = create_atc_map(med_voc)
    cid2atc = create_cid2atc_map(DATADIR + "drug-atc.csv", atc_map)

    ## Reading and cleaning DDI (drug-drug interaction) file
    print("Reading and cleaning DDI file...")
    ddi = pd.read_csv(DATADIR + "drug-DDI.csv")
    ddi_df = clean_ddi(ddi)

    ## Creating EHR adjacency matrix
    print("Creating EHR adjacency matrix...")
    ehr_adj = create_ehr_adj(records, size=len(med_voc.idx2word), savepath=DATADIR)

    ## Creating DDI adjacency matrix
    print("Creating DDI adjacency matrix...")
    ddi_adj = create_ddi_adj(ddi_df, med_voc, cid2atc, atc_map, 
                             size=len(med_voc.idx2word), savepath=DATADIR)
    print("Done!")

if __name__ == '__main__':
    main()