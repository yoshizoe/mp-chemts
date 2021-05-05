
import csv
import itertools
import operator
import numpy as np
import nltk
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

def get_val():
    sen_space=split_smiles()
    all_smile=[]
    length=[]
    end="\n"
    element_table=["C","N","B","O","P","S","F","Cl","Br","I","(",")","=","#","Si"]
    ring=["1","2","3","4","5","6","7","8","9","10"]
    for i in range(len(sen_space)):
        word_space=sen_space[i]
        word=[]
        j=0
        while j<len(word_space):
            word_space1=[]
            if word_space[j]=="[":
                word_space1.append(word_space[j])
                j=j+1
                while word_space[j]!="]":
                    word_space1.append(word_space[j])
                    j=j+1
                word_space1.append(word_space[j])
                word_space2=''.join(word_space1)


                word.append(word_space2)
                j=j+1
            else:
                word_space1.append(word_space[j])

                if j+1<len(word_space):
                    word_space1.append(word_space[j+1])
                    word_space2=''.join(word_space1)
                else:
                    word_space1.insert(0,word_space[j-1])
                    word_space2=''.join(word_space1)

                if word_space2 not in element_table:
                    word.append(word_space[j])
                    j=j+1
                else:
                    word.append(word_space2)
                    j=j+2
        word.append(end)
        word.insert(0,"&")
        len1=len(word)
        length.append(len1)
        if '[SiH2]' not in list(word):
            if '[SiH3]' not in list(word):
                if '[SiH]' not in list(word):
                    all_smile.append(list(word))

    after_all_smile=all_smile
    val=["\n"]
    delid=[]
    all_smile_go=[]
    for i in range(len(after_all_smile)):
        for j in range(len(after_all_smile[i])):
            if after_all_smile[i][j] not in val:
                val.append(after_all_smile[i][j])

    return val, max(length)


def zinc_logp(smile):
    logp_value=[]
    compound=[]
    for i in range(len(smile)):
        m = Chem.MolFromSmiles(smile[i])
        if m!=None:
            compound.append(smile[i])
            logp=Descriptors.MolLogP(m)
            logp_value.append(logp)
    return compound


def split_smiles():
    sen_space=[]
    f = open('data/250k_rndm_zinc_drugs_clean.smi', 'r')
    reader = csv.reader(f)
    for row in reader:
        sen_space.append(row)
    f.close()
    word1=sen_space
    end="\n"
    zinc_processed=[]
    organic_smile=[]
    t=0
    for i in range(len(sen_space)):
        word1=sen_space[i]
        if word1!=[]:
            zinc_processed.append(word1[0])
    return zinc_processed
