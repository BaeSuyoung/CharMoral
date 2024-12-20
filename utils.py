import pandas as pd
import json
import os 
import io
import re
import tqdm
from zipfile import ZipFile
from collections import Counter
import pickle5 as pickle

#from retry import retry


import spacy
NER=spacy.load('en_core_web_sm')

from pycorenlp import StanfordCoreNLP

def load_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(input_path, datasets):
    with open(input_path, 'w') as outfile:
        json.dump(datasets, outfile)
        
def save_pickle(input_path, df):
    with open(input_path, 'wb') as f:
        pickle.dump(df, f)
        
def load_pickle(input_path):
    with open(input_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data

def token_len(documents):
    overview_len=[]
    for i in range(len(documents)):
        overview_len.append(len(documents[i].split(" ")))
    print("average token length: ", sum(overview_len)/len(overview_len))
    print("min token length: ", min(overview_len), "\nmax token length: ", max(overview_len))
    print(overview_len[:5])
    return overview_len

def preprocess(documents):
    documents=documents.str.replace(r'\([^()]*\)', '', regex=True)
    documents=documents.str.replace("Mr. ", "")
    documents=documents.str.replace("Mrs. ", "")
    documents=documents.str.replace("Dr. ", "")
    documents=documents.str.replace('--', ' ')
    documents=documents.str.replace('-', ' ')
    documents=documents.str.replace('[', ' ')
    documents=documents.str.replace(']', ' ')
    documents=documents.str.replace('*', '')
    documents=documents.str.replace(':', '.')
    documents=documents.str.replace(';', '.')
    documents=documents.str.replace('_', '')
    documents=documents.str.replace('\n', ' ')
    documents=documents.str.replace('  ', ' ')

    return documents

def resolve_coreferences(text):
    
    final_output=""
    nlp = StanfordCoreNLP('http://localhost:9000')

    # len split
    segment_texts=[]
    text_list=text.split(".")
    for i in range(0, len(text_list), 50):
        merged_element = ".".join(map(str, text_list[i:i + 50]))
        segment_texts.append(merged_element)

    for seg in segment_texts:
    
        output = nlp.annotate(str(seg), properties={
            'annotators': 'coref',
            'outputFormat': 'json'
        })

        sentences=[]
        for out_sents in output['sentences']:
            sentence=[]
            for out_sent in out_sents['tokens']:
                sentence.append(out_sent['word'])
            sentences.append(sentence)

        #print(sentences)
        for cluster in output['corefs'].values():
            main_mention = cluster[0]['text']

            #print(main_mention)
            
            #print(cluster)
            for mention in cluster[1:]:
                sent_num=mention['sentNum']-1 #1

                start_idx = mention['startIndex'] -1
                end_idx = mention['endIndex'] -1
                
                #print(start_idx, end_idx, main_mention)
                #print("before: ", sentences[sent_num])
                sentences[sent_num][start_idx]=main_mention

                if start_idx != (end_idx-1) : 
                    for i in range(start_idx+1, end_idx):
                        sentences[sent_num][i]=''
                #print("after", sentences[sent_num])
        
        sentences =" ".join([" ".join(x) for x in sentences])
        final_output+=sentences+" "
    #print(final_output)
    return final_output

def character_replacement(txt):
    doc = NER(txt)
    char_set=set()
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            char_set.add(ent.text)
    
    for char in list(char_set):
        
        txt=txt.replace(char, "["+char.replace(" ", "_")+"]")

    return txt

def find_pattern_in_text(text):
    pattern = r'\[[a-zA-Z_]+\]'
    matches = re.findall(pattern, text)
    matches=[x[1:-1] for x in matches]
    return matches


def merge_segments_with_characters(segment_list, character_list):
    merged_segments = []
    merged_characters=[]
    current_segment = ""
    current_character = None
    for segment, character in zip(segment_list, character_list):
        if current_character is None:
            current_segment = segment
            current_character = character
        elif current_character == character:
            current_segment += " " + segment
        else:
            merged_segments.append(current_segment)
            merged_characters.append(current_character)
            current_segment = segment
            current_character = character

    if current_segment:
        merged_segments.append(current_segment)
        merged_characters.append(current_character)

    assert len(merged_segments) == len(merged_characters)

    return merged_segments, merged_characters