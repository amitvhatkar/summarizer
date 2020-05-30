import os
import sys
import datetime
import math 

import spacy
import textacy
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import pandas as pd 
import numpy as np
import string
from gensim.summarization.summarizer import summarize
import re
from rouge import Rouge
import json

import datetime
import pickle
nlp = spacy.load('en')
stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

f = open('all_summarize_generate_lookup_log.txt', 'w')#open('validation_chk_log.txt','w')
sys.stdout = f

DICT_REDUCTION_FACTOR = 0.65
TEXT_REDUCTION_FACTOR = 1

# TEXT_DATA_LOCATION = "../Dataset/training/annual_reports/"
# SUMMARY_DATA_LOCATION = "../Dataset/training/gold_summaries/"

# VALIDATION_TEXT_PATH = "../Dataset/validation/annual_reports/"
# VALIDATION_SUMMARY_PATH = "../Dataset/validation/gold_summaries/"

# TEST_TEXT_PATH = '../Dataset/testing/annual_reports/'

LOWER_WORD_LIMIT = 0#500
UPPER_WORD_LIMIT = 1000
# CLAUSE = f"complte_sum_w_enc_{DICT_REDUCTION_FACTOR}_uni"

# def generate_rouge_score(system_summary, reference_summary):

# 	if len(reference_summary) > 50000:
# 		reference_summary = reference_summary[:50000]

# 	if len(system_summary) > 200000 and ((len(reference_summary) == len(system_summary))):
# 		print("score cant be computed", len(KG_fold_summary), len(reference_summary))
# 		return -1

# 	rouge = Rouge()	
# 	try:
# 		rouge_score = rouge.get_scores(system_summary, reference_summary)
# 	except Exception as e:
# 		print("More than one sentence Exception: Line 57", e, len(system_summary.split(' ')), "-----------------------------------------",len(reference_summary.split(' ')))
# 		return -1

# 	return rouge_score

# def generate_result_xlsx(fold_number, summary_file_name_list, is_intermediate, is_all_or = False):
	
# 	temp_list= [file_name.split("_")[0] for file_name in os.listdir(f"./Folds/Fold_{fold_number}/Result/KG_file/")]
# 	summary_file_name_list = list(set(temp_list))
# 	print(len(temp_list), len(summary_file_name_list), summary_file_name_list)

# 	if is_all_or:
# 		f = open(f'./logs/all_or_result_{CLAUSE}_generation_log.txt', 'w')#open('validation_chk_log.txt','w')
# 	else:
# 		f = open(f'./logs/result_{CLAUSE}_generation_log.txt', 'w')#open('validation_chk_log.txt','w')
# 	sys.stdout = f

# 	# print(summary_file_name_list[:10])

# 	file_name_list = []
# 	'''KG TR'''
# 	tr_rouge_KG_f = []
# 	tr_rouge_KG_p = [] 
# 	tr_rouge_KG_r = []

# 	tr_rouge_KG_fl = []
# 	tr_rouge_KG_pl = [] 
# 	tr_rouge_KG_rl = []

# 	tr_rouge_KG_f2 = []
# 	tr_rouge_KG_p2 = [] 
# 	tr_rouge_KG_r2 = []

# 	'''KG OR'''
# 	or_rouge_KG_f = []
# 	or_rouge_KG_p = []
# 	or_rouge_KG_r = []

# 	or_rouge_KG_fl = []
# 	or_rouge_KG_pl = []
# 	or_rouge_KG_rl = []

# 	or_rouge_KG_f2 = []
# 	or_rouge_KG_p2 = []
# 	or_rouge_KG_r2 = []


# 	'''UKG TR'''
# 	tr_rouge_KG_U_f = []
# 	tr_rouge_KG_U_p = []
# 	tr_rouge_KG_U_r = []

# 	tr_rouge_KG_U_fl = []
# 	tr_rouge_KG_U_pl = []
# 	tr_rouge_KG_U_rl = []

# 	tr_rouge_KG_U_f2 = []
# 	tr_rouge_KG_U_p2 = []
# 	tr_rouge_KG_U_r2 = []

# 	'''UKG OR'''
# 	or_rouge_KG_U_f = []
# 	or_rouge_KG_U_p = []
# 	or_rouge_KG_U_r = []

# 	or_rouge_KG_U_fl = []
# 	or_rouge_KG_U_pl = []
# 	or_rouge_KG_U_rl = []

# 	or_rouge_KG_U_f2 = []
# 	or_rouge_KG_U_p2 = []
# 	or_rouge_KG_U_r2 = []


# 	'''TR OR'''
# 	tr_rouge_OR_f = []
# 	tr_rouge_OR_p = []
# 	tr_rouge_OR_r = []

# 	tr_rouge_OR_fl = []
# 	tr_rouge_OR_pl = []
# 	tr_rouge_OR_rl = []

# 	tr_rouge_OR_f2 = []
# 	tr_rouge_OR_p2 = []
# 	tr_rouge_OR_r2 = []	
	

# 	'''OR TR'''
# 	or_rouge_tr_f = []
# 	or_rouge_tr_p = []
# 	or_rouge_tr_r = []

# 	or_rouge_tr_fl = []
# 	or_rouge_tr_pl = []
# 	or_rouge_tr_rl = []

# 	or_rouge_tr_f2 = []
# 	or_rouge_tr_p2 = []
# 	or_rouge_tr_r2 = []

# 	# max_count_list = []
# 	# num_triple_list = []
# 	# num_sent_list = []
# 	# svo_count_dict_list = []
# 	# svo_count_dict_updated_list = []

# 	# KG_fold_summary_list = []
# 	'''FKG TR'''
# 	tr_rouge_Fold_KG_list_f = []
# 	tr_rouge_Fold_KG_list_p = []
# 	tr_rouge_Fold_KG_list_r = []
	
# 	tr_rouge_Fold_KG_list_fl = []
# 	tr_rouge_Fold_KG_list_pl = []
# 	tr_rouge_Fold_KG_list_rl = []

# 	tr_rouge_Fold_KG_list_f2 = []
# 	tr_rouge_Fold_KG_list_p2 = []
# 	tr_rouge_Fold_KG_list_r2 = []


# 	'''FKG OR'''
# 	or_rouge_Fold_KG_list_f = []
# 	or_rouge_Fold_KG_list_p = []
# 	or_rouge_Fold_KG_list_r = []

# 	or_rouge_Fold_KG_list_fl = []
# 	or_rouge_Fold_KG_list_pl = []
# 	or_rouge_Fold_KG_list_rl = []

# 	or_rouge_Fold_KG_list_f2 = []
# 	or_rouge_Fold_KG_list_p2 = []
# 	or_rouge_Fold_KG_list_r2 = []

# 	fold_svo_count_dict_list = []
# 	# return

# 	print(f"**************************Starting scoring {datetime.datetime.now()}**************************")

# 	if LOWER_WORD_LIMIT != -1:

# 		if is_all_or:
# 			result_file_path = f"./Folds_Complete/Fold_{fold_number}/{fold_number}_all_or_result_{LOWER_WORD_LIMIT}_{UPPER_WORD_LIMIT}_{CLAUSE}.xlsx"
# 		else:
# 			result_file_path = f"./Folds_Complete/Fold_{fold_number}/{fold_number}_result_{LOWER_WORD_LIMIT}_{UPPER_WORD_LIMIT}_{CLAUSE}.xlsx"
# 	else:
# 		if is_all_or:
# 			result_file_path = f"./Folds_Complete/Fold_{fold_number}/{fold_number}_all_or_result_{CLAUSE}.xlsx"
# 		else:
# 			result_file_path = f"./Folds_Complete/Fold_{fold_number}/{fold_number}_result_{CLAUSE}.xlsx"

# 	if not os.path.isfile(result_file_path):
# 		file = open(result_file_path, 'w+')
# 		file.close()
# 		is_intermediate = False
# 	else:
# 		is_intermediate = True

# 	if fold_number == -1:
# 		base_summary_location = VALIDATION_SUMMARY_PATH
# 	else:
# 		base_summary_location = SUMMARY_DATA_LOCATION
	
# 	df = pd.DataFrame(list(zip(file_name_list,\
# 			tr_rouge_KG_f, tr_rouge_KG_p, tr_rouge_KG_r,\
# 			tr_rouge_KG_fl, tr_rouge_KG_pl, tr_rouge_KG_rl,\
# 			tr_rouge_KG_f2, tr_rouge_KG_p2, tr_rouge_KG_r2,\

# 			or_rouge_tr_f, or_rouge_tr_p, or_rouge_tr_r,\
# 			or_rouge_tr_fl, or_rouge_tr_pl, or_rouge_tr_rl,\
# 			or_rouge_tr_f2, or_rouge_tr_p2, or_rouge_tr_r2,\

# 			tr_rouge_KG_U_f, tr_rouge_KG_U_p, tr_rouge_KG_U_r,\
# 			tr_rouge_KG_U_fl, tr_rouge_KG_U_pl, tr_rouge_KG_U_rl,\
# 			tr_rouge_KG_U_f2, tr_rouge_KG_U_p2, tr_rouge_KG_U_r2,\

# 			or_rouge_KG_U_f, or_rouge_KG_U_p, or_rouge_KG_U_r,\
# 			or_rouge_KG_U_fl, or_rouge_KG_U_pl, or_rouge_KG_U_rl,\
# 			or_rouge_KG_U_f2, or_rouge_KG_U_p2, or_rouge_KG_U_r2,\

# 			tr_rouge_Fold_KG_list_f, tr_rouge_Fold_KG_list_p, tr_rouge_Fold_KG_list_r,\
# 			tr_rouge_Fold_KG_list_fl, tr_rouge_Fold_KG_list_pl, tr_rouge_Fold_KG_list_rl,\
# 			tr_rouge_Fold_KG_list_f2, tr_rouge_Fold_KG_list_p2, tr_rouge_Fold_KG_list_r2,\

# 			or_rouge_Fold_KG_list_f, or_rouge_Fold_KG_list_p, or_rouge_Fold_KG_list_r,\
# 			or_rouge_Fold_KG_list_fl, or_rouge_Fold_KG_list_pl, or_rouge_Fold_KG_list_rl,\
# 			or_rouge_Fold_KG_list_f2, or_rouge_Fold_KG_list_p2, or_rouge_Fold_KG_list_r2,\


# 			tr_rouge_OR_f, tr_rouge_OR_p, tr_rouge_OR_r,\
# 			tr_rouge_OR_fl, tr_rouge_OR_pl, tr_rouge_OR_rl,\
# 			tr_rouge_OR_f2, tr_rouge_OR_p2, tr_rouge_OR_r2,\

# 			or_rouge_tr_f, or_rouge_tr_p, or_rouge_tr_r,\
# 			or_rouge_tr_fl, or_rouge_tr_pl, or_rouge_tr_rl,\
# 			or_rouge_tr_f2, or_rouge_tr_p2, or_rouge_tr_r2

# 			)),	columns =["File_Name",\
# 			"KG_wrt_TR_F", "KG_wrt_TR_P", "KG_wrt_TR_R",\
# 			"KG_wrt_TR_Fl", "KG_wrt_TR_Pl", "KG_wrt_TR_Rl",\
# 			"KG_wrt_TR_F2", "KG_wrt_TR_P2", "KG_wrt_TR_R2",\

# 			"KG_wrt_OR_F", "KG_wrt_OR_P", "KG_wrt_OR_R",\
# 			"KG_wrt_OR_Fl", "KG_wrt_OR_Pl", "KG_wrt_OR_Rl",\
# 			"KG_wrt_OR_F2", "KG_wrt_OR_P2", "KG_wrt_OR_R2",\
			
# 			"UKG_wrt_TR_F", "UKG_wrt_TR_P", "UKG_wrt_TR_R",\
# 			"UKG_wrt_TR_Fl", "UKG_wrt_TR_Pl", "UKG_wrt_TR_Rl",\
# 			"UKG_wrt_TR_F2", "UKG_wrt_TR_P2", "UKG_wrt_TR_R2",\
			
# 			"UKG_wrt_OR_F", "UKG_wrt_OR_P", "UKG_wrt_OR_R",\
# 			"UKG_wrt_OR_Fl", "UKG_wrt_OR_Pl", "UKG_wrt_OR_Rl",\
# 			"UKG_wrt_OR_F2", "UKG_wrt_OR_P2", "UKG_wrt_OR_R2",\

# 			"FKG_wrt_TR_F", "FKG_wrt_TR_P", "FKG_wrt_TR_R",\
# 			"FKG_wrt_TR_Fl", "FKG_wrt_TR_Pl", "FKG_wrt_TR_Rl",\
# 			"FKG_wrt_TR_F2", "FKG_wrt_TR_P2", "FKG_wrt_TR_R2",\

# 			"FKG_wrt_OR_F", "FKG_wrt_OR_P", "FKG_wrt_OR_R",\
# 			"FKG_wrt_OR_Fl", "FKG_wrt_OR_Pl", "FKG_wrt_OR_Rl",\
# 			"FKG_wrt_OR_F2", "FKG_wrt_OR_P2", "FKG_wrt_OR_R2",\

# 			"TR_wrt_OR_F", "TR_wrt_OR_P", "TR_wrt_OR_R",\
# 			"TR_wrt_OR_Fl", "TR_wrt_OR_Pl", "TR_wrt_OR_Rl",\
# 			"TR_wrt_OR_F2", "TR_wrt_OR_P2", "TR_wrt_OR_R2",\

# 			"OR_wrt_TR_F", "OR_wrt_TR_P", "OR_wrt_TR_R",\
# 			"OR_wrt_TR_Fl", "OR_wrt_TR_Pl", "OR_wrt_TR_Rl",\
# 			"OR_wrt_TR_F2", "OR_wrt_TR_P2", "OR_wrt_TR_R2"
# 		])
	
# 	if not is_intermediate:
# 		df.to_excel(result_file_path)
# 		# is_intermediate = True
# 	else:
# 		# print("REsult file exists", os.path.exists(f"./Folds/Fold_{fold_number}/{fold_number}_result.xlsx"))
# 		if os.path.exists(result_file_path):
# 			print(result_file_path)
# 			temp_df = pd.read_excel(result_file_path, sheet_name="Sheet1")
# 			df = df.append(temp_df, sort=True, ignore_index = True)
# 			# print(len(df), len(temp_df))

# 	for index, file_name in enumerate(summary_file_name_list):


# 		if not os.path.exists(f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/KG_fold/{file_name.split('.')[0]}_KG_fold_summary.txt"):
# 			print("File not exists", file_name, f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/KG_fold/{file_name.split('.')[0]}_KG_fold_summary.txt")
# 			continue

# 		# print(len(df) > 0,end file_name in df.File_Name, file_name, df.File_Name)
		
# 		if len(df) > 0 and file_name in df['File_Name'].tolist():
# 			print("Continueing...", file_name)
# 			# break
# 			continue

# 		# print("\t\t\tLine 139", file_name)
# 		file_name = file_name.split('.')[0]
# 		if (index+1) % 20 == 0:
# 			print(f"Fold {fold_number} end time {datetime.datetime.now()}")
# 			print(f"Averaged Result: {fold_number}  :  {len(tr_rouge_KG_f)}")
# 			print('KG TR')
# 			print(round(np.average(tr_rouge_KG_f), 4), round(np.average(tr_rouge_KG_p), 4), round(np.average(tr_rouge_KG_r), 4))
# 			print(round(np.average(tr_rouge_KG_fl), 4), round(np.average(tr_rouge_KG_pl), 4), round(np.average(tr_rouge_KG_rl), 4))
# 			print(round(np.average(tr_rouge_KG_f2), 4), round(np.average(tr_rouge_KG_p2), 4), round(np.average(tr_rouge_KG_r2), 4))
# 			print('KG OR')
# 			print(round(np.average(or_rouge_tr_f), 4), round(np.average(or_rouge_tr_p), 4), round(np.average(or_rouge_tr_r), 4))
# 			print(round(np.average(or_rouge_tr_fl), 4), round(np.average(or_rouge_tr_pl), 4), round(np.average(or_rouge_tr_rl), 4))
# 			print(round(np.average(or_rouge_tr_f2), 4), round(np.average(or_rouge_tr_p2), 4), round(np.average(or_rouge_tr_r2), 4))
# 			print('UKG TR')
# 			print(round(np.average(tr_rouge_KG_U_f), 4), round(np.average(tr_rouge_KG_U_p), 4), round(np.average(tr_rouge_KG_U_r), 4))
# 			print(round(np.average(tr_rouge_KG_U_fl), 4), round(np.average(tr_rouge_KG_U_pl), 4), round(np.average(tr_rouge_KG_U_rl), 4))
# 			print(round(np.average(tr_rouge_KG_U_f2), 4), round(np.average(tr_rouge_KG_U_p2), 4), round(np.average(tr_rouge_KG_U_r2), 4))
# 			print('UKG OR')
# 			print(round(np.average(or_rouge_KG_U_f), 4), round(np.average(or_rouge_KG_U_p), 4), round(np.average(or_rouge_KG_U_r), 4))
# 			print(round(np.average(or_rouge_KG_U_fl), 4), round(np.average(or_rouge_KG_U_pl), 4), round(np.average(or_rouge_KG_U_rl), 4))
# 			print(round(np.average(or_rouge_KG_U_f2), 4), round(np.average(or_rouge_KG_U_p2), 4), round(np.average(or_rouge_KG_U_r2), 4))
# 			print('FKG TR')
# 			print(round(np.average(tr_rouge_Fold_KG_list_f), 4), round(np.average(tr_rouge_Fold_KG_list_p), 4), round(np.average(tr_rouge_Fold_KG_list_r), 4))
# 			print(round(np.average(tr_rouge_Fold_KG_list_fl), 4), round(np.average(tr_rouge_Fold_KG_list_pl), 4), round(np.average(tr_rouge_Fold_KG_list_rl), 4))
# 			print(round(np.average(tr_rouge_Fold_KG_list_f2), 4), round(np.average(tr_rouge_Fold_KG_list_p2), 4), round(np.average(tr_rouge_Fold_KG_list_r2), 4))
# 			print('FKG OR')
# 			print(round(np.average(or_rouge_Fold_KG_list_f), 4), round(np.average(or_rouge_Fold_KG_list_p), 4), round(np.average(or_rouge_Fold_KG_list_r), 4))
# 			print(round(np.average(or_rouge_Fold_KG_list_fl), 4), round(np.average(or_rouge_Fold_KG_list_pl), 4), round(np.average(or_rouge_Fold_KG_list_rl), 4))
# 			print(round(np.average(or_rouge_Fold_KG_list_f2), 4), round(np.average(or_rouge_Fold_KG_list_p2), 4), round(np.average(or_rouge_Fold_KG_list_r2), 4))
# 			print('TR OR')
# 			print(round(np.average(tr_rouge_OR_f), 4), round(np.average(tr_rouge_OR_p), 4), round(np.average(tr_rouge_OR_r), 4))
# 			print(round(np.average(tr_rouge_OR_fl), 4), round(np.average(tr_rouge_OR_pl), 4), round(np.average(tr_rouge_OR_rl), 4))
# 			print(round(np.average(tr_rouge_OR_f2), 4), round(np.average(tr_rouge_OR_p2), 4), round(np.average(tr_rouge_OR_r2), 4))
# 			print('OR TR')
# 			print(round(np.average(or_rouge_tr_f), 4), round(np.average(or_rouge_tr_p), 4), round(np.average(or_rouge_tr_r), 4))
# 			print(round(np.average(or_rouge_tr_fl), 4), round(np.average(or_rouge_tr_pl), 4), round(np.average(or_rouge_tr_rl), 4))
# 			print(round(np.average(or_rouge_tr_f2), 4), round(np.average(or_rouge_tr_p2), 4), round(np.average(or_rouge_tr_r2), 4))
# 			print("\n\n\n")


# 			df = pd.DataFrame(list(zip(file_name_list,\
# 				tr_rouge_KG_f, tr_rouge_KG_p, tr_rouge_KG_r,\
# 				tr_rouge_KG_fl, tr_rouge_KG_pl, tr_rouge_KG_rl,\
# 				tr_rouge_KG_f2, tr_rouge_KG_p2, tr_rouge_KG_r2,\

# 				or_rouge_tr_f, or_rouge_tr_p, or_rouge_tr_r,\
# 				or_rouge_tr_fl, or_rouge_tr_pl, or_rouge_tr_rl,\
# 				or_rouge_tr_f2, or_rouge_tr_p2, or_rouge_tr_r2,\

# 				tr_rouge_KG_U_f, tr_rouge_KG_U_p, tr_rouge_KG_U_r,\
# 				tr_rouge_KG_U_fl, tr_rouge_KG_U_pl, tr_rouge_KG_U_rl,\
# 				tr_rouge_KG_U_f2, tr_rouge_KG_U_p2, tr_rouge_KG_U_r2,\

# 				or_rouge_KG_U_f, or_rouge_KG_U_p, or_rouge_KG_U_r,\
# 				or_rouge_KG_U_fl, or_rouge_KG_U_pl, or_rouge_KG_U_rl,\
# 				or_rouge_KG_U_f2, or_rouge_KG_U_p2, or_rouge_KG_U_r2,\

# 				tr_rouge_Fold_KG_list_f, tr_rouge_Fold_KG_list_p, tr_rouge_Fold_KG_list_r,\
# 				tr_rouge_Fold_KG_list_fl, tr_rouge_Fold_KG_list_pl, tr_rouge_Fold_KG_list_rl,\
# 				tr_rouge_Fold_KG_list_f2, tr_rouge_Fold_KG_list_p2, tr_rouge_Fold_KG_list_r2,\

# 				or_rouge_Fold_KG_list_f, or_rouge_Fold_KG_list_p, or_rouge_Fold_KG_list_r,\
# 				or_rouge_Fold_KG_list_fl, or_rouge_Fold_KG_list_pl, or_rouge_Fold_KG_list_rl,\
# 				or_rouge_Fold_KG_list_f2, or_rouge_Fold_KG_list_p2, or_rouge_Fold_KG_list_r2,\


# 				tr_rouge_OR_f, tr_rouge_OR_p, tr_rouge_OR_r,\
# 				tr_rouge_OR_fl, tr_rouge_OR_pl, tr_rouge_OR_rl,\
# 				tr_rouge_OR_f2, tr_rouge_OR_p2, tr_rouge_OR_r2,\

# 				or_rouge_tr_f, or_rouge_tr_p, or_rouge_tr_r,\
# 				or_rouge_tr_fl, or_rouge_tr_pl, or_rouge_tr_rl,\
# 				or_rouge_tr_f2, or_rouge_tr_p2, or_rouge_tr_r2

# 				)),	columns =["File_Name",\
# 				"KG_wrt_TR_F", "KG_wrt_TR_P", "KG_wrt_TR_R",\
# 				"KG_wrt_TR_Fl", "KG_wrt_TR_Pl", "KG_wrt_TR_Rl",\
# 				"KG_wrt_TR_F2", "KG_wrt_TR_P2", "KG_wrt_TR_R2",\

# 				"KG_wrt_OR_F", "KG_wrt_OR_P", "KG_wrt_OR_R",\
# 				"KG_wrt_OR_Fl", "KG_wrt_OR_Pl", "KG_wrt_OR_Rl",\
# 				"KG_wrt_OR_F2", "KG_wrt_OR_P2", "KG_wrt_OR_R2",\
				
# 				"UKG_wrt_TR_F", "UKG_wrt_TR_P", "UKG_wrt_TR_R",\
# 				"UKG_wrt_TR_Fl", "UKG_wrt_TR_Pl", "UKG_wrt_TR_Rl",\
# 				"UKG_wrt_TR_F2", "UKG_wrt_TR_P2", "UKG_wrt_TR_R2",\
				
# 				"UKG_wrt_OR_F", "UKG_wrt_OR_P", "UKG_wrt_OR_R",\
# 				"UKG_wrt_OR_Fl", "UKG_wrt_OR_Pl", "UKG_wrt_OR_Rl",\
# 				"UKG_wrt_OR_F2", "UKG_wrt_OR_P2", "UKG_wrt_OR_R2",\

# 				"FKG_wrt_TR_F", "FKG_wrt_TR_P", "FKG_wrt_TR_R",\
# 				"FKG_wrt_TR_Fl", "FKG_wrt_TR_Pl", "FKG_wrt_TR_Rl",\
# 				"FKG_wrt_TR_F2", "FKG_wrt_TR_P2", "FKG_wrt_TR_R2",\

# 				"FKG_wrt_OR_F", "FKG_wrt_OR_P", "FKG_wrt_OR_R",\
# 				"FKG_wrt_OR_Fl", "FKG_wrt_OR_Pl", "FKG_wrt_OR_Rl",\
# 				"FKG_wrt_OR_F2", "FKG_wrt_OR_P2", "FKG_wrt_OR_R2",\

# 				"TR_wrt_OR_F", "TR_wrt_OR_P", "TR_wrt_OR_R",\
# 				"TR_wrt_OR_Fl", "TR_wrt_OR_Pl", "TR_wrt_OR_Rl",\
# 				"TR_wrt_OR_F2", "TR_wrt_OR_P2", "TR_wrt_OR_R2",\

# 				"OR_wrt_TR_F", "OR_wrt_TR_P", "OR_wrt_TR_R",\
# 				"OR_wrt_TR_Fl", "OR_wrt_TR_Pl", "OR_wrt_TR_Rl",\
# 				"OR_wrt_TR_F2", "OR_wrt_TR_P2", "OR_wrt_TR_R2"
# 			])

# 			if is_intermediate:
# 				temp_df = pd.read_excel(result_file_path)
# 				df = df.append(temp_df, sort = True, ignore_index = True)

# 			df.to_excel(result_file_path)
# 			is_intermediate = True

# 			f.flush()
# 			# break

		

# 		if LOWER_WORD_LIMIT != -1:
# 			KG_fold_summary_word_list = word_tokenize(open(f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/KG_fold/{file_name}_KG_fold_summary.txt", 'r').read())
# 			KG_file_summary_word_list = word_tokenize(open(f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/KG_file/{file_name}_KG_summary.txt", 'r').read())
# 			UKG_file_summary_word_list = word_tokenize(open(f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/UKG_file/{file_name}_UKG_summary.txt").read())
# 			tr_summary_word_list = word_tokenize(open(f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/tr/{file_name}_TR_summary.txt", 'r').read())
# 		else:
# 			KG_fold_summary_word_list = open(f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/KG_fold/{file_name}_KG_fold_summary.txt", 'r').read()
# 			KG_file_summary_word_list = open(f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/KG_file/{file_name}_KG_summary.txt", 'r').read()
# 			UKG_file_summary_word_list = open(f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/UKG_file/{file_name}_UKG_summary.txt").read()
# 			tr_summary_word_list = open(f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/tr/{file_name}_TR_summary.txt", 'r').read()

		

# 		or_summary = str(open(f"{base_summary_location}/{file_name}_1.txt", 'r', encoding='utf-8').read().encode('utf8')).replace("\\n", " ").replace("\\t", " ")#[LOWER_WORD_LIMIT:UPPER_WORD_LIMIT]
# 		or_summary = or_summary[2: len(or_summary) - 1]
		
# 		if is_all_or:
# 			for i in range(2,7):
# 				if os.path.isfile(f"{base_summary_location}/{file_name}_{i}.txt"):
# 					or_single_summary = str(open(f"{base_summary_location}/{file_name}_{i}.txt", 'r', encoding='utf-8').read().encode('utf8')).replace("\\n", " ").replace("\\t", " ")#[LOWER_WORD_LIMIT:UPPER_WORD_LIMIT]
# 					or_single_summary = or_single_summary[2: len(or_single_summary) - 1]
# 					or_summary = or_summary + '\n' + or_single_summary
		

		

# 		if len(KG_fold_summary_word_list) >= UPPER_WORD_LIMIT:
# 			KG_fold_summary = ' '.join(KG_fold_summary_word_list[LOWER_WORD_LIMIT:UPPER_WORD_LIMIT])
# 		else:
# 			KG_fold_summary = ' '.join(KG_fold_summary_word_list[LOWER_WORD_LIMIT:])

# 		if len(KG_file_summary_word_list) >= UPPER_WORD_LIMIT:
# 			KG_file_summary = ' '.join(KG_file_summary_word_list[LOWER_WORD_LIMIT:UPPER_WORD_LIMIT])
# 		else:
# 			KG_file_summary = ' '.join(KG_file_summary_word_list[LOWER_WORD_LIMIT:])

# 		if len(UKG_file_summary_word_list) >= UPPER_WORD_LIMIT:
# 			UKG_file_summary = ' '.join(UKG_file_summary_word_list[LOWER_WORD_LIMIT:UPPER_WORD_LIMIT])
# 		else:
# 			UKG_file_summary = ' '.join(UKG_file_summary_word_list[LOWER_WORD_LIMIT:])

# 		if len(tr_summary_word_list) >= UPPER_WORD_LIMIT:
# 			tr_summary = ' '.join(tr_summary_word_list[LOWER_WORD_LIMIT:UPPER_WORD_LIMIT])
# 		else:
# 			tr_summary = ' '.join(tr_summary_word_list[LOWER_WORD_LIMIT:])

# 		print("Considering ",(index + 1) , '/', len(summary_file_name_list), datetime.datetime.now(), len(word_tokenize(KG_file_summary)), len(UKG_file_summary_word_list), len(KG_fold_summary_word_list), len(tr_summary_word_list), len(word_tokenize(or_summary)))

# 		KG_TR_rouge_result = generate_rouge_score(KG_file_summary, tr_summary)
# 		KG_OR_rouge_result = generate_rouge_score(KG_file_summary, or_summary)
# 		UKG_TR_rouge_result = generate_rouge_score(UKG_file_summary, tr_summary)
# 		UKG_OR_rouge_result = generate_rouge_score(UKG_file_summary, or_summary)
# 		TR_OR_rouge_result = generate_rouge_score(tr_summary, or_summary)
# 		OR_TR_rouge_result = generate_rouge_score(or_summary, tr_summary)
# 		FKG_TR_rouge_result = generate_rouge_score(KG_fold_summary, tr_summary)
# 		FKG_OR_rouge_result = generate_rouge_score(KG_fold_summary, or_summary)

# 		del KG_file_summary, UKG_file_summary, KG_fold_summary, tr_summary, or_summary

# 		if KG_TR_rouge_result == -1:
# 			print("Error in KG_file_summary wrt TR", file_name)
# 			continue
# 		if KG_OR_rouge_result == -1:
# 			print("Error in KG_file_summary wrt OR", file_name)
# 			continue
# 		if UKG_TR_rouge_result == -1:
# 			print("Error in UKG_file_summary wrt TR", file_name)
# 			continue
# 		if UKG_OR_rouge_result == -1:
# 			print("Error in UKG_file_summary wrt OR", file_name)
# 			continue
# 		if TR_OR_rouge_result == -1:
# 			print("Error in tr_summary wrt or_summary", file_name)
# 			continue
# 		if OR_TR_rouge_result == -1:
# 			print("Error in or_summary wrt TR", file_name)
# 			continue
# 		if FKG_TR_rouge_result == -1:
# 			print("Error in KG_fold_summary wrt TR", file_name)
# 			continue
# 		if FKG_OR_rouge_result == -1:
# 			print("Error in KG_fold_summary wrt OR", file_name)
# 			continue

# 		'''KG TR'''
# 		tr_rouge_KG_f.append(round(KG_TR_rouge_result[0]['rouge-1']['f'], 4))
# 		tr_rouge_KG_p.append(round(KG_TR_rouge_result[0]['rouge-1']['p'], 4))
# 		tr_rouge_KG_r.append(round(KG_TR_rouge_result[0]['rouge-1']['r'], 4))

# 		tr_rouge_KG_fl.append(round(KG_TR_rouge_result[0]['rouge-l']['f'], 4))
# 		tr_rouge_KG_pl.append(round(KG_TR_rouge_result[0]['rouge-l']['p'], 4))
# 		tr_rouge_KG_rl.append(round(KG_TR_rouge_result[0]['rouge-l']['r'], 4))

# 		tr_rouge_KG_f2.append(round(KG_TR_rouge_result[0]['rouge-2']['f'], 4))
# 		tr_rouge_KG_p2.append(round(KG_TR_rouge_result[0]['rouge-2']['p'], 4))
# 		tr_rouge_KG_r2.append(round(KG_TR_rouge_result[0]['rouge-2']['r'], 4))		
# 		del KG_TR_rouge_result
# 		'''KG OR'''
# 		or_rouge_KG_f.append(round(KG_OR_rouge_result[0]['rouge-1']['f'], 4))
# 		or_rouge_KG_p.append(round(KG_OR_rouge_result[0]['rouge-1']['p'], 4))
# 		or_rouge_KG_r.append(round(KG_OR_rouge_result[0]['rouge-1']['r'], 4))		
		
# 		or_rouge_KG_fl.append(round(KG_OR_rouge_result[0]['rouge-l']['f'], 4))
# 		or_rouge_KG_pl.append(round(KG_OR_rouge_result[0]['rouge-l']['p'], 4))
# 		or_rouge_KG_rl.append(round(KG_OR_rouge_result[0]['rouge-l']['r'], 4))

# 		or_rouge_KG_f2.append(round(KG_OR_rouge_result[0]['rouge-2']['f'], 4))
# 		or_rouge_KG_p2.append(round(KG_OR_rouge_result[0]['rouge-2']['p'], 4))
# 		or_rouge_KG_r2.append(round(KG_OR_rouge_result[0]['rouge-2']['r'], 4))
# 		del KG_OR_rouge_result
# 		'''UKG TR'''
# 		tr_rouge_KG_U_f.append(round(UKG_TR_rouge_result[0]['rouge-1']['f'], 4))
# 		tr_rouge_KG_U_p.append(round(UKG_TR_rouge_result[0]['rouge-1']['p'], 4))
# 		tr_rouge_KG_U_r.append(round(UKG_TR_rouge_result[0]['rouge-1']['r'], 4))
		
# 		tr_rouge_KG_U_fl.append(round(UKG_TR_rouge_result[0]['rouge-l']['f'], 4))
# 		tr_rouge_KG_U_pl.append(round(UKG_TR_rouge_result[0]['rouge-l']['p'], 4))
# 		tr_rouge_KG_U_rl.append(round(UKG_TR_rouge_result[0]['rouge-l']['r'], 4))

# 		tr_rouge_KG_U_f2.append(round(UKG_TR_rouge_result[0]['rouge-2']['f'], 4))
# 		tr_rouge_KG_U_p2.append(round(UKG_TR_rouge_result[0]['rouge-2']['p'], 4))
# 		tr_rouge_KG_U_r2.append(round(UKG_TR_rouge_result[0]['rouge-2']['r'], 4))
# 		del UKG_TR_rouge_result
# 		'''UKG OR'''
# 		or_rouge_KG_U_f.append(round(UKG_OR_rouge_result[0]['rouge-1']['f'], 4))
# 		or_rouge_KG_U_p.append(round(UKG_OR_rouge_result[0]['rouge-1']['p'], 4))
# 		or_rouge_KG_U_r.append(round(UKG_OR_rouge_result[0]['rouge-1']['r'], 4))
		
# 		or_rouge_KG_U_fl.append(round(UKG_OR_rouge_result[0]['rouge-l']['f'], 4))
# 		or_rouge_KG_U_pl.append(round(UKG_OR_rouge_result[0]['rouge-l']['p'], 4))
# 		or_rouge_KG_U_rl.append(round(UKG_OR_rouge_result[0]['rouge-l']['r'], 4))

# 		or_rouge_KG_U_f2.append(round(UKG_OR_rouge_result[0]['rouge-2']['f'], 4))
# 		or_rouge_KG_U_p2.append(round(UKG_OR_rouge_result[0]['rouge-2']['p'], 4))
# 		or_rouge_KG_U_r2.append(round(UKG_OR_rouge_result[0]['rouge-2']['r'], 4))
# 		del UKG_OR_rouge_result

# 		'''TR OR'''
# 		tr_rouge_OR_f.append(round(TR_OR_rouge_result[0]['rouge-1']['f'], 4))
# 		tr_rouge_OR_p.append(round(TR_OR_rouge_result[0]['rouge-1']['p'], 4))
# 		tr_rouge_OR_r.append(round(TR_OR_rouge_result[0]['rouge-1']['r'], 4))

# 		tr_rouge_OR_fl.append(round(TR_OR_rouge_result[0]['rouge-l']['f'], 4))
# 		tr_rouge_OR_pl.append(round(TR_OR_rouge_result[0]['rouge-l']['p'], 4))
# 		tr_rouge_OR_rl.append(round(TR_OR_rouge_result[0]['rouge-l']['r'], 4))

# 		tr_rouge_OR_f2.append(round(TR_OR_rouge_result[0]['rouge-2']['f'], 4))
# 		tr_rouge_OR_p2.append(round(TR_OR_rouge_result[0]['rouge-2']['p'], 4))
# 		tr_rouge_OR_r2.append(round(TR_OR_rouge_result[0]['rouge-2']['r'], 4))
		
# 		'''OR TR'''
# 		or_rouge_tr_f.append(round(OR_TR_rouge_result[0]['rouge-1']['f'], 4))
# 		or_rouge_tr_p.append(round(OR_TR_rouge_result[0]['rouge-1']['p'], 4))
# 		or_rouge_tr_r.append(round(OR_TR_rouge_result[0]['rouge-1']['r'], 4))

# 		or_rouge_tr_fl.append(round(OR_TR_rouge_result[0]['rouge-l']['f'], 4))
# 		or_rouge_tr_pl.append(round(OR_TR_rouge_result[0]['rouge-l']['p'], 4))
# 		or_rouge_tr_rl.append(round(OR_TR_rouge_result[0]['rouge-l']['r'], 4))

# 		or_rouge_tr_f2.append(round(OR_TR_rouge_result[0]['rouge-2']['f'], 4))
# 		or_rouge_tr_p2.append(round(OR_TR_rouge_result[0]['rouge-2']['p'], 4))
# 		or_rouge_tr_r2.append(round(OR_TR_rouge_result[0]['rouge-2']['r'], 4))

# 		'''FKG TR'''
# 		tr_rouge_Fold_KG_list_f.append(round(FKG_TR_rouge_result[0]['rouge-1']['f'], 4))
# 		tr_rouge_Fold_KG_list_p.append(round(FKG_TR_rouge_result[0]['rouge-1']['p'], 4))
# 		tr_rouge_Fold_KG_list_r.append(round(FKG_TR_rouge_result[0]['rouge-1']['r'], 4))

# 		tr_rouge_Fold_KG_list_fl.append(round(FKG_TR_rouge_result[0]['rouge-l']['f'], 4))
# 		tr_rouge_Fold_KG_list_pl.append(round(FKG_TR_rouge_result[0]['rouge-l']['p'], 4))
# 		tr_rouge_Fold_KG_list_rl.append(round(FKG_TR_rouge_result[0]['rouge-l']['r'], 4))

# 		tr_rouge_Fold_KG_list_f2.append(round(FKG_TR_rouge_result[0]['rouge-2']['f'], 4))
# 		tr_rouge_Fold_KG_list_p2.append(round(FKG_TR_rouge_result[0]['rouge-2']['p'], 4))
# 		tr_rouge_Fold_KG_list_r2.append(round(FKG_TR_rouge_result[0]['rouge-2']['r'], 4))

# 		'''FKG OR'''
# 		or_rouge_Fold_KG_list_f.append(round(FKG_OR_rouge_result[0]['rouge-1']['f'], 4))
# 		or_rouge_Fold_KG_list_p.append(round(FKG_OR_rouge_result[0]['rouge-1']['p'], 4))
# 		or_rouge_Fold_KG_list_r.append(round(FKG_OR_rouge_result[0]['rouge-1']['r'], 4))

# 		or_rouge_Fold_KG_list_fl.append(round(FKG_OR_rouge_result[0]['rouge-l']['f'], 4))
# 		or_rouge_Fold_KG_list_pl.append(round(FKG_OR_rouge_result[0]['rouge-l']['p'], 4))
# 		or_rouge_Fold_KG_list_rl.append(round(FKG_OR_rouge_result[0]['rouge-l']['r'], 4))

# 		or_rouge_Fold_KG_list_f2.append(round(FKG_OR_rouge_result[0]['rouge-2']['f'], 4))
# 		or_rouge_Fold_KG_list_p2.append(round(FKG_OR_rouge_result[0]['rouge-2']['p'], 4))
# 		or_rouge_Fold_KG_list_r2.append(round(FKG_OR_rouge_result[0]['rouge-2']['r'], 4))

# 		file_name_list.append(f"{file_name}.txt")
# 		# print('line 352:', tr_rouge_KG_f)

# 		# break

# 		del FKG_OR_rouge_result, FKG_TR_rouge_result, OR_TR_rouge_result, TR_OR_rouge_result
# 	if len(tr_rouge_KG_f) > 0:


# 		df = pd.DataFrame(list(zip(file_name_list,\
# 				tr_rouge_KG_f, tr_rouge_KG_p, tr_rouge_KG_r,\
# 				tr_rouge_KG_fl, tr_rouge_KG_pl, tr_rouge_KG_rl,\
# 				tr_rouge_KG_f2, tr_rouge_KG_p2, tr_rouge_KG_r2,\

# 				or_rouge_tr_f, or_rouge_tr_p, or_rouge_tr_r,\
# 				or_rouge_tr_fl, or_rouge_tr_pl, or_rouge_tr_rl,\
# 				or_rouge_tr_f2, or_rouge_tr_p2, or_rouge_tr_r2,\

# 				tr_rouge_KG_U_f, tr_rouge_KG_U_p, tr_rouge_KG_U_r,\
# 				tr_rouge_KG_U_fl, tr_rouge_KG_U_pl, tr_rouge_KG_U_rl,\
# 				tr_rouge_KG_U_f2, tr_rouge_KG_U_p2, tr_rouge_KG_U_r2,\

# 				or_rouge_KG_U_f, or_rouge_KG_U_p, or_rouge_KG_U_r,\
# 				or_rouge_KG_U_fl, or_rouge_KG_U_pl, or_rouge_KG_U_rl,\
# 				or_rouge_KG_U_f2, or_rouge_KG_U_p2, or_rouge_KG_U_r2,\

# 				tr_rouge_Fold_KG_list_f, tr_rouge_Fold_KG_list_p, tr_rouge_Fold_KG_list_r,\
# 				tr_rouge_Fold_KG_list_fl, tr_rouge_Fold_KG_list_pl, tr_rouge_Fold_KG_list_rl,\
# 				tr_rouge_Fold_KG_list_f2, tr_rouge_Fold_KG_list_p2, tr_rouge_Fold_KG_list_r2,\

# 				or_rouge_Fold_KG_list_f, or_rouge_Fold_KG_list_p, or_rouge_Fold_KG_list_r,\
# 				or_rouge_Fold_KG_list_fl, or_rouge_Fold_KG_list_pl, or_rouge_Fold_KG_list_rl,\
# 				or_rouge_Fold_KG_list_f2, or_rouge_Fold_KG_list_p2, or_rouge_Fold_KG_list_r2,\


# 				tr_rouge_OR_f, tr_rouge_OR_p, tr_rouge_OR_r,\
# 				tr_rouge_OR_fl, tr_rouge_OR_pl, tr_rouge_OR_rl,\
# 				tr_rouge_OR_f2, tr_rouge_OR_p2, tr_rouge_OR_r2,\

# 				or_rouge_tr_f, or_rouge_tr_p, or_rouge_tr_r,\
# 				or_rouge_tr_fl, or_rouge_tr_pl, or_rouge_tr_rl,\
# 				or_rouge_tr_f2, or_rouge_tr_p2, or_rouge_tr_r2

# 				)),	columns =["File_Name",\
# 				"KG_wrt_TR_F", "KG_wrt_TR_P", "KG_wrt_TR_R",\
# 				"KG_wrt_TR_Fl", "KG_wrt_TR_Pl", "KG_wrt_TR_Rl",\
# 				"KG_wrt_TR_F2", "KG_wrt_TR_P2", "KG_wrt_TR_R2",\

# 				"KG_wrt_OR_F", "KG_wrt_OR_P", "KG_wrt_OR_R",\
# 				"KG_wrt_OR_Fl", "KG_wrt_OR_Pl", "KG_wrt_OR_Rl",\
# 				"KG_wrt_OR_F2", "KG_wrt_OR_P2", "KG_wrt_OR_R2",\
				
# 				"UKG_wrt_TR_F", "UKG_wrt_TR_P", "UKG_wrt_TR_R",\
# 				"UKG_wrt_TR_Fl", "UKG_wrt_TR_Pl", "UKG_wrt_TR_Rl",\
# 				"UKG_wrt_TR_F2", "UKG_wrt_TR_P2", "UKG_wrt_TR_R2",\
				
# 				"UKG_wrt_OR_F", "UKG_wrt_OR_P", "UKG_wrt_OR_R",\
# 				"UKG_wrt_OR_Fl", "UKG_wrt_OR_Pl", "UKG_wrt_OR_Rl",\
# 				"UKG_wrt_OR_F2", "UKG_wrt_OR_P2", "UKG_wrt_OR_R2",\

# 				"FKG_wrt_TR_F", "FKG_wrt_TR_P", "FKG_wrt_TR_R",\
# 				"FKG_wrt_TR_Fl", "FKG_wrt_TR_Pl", "FKG_wrt_TR_Rl",\
# 				"FKG_wrt_TR_F2", "FKG_wrt_TR_P2", "FKG_wrt_TR_R2",\

# 				"FKG_wrt_OR_F", "FKG_wrt_OR_P", "FKG_wrt_OR_R",\
# 				"FKG_wrt_OR_Fl", "FKG_wrt_OR_Pl", "FKG_wrt_OR_Rl",\
# 				"FKG_wrt_OR_F2", "FKG_wrt_OR_P2", "FKG_wrt_OR_R2",\

# 				"TR_wrt_OR_F", "TR_wrt_OR_P", "TR_wrt_OR_R",\
# 				"TR_wrt_OR_Fl", "TR_wrt_OR_Pl", "TR_wrt_OR_Rl",\
# 				"TR_wrt_OR_F2", "TR_wrt_OR_P2", "TR_wrt_OR_R2",\

# 				"OR_wrt_TR_F", "OR_wrt_TR_P", "OR_wrt_TR_R",\
# 				"OR_wrt_TR_Fl", "OR_wrt_TR_Pl", "OR_wrt_TR_Rl",\
# 				"OR_wrt_TR_F2", "OR_wrt_TR_P2", "OR_wrt_TR_R2"
# 			])

# 		if is_intermediate:
# 			temp_df = pd.read_excel(result_file_path)
# 			df = df.append(temp_df, sort=True, ignore_index = True)

# 		print(f"Fold {fold_number} end time {datetime.datetime.now()}")
# 		print(f"Averaged Result: {fold_number}  :  {len(tr_rouge_KG_f)}")
# 		print('KG TR')
# 		print(round(np.average(tr_rouge_KG_f), 4), round(np.average(tr_rouge_KG_p), 4), round(np.average(tr_rouge_KG_r), 4))
# 		print(round(np.average(tr_rouge_KG_fl), 4), round(np.average(tr_rouge_KG_pl), 4), round(np.average(tr_rouge_KG_rl), 4))
# 		print(round(np.average(tr_rouge_KG_f2), 4), round(np.average(tr_rouge_KG_p2), 4), round(np.average(tr_rouge_KG_r2), 4))
# 		print('KG OR')
# 		print(round(np.average(or_rouge_tr_f), 4), round(np.average(or_rouge_tr_p), 4), round(np.average(or_rouge_tr_r), 4))
# 		print(round(np.average(or_rouge_tr_fl), 4), round(np.average(or_rouge_tr_pl), 4), round(np.average(or_rouge_tr_rl), 4))
# 		print(round(np.average(or_rouge_tr_f2), 4), round(np.average(or_rouge_tr_p2), 4), round(np.average(or_rouge_tr_r2), 4))
# 		print('UKG TR')
# 		print(round(np.average(tr_rouge_KG_U_f), 4), round(np.average(tr_rouge_KG_U_p), 4), round(np.average(tr_rouge_KG_U_r), 4))
# 		print(round(np.average(tr_rouge_KG_U_fl), 4), round(np.average(tr_rouge_KG_U_pl), 4), round(np.average(tr_rouge_KG_U_rl), 4))
# 		print(round(np.average(tr_rouge_KG_U_f2), 4), round(np.average(tr_rouge_KG_U_p2), 4), round(np.average(tr_rouge_KG_U_r2), 4))
# 		print('UKG OR')
# 		print(round(np.average(or_rouge_KG_U_f), 4), round(np.average(or_rouge_KG_U_p), 4), round(np.average(or_rouge_KG_U_r), 4))
# 		print(round(np.average(or_rouge_KG_U_fl), 4), round(np.average(or_rouge_KG_U_pl), 4), round(np.average(or_rouge_KG_U_rl), 4))
# 		print(round(np.average(or_rouge_KG_U_f2), 4), round(np.average(or_rouge_KG_U_p2), 4), round(np.average(or_rouge_KG_U_r2), 4))
# 		print('FKG TR')
# 		print(round(np.average(tr_rouge_Fold_KG_list_f), 4), round(np.average(tr_rouge_Fold_KG_list_p), 4), round(np.average(tr_rouge_Fold_KG_list_r), 4))
# 		print(round(np.average(tr_rouge_Fold_KG_list_fl), 4), round(np.average(tr_rouge_Fold_KG_list_pl), 4), round(np.average(tr_rouge_Fold_KG_list_rl), 4))
# 		print(round(np.average(tr_rouge_Fold_KG_list_f2), 4), round(np.average(tr_rouge_Fold_KG_list_p2), 4), round(np.average(tr_rouge_Fold_KG_list_r2), 4))
# 		print('FKG OR')
# 		print(round(np.average(or_rouge_Fold_KG_list_f), 4), round(np.average(or_rouge_Fold_KG_list_p), 4), round(np.average(or_rouge_Fold_KG_list_r), 4))
# 		print(round(np.average(or_rouge_Fold_KG_list_fl), 4), round(np.average(or_rouge_Fold_KG_list_pl), 4), round(np.average(or_rouge_Fold_KG_list_rl), 4))
# 		print(round(np.average(or_rouge_Fold_KG_list_f2), 4), round(np.average(or_rouge_Fold_KG_list_p2), 4), round(np.average(or_rouge_Fold_KG_list_r2), 4))
# 		print('TR OR')
# 		print(round(np.average(tr_rouge_OR_f), 4), round(np.average(tr_rouge_OR_p), 4), round(np.average(tr_rouge_OR_r), 4))
# 		print(round(np.average(tr_rouge_OR_fl), 4), round(np.average(tr_rouge_OR_pl), 4), round(np.average(tr_rouge_OR_rl), 4))
# 		print(round(np.average(tr_rouge_OR_f2), 4), round(np.average(tr_rouge_OR_p2), 4), round(np.average(tr_rouge_OR_r2), 4))
# 		print('OR TR')
# 		print(round(np.average(or_rouge_tr_f), 4), round(np.average(or_rouge_tr_p), 4), round(np.average(or_rouge_tr_r), 4))
# 		print(round(np.average(or_rouge_tr_fl), 4), round(np.average(or_rouge_tr_pl), 4), round(np.average(or_rouge_tr_rl), 4))
# 		print(round(np.average(or_rouge_tr_f2), 4), round(np.average(or_rouge_tr_p2), 4), round(np.average(or_rouge_tr_r2), 4))
# 		print("\n\n\n")

# 		# return
# 		df.to_excel(result_file_path)
# 	else:
# 		print(f"Fold {fold_number} failed to generate summaries")

def score_triples(sub_count_dict, obj_count_dict, svo_count_dict):
	updated_svo_count = {}
	# i = 1
	for triple, count in svo_count_dict.items():

		# if i % 1000 == 0:
		# 	print(i)
		# i = i + 1

		svo = triple.split(',')
		if svo[0].strip() in sub_count_dict:
			count = count + sub_count_dict[svo[0].strip()]
		if svo[2].strip() in obj_count_dict:
			count = count + obj_count_dict[svo[2].strip()]
		updated_svo_count[triple] = count

	updated_svo_count = {str(k): v for k, v in sorted(updated_svo_count.items(), key=lambda item: item[1], reverse=True)}	
	return updated_svo_count

def summary_by_triple(svo_count_dict, svo_sent_dict ,sent_ord):

	# print("Generating summary from svo dict")
	# f.flush()

	summary = {}
	if(len(svo_count_dict)) == 0:
		print("No triple found hack in KG")
		return -1 #'\n'.join(tokenized_sent), -1, -1, -1, -1
	# elif list(svo_count_dict.values())[0] == 1 and list(svo_count_dict.values())[-1] == 1:
	# 	print("No imp triple: extract sentences using verb appearence", list(svo_count_dict.values())[0], "  ", list(svo_count_dict.values())[-1])
	# 	print("hack in KG")
	# 	return -1 #'\n'.join(tokenized_sent), -1, -1, -1, -1
	else:	
		for svo in list(svo_count_dict.keys()):
			if svo in svo_sent_dict.keys():
				for sentence in svo_sent_dict[svo]:
					# print(sentence, sent_ord[sentence])
					summary[sentence] =  sent_ord[sentence]
					break

	summary = {str(k): v for k, v in summary.items()}#sorted(summary.items(), key=lambda item: item[1], reverse=False)}
	
	summary = ' '.join(list(summary.keys())).replace("\n", " ").replace("  ", " ")
	# print('line number 872', summary)
	# if LOWER_WORD_LIMIT != -1:
	# 	summary = ' '.join(list(summary.keys())).replace("\n", " ").replace("  ", " ")
	# else:
	# 	# print("Generated summary line : 537",summary.keys())
	# 	summary = ' '.join(list(summary.keys())).replace("\n", " ").replace("  ", " ")

	# print(' '.join(summary.split(" ")[:min(len(summary), 1000)]))
	return summary#' '.join(word_tokenize(summary)[:min(len(summary), UPPER_WORD_LIMIT)])

def summarize_text_using_file(file_text):

	tokenized_sent = sent_tokenize(file_text)
	
	svo_sent_dict = {}
	svo_count_dict = {}
	svo_list = []
	sub_list = []
	obj_list = []
	verb_list = []

	sent_ord = {}
	for i, sent in enumerate(tokenized_sent, start=1):
		sent_svo_triple = list(textacy.extract.subject_verb_object_triples(nlp(sent)))

		# sent_pos_dict[sent] = i
		sent_ord[sent] = i
		for svo in sent_svo_triple:

			s = str(svo[0])
			v = str(svo[1])
			o = str(svo[2])
			# sent_svo = list(sent_svo_triple)

			verb_list.append(v)
			sub_list.append(s)
			obj_list.append(o)
			svo_str = s + ',' + v + ',' + o#', '.join(svo).strip()#str(list(svo)[0]) + +str(list(svo)[1]) + str(list(svo)[2])


			if svo_str in svo_list:
				svo_count_dict[svo_str] = svo_count_dict[svo_str] + 1
				svo_sent_dict[svo_str]  = svo_sent_dict[svo_str] + [sent]#.append(sent)
			else:
				svo_count_dict[svo_str] = 1
				svo_sent_dict[svo_str] = [sent]
				svo_list.append(svo_str)

	#Remove stop words
	sub_list = [w for w in sub_list if not w in stop_words]
	obj_list = [w for w in obj_list if not w in stop_words]

	sub_count_dict = Counter(sub_list)
	obj_count_dict = Counter(obj_list)

	sub_count_dict = {str(k): v for k, v in sorted(sub_count_dict.items(), key=lambda item: item[1], reverse=True)}
	obj_count_dict = {str(k): v for k, v in sorted(obj_count_dict.items(), key=lambda item: item[1], reverse=True)}

	svo_count_dict = {str(k): v for k, v in sorted(svo_count_dict.items(), key=lambda item: item[1], reverse=True)}

	summary_length = max(1, min(math.floor(len(svo_count_dict)* DICT_REDUCTION_FACTOR), int(len(tokenized_sent) * TEXT_REDUCTION_FACTOR)))
	svo_count_dict_updated = dict(Counter(score_triples(sub_count_dict, obj_count_dict, svo_count_dict)).most_common(summary_length))

	num_triple =  len(svo_count_dict)
	if num_triple != 0:
		max_count = list(svo_count_dict.values())[0]	
	num_sent = len(tokenized_sent)


	KG_triple_summary = summary_by_triple(svo_count_dict, svo_sent_dict ,sent_ord)
	KG_updated_triple_summary = summary_by_triple(svo_count_dict_updated, svo_sent_dict ,sent_ord)

	if KG_triple_summary == -1:
		print("KG_summary causing problem")
		return -1
	if KG_updated_triple_summary == -1:
		print("KG_updated_triple_summary causing problem")
		return -1

	summary_textrank = ""
	# summary_textrank = summarize(' '.join([x.strip() for x in sent_tokenize(file_text.strip().replace('\n', ''))])) #, word_count = int(sum([len(word_tokenize(x)) for x in sent_tokenize(text)])/3)
	try:
		summary_textrank = summarize(' '.join([x.strip() for x in sent_tokenize(file_text.strip().replace('\n', ''))])) #, word_count = int(sum([len(word_tokenize(x)) for x in sent_tokenize(text)])/3)
		
		# if LOWER_WORD_LIMIT != -1:
		# 	summary_textrank = ' '.join(summary_textrank.split(' ')[LOWER_WORD_LIMIT:UPPER_WORD_LIMIT])
		# else:
		# 	summary_textrank = ' '.join(summary_textrank.split(' ')[:UPPER_WORD_LIMIT])
	except Exception as e:
		print("More than one sentence Exception:line 228", e)
		return -1
	
	return KG_triple_summary, KG_updated_triple_summary, summary_textrank, svo_count_dict, svo_count_dict_updated,\
	max_count, num_triple, num_sent, 


# def summarize_text_using_fold(file_text, test_fold_number):

# 	global TEST_FOLD_TRIPLES_DICT
# 	svo_sent_dict = {}
# 	svo_count_dict = {}
# 	svo_list = []
# 	verb_list = []
# 	sub_list = []
# 	obj_list = []

# 	sent_ord = {}
# 	# sent_pos_dict = {}

# 	tokenized_sent = sent_tokenize(file_text)
	
# 	for i, sent in enumerate(tokenized_sent, start=1):
# 		sent_svo_triple = list(textacy.extract.subject_verb_object_triples(nlp(sent)))

# 		# sent_pos_dict[sent] = i
# 		sent_ord[sent] = i
# 		for j, svo in enumerate(sent_svo_triple):

# 			s = lemmatizer.lemmatize(str(svo[0]).lower())
# 			v = lemmatizer.lemmatize(str(svo[1]).lower())
# 			o = lemmatizer.lemmatize(str(svo[2]).lower())
# 			sent_svo = list(sent_svo_triple)

# 			svo_str = s + ',' +v + ',' + o

# 			if svo_str in svo_list:
# 				svo_count_dict[svo_str] = svo_count_dict[svo_str] + 1
# 				svo_sent_dict[svo_str]  = svo_sent_dict[svo_str] + [sent]#.append(sent)
# 			else:
# 				svo_count_dict[svo_str] = 1
# 				svo_sent_dict[svo_str] = [sent]
# 				svo_list.append(svo_str)
	

# 	for svo in svo_list:
# 		if svo in TEST_FOLD_TRIPLES_DICT.keys():
# 			# print(svo, TEST_FOLD_TRIPLES_DICT[svo], svo_count_dict[svo])
# 			svo_count_dict[svo] = svo_count_dict[svo] + TEST_FOLD_TRIPLES_DICT[svo]


# 	svo_count_dict = {str(k): v for k, v in sorted(svo_count_dict.items(), key=lambda item: item[1], reverse=True)}

# 	# if LOWER_WORD_LIMIT != -1:
# 	# 	KG_fold_summary = ' '.join(word_tokenize(' '.join(summary_by_triple(svo_count_dict, svo_sent_dict, sent_ord))[LOWER_WORD_LIMIT:UPPER_WORD_LIMIT]))
# 	# else:
# 	# 	KG_fold_summary = ' '.join(summary_by_triple(svo_count_dict, svo_sent_dict, sent_ord))

# 	KG_fold_summary = summary_by_triple(svo_count_dict, svo_sent_dict, sent_ord)

# 	if KG_fold_summary == -1:
# 		print("KG_summary causing problem")
# 		return -1;

# 	return KG_fold_summary, svo_count_dict

# def summarize_fold(file_names, fold_number):


# 	print("Inside summarize fold for: ", len(file_names), " files")
	
# 	if fold_number == -2:
# 		base_location = TEST_TEXT_PATH
# 	elif fold_number == -1:
# 		base_location = VALIDATION_TEXT_PATH
# 	else:
# 		base_location = TEXT_DATA_LOCATION


# 	triple_fold_df = pd.read_excel(f"./Folds_Complete/Fold_{fold_number}/triple_frequency_lookup.xlsx")

# 	global TEST_FOLD_TRIPLES_DICT
# 	TEST_FOLD_TRIPLES_DICT = dict(zip(triple_fold_df["Triple"].tolist(), triple_fold_df["Frequency"].tolist()))


# 	KG_file_path = f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/KG_file/"
# 	if not os.path.exists(KG_file_path):
# 		os.makedirs(KG_file_path)

# 	UKG_file_path = f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/UKG_file/"
# 	if not os.path.exists(UKG_file_path):
# 		os.makedirs(UKG_file_path)

# 	KG_fold_path = f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/KG_fold/"
# 	if not os.path.exists(KG_fold_path):
# 		os.makedirs(KG_fold_path)

# 	tr_path = f"./Folds_Complete/Fold_{fold_number}/Result_{DICT_REDUCTION_FACTOR}/tr"
# 	if not os.path.exists(tr_path):
# 		os.makedirs(tr_path)


# 	f.flush()
# 	for index, file_name in enumerate(file_names):


# 		# if file_name in summarized_file_name_list:
# 		# 	continue
# 		if os.path.exists(f"{KG_fold_path}/{file_name.split('.')[0]}_KG_fold_summary.txt"):
# 			print("Continueing for file: ", file_name)
# 			continue

# 		if index % 32 == 0:
# 			print(f"Fold {fold_number} end time {datetime.datetime.now()}\n\n\n")
# 			print(f"Generated summary {file_name}  {index} / {len(file_names)}", datetime.datetime.now())
			
# 			f.flush()

# 		text = open(f"{base_location}/{file_name}").read()
		
# 		# summary = open(f"{SUMMARY_DATA_LOCATION}/{file_name.split('.')[0]}_1.txt").read()
		
# 		KG_plain_result = summarize_text_using_file(text)
# 		print(len(text))
# 		# break
# 		if KG_plain_result == -1:
# 			print(f"{fold_number}, {index}, {file_name}, Issue generating file summary")
# 			continue
# 		# print("Plain KG Summary generated......")

# 		# # print("KG fold summary generation started")
# 		KG_fold_result = summarize_text_using_fold(file_text = text, test_fold_number = fold_number)
# 		if KG_fold_result == -1:
# 			print(f"{fold_number}, {index}, {file_name}, Issue generating fold summary")
# 			continue			


# 		# print("\n\n\n\nThis is KG plain result\n",KG_plain_result)

# 		# break
# 		# print('\t\t\t', KG_plain_result)
# 		# print('\t\t\t', KG_fold_result)
# 		print("Saving files")
# 		# file_name_list.append(file_name)

# 		save_file = open(f"{KG_file_path}/{file_name.split('.')[0]}_KG_summary.txt", "w")
# 		save_file.write(KG_plain_result[0])
# 		save_file.close()

# 		save_file = open(f"{KG_file_path}/{file_name.split('.')[0]}_svo_dict.txt", "w")
# 		save_file.write(json.dumps(KG_plain_result[3]))
# 		save_file.close()


# 		save_file = open(f"{UKG_file_path}/{file_name.split('.')[0]}_UKG_summary.txt", "w")
# 		save_file.write(KG_plain_result[1])
# 		save_file.close()

# 		save_file = open(f"{UKG_file_path}/{file_name.split('.')[0]}_updated_svo_dict.txt", "w")
# 		save_file.write(json.dumps(KG_plain_result[4]))
# 		save_file.close()



# 		save_file = open(f"{tr_path}/{file_name.split('.')[0]}_TR_summary.txt", "w")
# 		save_file.write(KG_plain_result[2])
# 		save_file.close()

		
# 		save_file = open(f"{KG_fold_path}/{file_name.split('.')[0]}_KG_fold_summary.txt", "w")
# 		save_file.write(KG_fold_result[0])
# 		save_file.close()

# 		save_file = open(f"{KG_fold_path}/{file_name.split('.')[0]}_svo_dict.txt", "w")
# 		save_file.write(json.dumps(KG_fold_result[1]))
# 		save_file.close()

# 		# break
# 	return
