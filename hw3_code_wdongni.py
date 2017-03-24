import codecs
import os
from collections import defaultdict

# Helper methods

def output_list(out_list, out_filename):
	""" Write the content of out_list to out_filename

	Args: 
		out_list: a list of strings to be written 
		out_filename: the path of the output file
	""" 

	with codecs.open(out_filename, 'w+', 'utf-8') as f:
		f.write("\n".join(out_list))


def read_to_list(in_filename):
	""" Read the content of in_filename to a list (separated by '\n')

	Args:
	    in_filename:  the path of the input file

	Returns:
	    A list of strings which are in the specified file 
	"""

	ret_list = []
	with codecs.open(in_filename, 'rU', 'utf-8') as f:
		for line in f:
			ret_list.append(line[:-1])
	return ret_list 


def extract_elem_from_dir(xml_dir_path, elem_name):
	""" Extract all the text elements with the specified name from xml_filename

	Args:
		xml_dir_path: a string representing the path to the xml_directory 
			that have XML files extracted by CoreNLP.
		elem_name: the name of the text elements that have the elem_name as 
			start tag and end tag.

		Returns: 
			A list of unique tags, sorted alphabetically.
	"""


	unique_tags = set()
	xml_files = os.listdir(xml_dir_path)
	start_tag, end_tag = "<"+elem_name+">", "</"+elem_name+">"
	tag_len = len(start_tag)
	for xml_filename in xml_files:
		with codecs.open(xml_dir_path+"/"+xml_filename, 'rU', 'utf-8') as f:
			for line in f:
				index_s, index_t = line.find(start_tag), line.find(end_tag)
				if index_s < 0: continue
				text_elem = line[index_s+tag_len:index_t]
				unique_tags.add(text_elem)
	tag_list = list(unique_tags)
	tag_list.sort()
	return tag_list


def map_tags(xml_filename, elem_list, elem_name):
	""" Get the feature vector in the feature space of the elem_list

	Args:
	    xml_filename: a string representing the name of a XML file that may 
			contain elements with tags in the elem_list.
		elem_list: a list of elements names. 
		elem_name: the name of the text elements that have the elem_name as 
			start tag and end tag.

	Returns:
	    A list of real numbers with the same size as elem_list. Each element
	    in the returned list represents the number of times the corresponding 
	    element name in elem_list has occurred in the XML input file (or the 
	    frequency) divided by the number of all tokens in the file.
	"""


	num_tokens = 0
	elem_cnt_dict = defaultdict(int)
	start_tag, end_tag = "<"+elem_name+">", "</"+elem_name+">"
	tag_len = len(start_tag)
	with codecs.open(xml_filename, 'rU', 'utf-8') as f:
		for line in f:
			if "</token>" in line: num_tokens += 1
			index_s, index_t = line.find(start_tag), line.find(end_tag)
			if index_s < 0: continue
			text_elem = line[index_s+tag_len:index_t]
			elem_cnt_dict[text_elem] += 1
	return [elem_cnt_dict[elem]*1.0/num_tokens for elem in elem_list]


def get_ptb_google_mapping(mapping_path):
	""" Get a dictionary that contains a mapping from PTB tags to Google
	    universal tags.
	"""


	in_list = read_to_list(mapping_path)
	mapping_dict = {}
	for m in in_list:
		m_tuple = m.split("\t")
		mapping_dict[m_tuple[0]] = m_tuple[1]
	return mapping_dict


# Main methods

# Part 4. Part of Speech Tags 

# 4.1
def extract_pos_tags(xml_dir_path):
	""" Extract all the unique POS tags from all documents in xml_directory

	Args:
		xml_dir_path: a string representing the path to the xml_directory 
			that have XML files extracted by CoreNLP.

	Returns:
		A list of unique POS tags, sorted alphabetically.
	"""

	return extract_elem_from_dir(xml_dir_path, "POS")
	


# 4.2
def map_pos_tags(xml_filename, pos_tag_list):
	""" Get the feature vector in the feature space of the pos_tag_list

	Args:
	    xml_filename: a string representing the name of a XML file that may 
			contain elements with tags in the pos_tag_list.
		pos_tag_list: a list of known POS tags. 

	Returns:
	    A list of real numbers with the same size as pos_tag_list. Each element
	    in the returned list represents the number of times the corresponding 
	    POS tag in pos_tag_list has occurred in the XML input file (or the 
	    frequency) divided by the number of all tokens in the file.
	"""

	return map_tags(xml_filename, pos_tag_list, "POS")


# 4.3 
def map_universal_tags(ptb_pos_feat_vector, pos_tag_list, ptb_google_mapping,
	                   universal_tag_list):
	""" Get the feature vector in the feature space of the universal_tag_list

	Args:
	    ptb_pos_feat_vector: the output vector from section 4.2.
	    pos_tag_list: the output from section 4.1.
	    ptb_google_mapping: a dictionary that contains a mapping from PTB tags 
	    to Google universal tags.
	    universal_tag_list: the list of Google universal tags sorted in 
	    alphabetical order.

	Returns:
	    A list of real numbers with the same size as universal tag list. Each
	    element in the returned list is equal to the fraction of tokens in the
	    text (represented by the input vector) with the corresponding universal
	    POS tag in universal tag list.
	""" 


	uni_vec = [0]*len(universal_tag_list)
	for ind, pos_tag in enumerate(pos_tag_list):
		uni_tag = ptb_google_mapping[pos_tag]
		uni_vec[universal_tag_list.index(uni_tag)] += ptb_pos_feat_vector[ind]
	return uni_vec


# 5.1 
def extract_ner_tags(xml_dir_path):
	""" Extract all the unique NER tags from all documents in xml_directory

	Args:
		xml_dir_path: a string representing the path to the xml_directory 
			that have XML files extracted by CoreNLP.

	Returns:
		A list of unique NER tags, sorted alphabetically.
	"""

	return extract_elem_from_dir(xml_dir_path, "NER")


# 5.2 

def map_named_entity_tags(xml_filename, entity_list):
	""" Get the feature vector in the feature space of the entity_list

	Args:
	    xml_filename: a string representing the name of a XML file that may 
			contain elements with tags in the entity_list.
		entity_list: a list of named entity classes. 

	Returns:
	    A list of real numbers with the same size as entity_list. Each element
	    in the returned list represents the number of times the corresponding 
	    named entity class in entity_list has occurred in the XML input file 
	    (or the frequency) divided by the number of all tokens in the file.
	"""

	return map_tags(xml_filename, entity_list, "NER")


# 6.1

def extract_dependencies(xml_dir_path):
	""" Extract all the unique dependency relations from all documents in 
		xml_directory

	Args:
		xml_dir_path: a string representing the path to the xml_directory 
			that have XML files extracted by CoreNLP.

	Returns:
		A list of unique dependency relations, sorted alphabetically.
	"""

	unique_attrs = set()
	xml_files = os.listdir(xml_dir_path)
	elem_name, attr_name = "dep", "type"
	basic_flag_s, basic_flag_t ="<basic-dependencies>", "</basic-dependencies>"
	basic_flag = False
	start_tag, end_tag = "<"+elem_name+" "+attr_name+"=\"", "\">"
	tag_len = len(start_tag)
	for xml_filename in xml_files:
		with codecs.open(xml_dir_path+"/"+xml_filename, 'rU', 'utf-8') as f:
			for line in f:
				if basic_flag_s in line: basic_flag = True
				if basic_flag_t in line: basic_flag = False
				if not basic_flag: continue 
				index_s, index_t = line.find(start_tag), line.find(end_tag)
				if index_s < 0: continue
				text_attr = line[index_s+tag_len:index_t]
				unique_attrs.add(text_attr)
	attr_list = list(unique_attrs)
	attr_list.sort()
	return attr_list









if __name__ == "__main__":
	pass