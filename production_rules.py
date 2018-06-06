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
	    xml_filename: a string representing the path of a XML file that may 
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
	    xml_filename: a string representing the path of a XML file that may 
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
	    xml_filename: a string representing the path of a XML file that may 
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


# 6.2 

def map_dependencies(xml_filename, dependency_list):
	""" Get a representation that indicates the fraction of a given type of 
		dependency for the entire text.

	Args: 
	    xml_filename: a string representing the name of a XML file that may 
			contain elements with tags in the entity_list.
		dependency_list: a list of dependency types. 

	Returns: 
		A list of the same length as dependency list. Each element in the 
		output list takes the value of the number of times the corresponding 
		dependency in dependency list appeared in the XML input file normalized
		by the number of all dependencies in the text.
	"""

	num_deps = 0
	deps_cnt_dict = defaultdict(int)
	elem_name, attr_name = "dep", "type"
	basic_flag_s, basic_flag_t ="<basic-dependencies>", "</basic-dependencies>"
	basic_flag = False
	start_tag, end_tag = "<"+elem_name+" "+attr_name+"=\"", "\">"
	tag_len = len(start_tag)
	with codecs.open(xml_filename, 'rU', 'utf-8') as f:
		for line in f:
			if basic_flag_s in line: basic_flag = True
			if basic_flag_t in line: basic_flag = False
			if not basic_flag: continue 
			if "</dep>" in line: num_deps += 1
			index_s, index_t = line.find(start_tag), line.find(end_tag)
			if index_s < 0: continue
			text_elem = line[index_s+tag_len:index_t]
			deps_cnt_dict[text_elem] += 1
	return [deps_cnt_dict[dep]*1.0/num_deps for dep in dependency_list]


# 7.1

class ProdTree: 
	""" Data structure for each node in the constructed tree and its functions
	 	for extracting production rules.


    Attributes:
       	root: A string representing the value of a tree node. None 
       		represents an empty tree or a terminal node (lexical leaf). 
       	children: A list of Prod_tree. An empty list means that the node 
        	only has a terminal node as the only child. 
	"""

	def __init__(self):
		""" Inits the Prod_tree
		"""
		self.root = None
		self.children = []

	def parse(self, tree_str):
		""" Parse the tree_str recursively. May add child to the self.children

		Returns: 
			A string that has left to be parsed because the parsing has reached
			a terminal node. Notice if a terminal node has multiple ")", each
			parse takes out only one.

		Raises:
			ValueError: An error occurs if tree_str is not well-formatted. 
		""" 
		if not tree_str: return 

		if tree_str[0] == "(": # start of a non-terminal node 
			tree_splits = tree_str.split(" ", 1)
			self.root = tree_splits[0][1:]
			return self.parse_children(tree_splits[1])
		else:
			brack_i = tree_str.find(")")
			if brack_i < 0: raise ValueError, "Formatting Error"
			return tree_str[brack_i+1:]

	def parse_children(self, children_str):
		""" Parse the children_str

		Returns:
			A string that has left to be parsed because the parsing has reached
			a terminal node. Notice if a terminal node has multiple ")", each
			parse takes out only one.
		"""
		while children_str:
			# print self.root, "\t sub ", children_str
			if children_str[0] == ")": # signals the end of parsing on a node
				return children_str[1:]
			elif children_str[0] == " ":
				children_str = children_str[1:]
			child_node = ProdTree()
			try:
				children_str = child_node.parse(children_str)

			except ValueError:
				# print "Node " + self.root + "Parse Error."
				return
			if child_node.root: 
				self.children.append(child_node)
			else: # child is terminal, just stop 
				# print "returned to root ", self.root, children_str
				return children_str
			# print "next-round: ", children_str

	def extract_rules(self):
		""" Extract production rules, recursively

		Returns:
			A set of production rules, sorted alphabetically.
		"""
		if not self.children: return set()
		rules_set = set()
		my_rule_list = [self.root]+[c.root for c in self.children]
		my_rule = "_".join(my_rule_list)
		rules_set.add(my_rule)
		for c in self.children:
			rules_set |= c.extract_rules()
		return rules_set			


def extract_prod_rules(xml_dir_path):
	""" Extract all the unique syntactic production rules for the parse trees 
		from all documents in xml_directory.

	Args:
		xml_dir_path: a string representing the path to the xml_directory 
			that have XML files extracted by CoreNLP.

	Returns:
		A list of unique production rules, sorted alphabetically.
	"""


	unique_rules = set()
	xml_files = os.listdir(xml_dir_path)
	elem_name = "parse"
	start_tag, end_tag = "<"+elem_name+">", "</"+elem_name+">"
	tag_len = len(start_tag)
	for xml_filename in xml_files:
		with codecs.open(xml_dir_path+"/"+xml_filename, 'rU', 'utf-8') as f:
			for line in f:
				index_s, index_t = line.find(start_tag), line.find(end_tag)
				if index_s < 0: continue
				text_elem = line[index_s+tag_len:index_t]
				root_node = ProdTree()
				root_node.parse(text_elem)
				unique_rules |= root_node.extract_rules()
	rule_list = list(unique_rules)
	rule_list.sort()
	return rule_list

# 7.2

def map_prod_rules(xml_filename, rules_list):
	""" Get the feature vector in the feature space of the rules_list

	Args:
	    xml_filename: a string representing the path of a XML file that may 
			contain elements with tags in the rules_list.
		rules_list: a list of production rules from 7.1.

	Returns:
	    A list of integers with the same size as rules_list. Each element
	    in the returned list represents existence of the corresponding rule
	"""

	file_rules = set()
	elem_name = "parse"
	start_tag, end_tag = "<"+elem_name+">", "</"+elem_name+">"
	tag_len = len(start_tag)
	with codecs.open(xml_filename, 'rU', 'utf-8') as f:
		for line in f:
			index_s, index_t = line.find(start_tag), line.find(end_tag)
			if index_s < 0: continue
			text_elem = line[index_s+tag_len:index_t]
			root_node = ProdTree()
			root_node.parse(text_elem)
			file_rules |= root_node.extract_rules()
	return [ 1 if rule in file_rules else 0 for rule in rules_list]

# 8

def get_generate_cluster_tables(brown_file_path):
	""" Generate cluster_code_list and word_cluster_mapping.
	
	Args:
		brown_file_path: a string representing the path to the brown cluster
			file. 

	Returns:
		A tuple of (cluster_code_list, word_cluster_mapping) defined below. 
	"""
	raw_brown = read_to_list(brown_file_path)
	cluster_code_set = set()
	word_cluster_mapping = {}
	for raw_line in raw_brown:
		line_segs = raw_line.split("\t")
		cluster_code_set.add(line_segs[0])
		word_cluster_mapping[line_segs[1]] = line_segs[0]
	return (list(cluster_code_set)+[u'8888'], word_cluster_mapping)


def map_brown_clusters(xml_file_path, cluster_code_list, word_cluster_mapping):
	""" Produce a representation that reflects the normalized frequency of 
		each known Brown cluster in the given text. For words that do not 
		appear in the precomputed Brown clusters, their code should be 8888.

	Args:
		xml_file_path: a string representing the path to the a XML file that is
			extracted by CoreNLP.
		cluster_code_list: a list of unique cluster names/codes present in 
			cluster file.
		word_cluster_mapping: a dict containing a mapping from words occurring 
		in the brown cluster file to their cluster codes.

	Returns: A vector (or list) of the same length of cluster code list. Each
		element in the output list takes the value of the number of times the 
		corresponding cluster in cluster code list appeared in the given text
		divided by the number of all words in the text.
	"""
	num_tokens = 0
	elem_cnt_dict = defaultdict(int)
	elem_name = "word"
	start_tag, end_tag = "<"+elem_name+">", "</"+elem_name+">"
	tag_len = len(start_tag)
	unk_cluster = u'8888'
	with codecs.open(xml_file_path, 'rU', 'utf-8') as f:
		for line in f:
			index_s, index_t = line.find(start_tag), line.find(end_tag)
			if index_s < 0: continue
			num_tokens += 1
			text_elem = line[index_s+tag_len:index_t]
			if text_elem in word_cluster_mapping:
				cluster_id = word_cluster_mapping[text_elem]
			else: cluster_id = unk_cluster
			elem_cnt_dict[cluster_id] += 1
	return [elem_cnt_dict[elem]*1.0/num_tokens for elem in cluster_code_list] 


# 9.2 


if __name__ == "__main__":
	pass
