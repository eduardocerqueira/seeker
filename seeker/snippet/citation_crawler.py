#date: 2025-03-11T16:50:59Z
#url: https://api.github.com/gists/682e87c4a8f72f9af95b84c7438a32bf
#owner: https://api.github.com/users/hclent

"""
NLPSec Semantic Scholar Citation Crawler
"""
import csv
import time 
import logging
import requests

############### API settings #####

api_key = 'YOUR_API_KEY_HERE'  # Replace with your API key
headers = {'x-api-key': api_key} # Define headers with API key

############### Initalize csv files #####

hit = open('hits_security.csv', 'a', newline='\n')
miss = open('misses_security.csv', 'a', newline='\n')
err = open('errors_security.csv', 'a', newline='\n')

hits_writer = csv.writer(hit)
miss_writer = csv.writer(miss)
err_writer = csv.writer(err)

total_citations = 0

############### NB #####

"""
Rate limit:
1 request per second for the following endpoints:
/paper/batch
/paper/search
/recommendations
10 requests / second for all other calls
"""

############### Example of Getting the paper ID's ##### 

# base_url = "https://api.semanticscholar.org/graph/v1"
# resource_path="paper/search"
# query_params = {"query": "THE NAME OF THE PAPER HERE"}
# url = f"{base_url}/{resource_path}"
# response = requests.get(url, params=query_params, headers=headers)
# if response.status_code == 200:
#    response_data = response.json()
#    # Process and print the response data as needed
#    print(response_data)
# else:
#    print(f"Request failed with status code {response.status_code}: {response.text}")

############### Check Academic Graph ########################

def check_terms(some_text): #some_text is all lower_case
	key_terms = ["attack", "defense", "defence", "security", "secur"]

	if "security" in some_text: #if the word 'security' is in the text, return true
		return (True, "security")
	else: 
		word_list = some_text.split(" ") #otherwise we will split the text on white space
		word_list = [w.lstrip() for w in word_list] #trim off any extra spaces to the left
		word_list = [w.strip() for w in word_list] #and to the right
		for word in word_list:
			for term in key_terms:
				if word.startswith(term): #then we will check all the words in the text to see if any start with our terms.
					return(True, term)

	return (False, None)


def search_title_abst(json_data_list, paperId, url, params):
	"""
	Input: Json[{},{},...{}] from response.json['data']

	Output: prints to csv
	"""

	key_terms = ["attack", "attacks", "defense", "defence", "security"]

	for entry in json_data_list:
		citing_id = entry['citingPaper']['paperId']
		title = entry['citingPaper']['title'].lower()
		abstract = entry['citingPaper']['abstract'] # may be None !!!! 
		year = entry['citingPaper']['year']
		venue = entry['citingPaper']['venue']

		title_check = check_terms(title) #returns Tuple(Bool, match_term|None)

		if abstract:
			abstract = abstract.lower()
			abstract_check = check_terms(abstract)
		else:
			abstract_check = (False, None)
	
		#write to csv if it meets the criteria. 
		if title_check[0] and abstract_check[0]:
			hits_writer.writerow([paperId, citing_id, title, abstract, year, venue,  "Both Title and Abstract", f"`{title_check[1]}` and `{abstract_check[1]}`", url, params])
		elif title_check[0]:
			hits_writer.writerow([paperId, citing_id, title, abstract, year, venue, "Title", f"`{title_check[1]}`", url, params])
		elif abstract_check[0]:
			hits_writer.writerow([paperId, citing_id, title, abstract, year, venue, "Abstract", f"`{abstract_check[1]}`", url, params])
		else:
			miss_writer.writerow([paperId, citing_id, title])



def get_paper_details(paperId):
	time.sleep(3)
	url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}"
	paper_data_query_params = {'fields': 'title,year,abstract,authors,venue'}

	# Send the API request and store the response in a variable
	response = requests.get(url, params=paper_data_query_paramsm, headers=headers)
	if response.status_code == 200:
		return response.json()
	else:
		#logging.debug(f"*[failed]:[get_paper_details]:[{response.status_code}]:{paperId}")
		err_writer.writerow([paperId,"f(get_paper_details)",response.status_code, str(url), str(paper_data_query_params)])
		return None


def get_number_ctations(paperId):
	#"citationCount"
	time.sleep(3)
	url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}"
	paper_data_query_params = {'fields': 'citationCount'}
	response = requests.get(url, params=paper_data_query_params, headers=headers)
	if response.status_code == 200:
		n = response.json()['citationCount']
		return n
	else:
		#logging.debug(f"*[failed]:[get_number_citations]:[{response.status_code}]:{paperId}")
		err_writer.writerow([paperId,"f(get_number_citations)",response.status_code, str(url), str(paper_data_query_params)])
		return None


def get_citation_queries(paperId, n_papers):
	"""
	paperId: String -- this is the id associated with the paper in semantic scholar. 
	n_papers: Int -- this is the number of citations for the pap 
	"""
	url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}/citations"

	#### Step 1: Lets generate the entire list of queries 
	#### We are allowed 10 requests / second for all other calls

	remainder = n_papers % 10 #All API calls will be by 10, except the last <remainder> papers
	#print(remainder)
	every_10 = n_papers - remainder

	offsets = [x for x in range(0, every_10, 10)] #IF there is no remainder, this will get us exactly the full number 
	#print(offsets)

	list_of_queries = []

	for o in offsets:
		if remainder == 0:
			paper_data_query_params = {'fields': 'title,year,abstract,authors,venue',
										'offset': o,
										'limit': 10,
										}
			#print(f"[No remainder] {o}--{o+10}")

		else:
			if o != offsets[-1]:
				paper_data_query_params = {'fields': 'title,year,abstract,authors,venue',
											'offset': o,
											'limit': 10,
											}
				#print(f"[Yes remainder] {o}--{o+10}")


			else: #Then for the last one the offset will be offsets[-1], with limit remainder
				#print(f"[!!!] REMAINDER")
				#print(f"{o+10} + {remainder}")
				paper_data_query_params = {'fields': 'title,year,abstract,authors,venue',
								'offset': o+10,
								'limit': remainder,
								}
		
		list_of_queries.append(paper_data_query_params)

	return list_of_queries


def search_citations(list_of_queries, n_papers, paperId):
	counter = 0
	print(list_of_queries[counter])
	url = f"https://api.semanticscholar.org/graph/v1/paper/{paperId}/citations"
	#print(url)
	#print(headers)
	#counter += 1
	#execute search query
	for q in list_of_queries:
		time.sleep(5) 
		counter += 1
		print(f"* {counter}/{len(list_of_queries)} - {q}")
		response = requests.get(url, params=q, headers=headers)
		if response.status_code == 200:
			print(response.status_code)
			json_data_list = response.json()['data']
			print(len(json_data_list))
			search_title_abst(json_data_list, paperId, url, q)
		else:
			err_writer.writerow([paperId,"f(search_citations)",response.status_code,response.text,str(url), str(q)])
			return None



def main(paper):
	target_paperId = paper['paperId']
	n = get_number_ctations(target_paperId)
	print(f"* {n} citations ... ")
	list_of_queries = get_citation_queries(target_paperId, n)
	print(list_of_queries[0])
	print(f"* {len(list_of_queries)} queries to make ...")
	print("-------------------")
	search_citations(list_of_queries, n, target_paperId)




#NLP papers
papers = [
	{'paperId': '87f40e6f3022adbc1f1905e3e506abad05a9964f', 'title': 'Distributed Representations of Words and Phrases and their Compositionality'}, 
	{'paperId': 'f6b51c8753a871dc94ff32152c00c01e94f90f09', 'title': 'Efficient Estimation of Word Representations in Vector Space'}, 
	{'paperId': 'df2b0e26d0599ce3e70df8a9da02e51594e0e992', 'title': 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'},
	{'paperId': 'a54b56af24bb4873ed0163b77df63b92bd018ddc', 'title': 'DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter'},
	{'paperId': '077f8329a7b6fa3b7c877a57b81eb6c18b5f87de', 'title': 'RoBERTa: A Robustly Optimized BERT Pretraining Approach'},
	{'paperId': '6c4b76232bb72897685d19b3d264c6ee3005bc2b', 'title': 'Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer'},
	{'paperId': '964bd39b546f0f6625ff3b9ef1083f797807ef2e', 'title': 'BLOOM: A 176B-Parameter Open-Access Multilingual Language Model'},
	{'paperId': '57e849d0de13ed5f91d086936296721d4ff75a75', 'title': 'LLaMA: Open and Efficient Foundation Language Models'},
	{'paperId': '104b0bb1da562d53cbda87aec79ef6a2827d191a', 'title': 'Llama 2: Open Foundation and Fine-Tuned Chat Models'},
	{'paperId': 'db633c6b1c286c0386f0078d8a2e6224e03a6227', 'title': 'Mistral 7B'},
	{'paperId': 'ace2b6367a067898f66a33fca19581ebe71fccc5', 'title': 'GPT-4 Technical Report'},
]


print(total_citations)

for paper in papers: 
	main(paper)
	target_paperId = paper['paperId']
	n = get_number_ctations(target_paperId)
	total_citations += n


hit.close()
miss.close()
err.close()

print(f"**** total_citations: {total_citations} **********")
