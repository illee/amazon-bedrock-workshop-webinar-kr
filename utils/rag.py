############################################################    
############################################################    
# RAG 관련 함수들
############################################################    
############################################################    


from utils import print_ww
import pandas as pd
import pickle

def to_pickle(data, path):
    
    with open(path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def load_pickle(path):
    
    with open(path, "rb") as f:
        data = pickle.load(f)
        

def show_context_used(context_list, limit=10):
    for idx, context in enumerate(context_list):
        if idx < limit:
            print("-----------------------------------------------")                
            print(f"{idx+1}. Chunk: {len(context.page_content)} Characters")
            print("-----------------------------------------------")        
            print_ww(context.page_content)        
            print_ww("metadata: \n", context.metadata)                    
        else:
            break

def show_chunk_stat(documents):
    doc_len_list = [len(doc.page_content) for doc in documents]
    print(pd.DataFrame(doc_len_list).describe())
    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents])//len(documents)
    avg_char_count_pre = avg_doc_length(documents)
    print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
    
    max_idx = doc_len_list.index(max(doc_len_list))
    print("\nShow document at maximum size")
    print(documents[max_idx].page_content)    


###########################################    
### 1.2 [한국어 임베딩벡터 모델] SageMaker 임베딩 벡터 모델 KoSimCSE-roberta endpiont handler
###########################################    

### SagemakerEndpointEmbeddingsJumpStart클래스는 SagemakerEndpointEmbeddings를 상속받아서 작성

# 매개변수 (Parameters):
# * texts: 임베딩을 생성할 텍스트의 리스트입니다.
# * chunk_size: 한 번의 요청에 그룹화될 입력 텍스트의 수를 정의합니다. 만약 None이면, 클래스에 지정된 청크 크기를 사용합니다.

# Returns:
# * 각 텍스트에 대한 임베딩의 리스트를 반환
###########################################    

from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.retrievers import AmazonKendraRetriever
from typing import Any, Dict, List, Optional
import json
import numpy as np

class SagemakerEndpointEmbeddingsJumpStart(SagemakerEndpointEmbeddings):
    def embed_documents(self, texts: List[str], chunk_size: int=1) -> List[List[float]]:
        """Compute doc embeddings using a SageMaker Inference Endpoint.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size defines how many input texts will
                be grouped together as request. If None, will use the
                chunk size specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = len(texts) if chunk_size > len(texts) else chunk_size
        
        print("text size: ", len(texts))
        print("_chunk_size: ", _chunk_size)

        for i in range(0, len(texts), _chunk_size):
            
            #print (i, texts[i : i + _chunk_size])
            response = self._embedding_func(texts[i : i + _chunk_size])
            #print (i, response, len(response[0].shape))
            
            results.extend(response)
        return results    
    
class KoSimCSERobertaContentHandler(EmbeddingsContentHandler):
    
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs={}) -> bytes:
        
        input_str = json.dumps({"inputs": prompt, **model_kwargs})
        
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        
        response_json = json.loads(output.read().decode("utf-8"))
        ndim = np.array(response_json).ndim    
        
        if ndim == 4:
            # Original shape (1, 1, n, 768)
            emb = response_json[0][0][0]
            emb = np.expand_dims(emb, axis=0).tolist()
        elif ndim == 2:
            # Original shape (n, 1)
            emb = []
            for ele in response_json:
                e = ele[0][0]
                emb.append(e)
        else:
            print(f"Other # of dimension: {ndim}")
            emb = None
        return emb    
    
    
def opensearch_pretty_print_documents(response):
    '''
    OpenSearch 결과인 LIST 를 파싱하는 함수
    '''
    for doc, score in response:
        print(f'\nScore: {score}')
        print(f'Document Number: {doc.metadata["row"]}')

        # Split the page content into lines
        lines = doc.page_content.split("\n")

        # Extract and print each piece of information if it exists
        for line in lines:
            split_line = line.split(": ")
            if len(split_line) > 1:
                print(f'{split_line[0]}: {split_line[1]}')

        print("Metadata:")
        print(f'Type: {doc.metadata["type"]}')
        print(f'Source: {doc.metadata["source"]}')        
                
        print('-' * 50)
    
def opensearch_pretty_print_documents_wo_filter(response):
    '''
    OpenSearch 결과인 LIST 를 파싱하는 함수
    '''
    for doc, score in response:
        print(f'\nScore: {score}')
        print(f'Document Number: {doc.metadata["row"]}')

        # Split the page content into lines
        lines = doc.page_content.split("\n")

        # Extract and print each piece of information if it exists
        for line in lines:
            split_line = line.split(": ")
            if len(split_line) > 1:
                print(f'{split_line[0]}: {split_line[1]}')
                
        print('-' * 50)

    
def get_embedding_model(boto3_bedrock, is_bedrock_embeddings, is_KoSimCSERobert, aws_region, endpont_name=None):
    '''
    Bedrock embeeding model or KoSimCSERobert model 가져오기
    '''
    if is_bedrock_embeddings:

        # We will be using the Titan Embeddings Model to generate our Embeddings.
        from langchain.embeddings import BedrockEmbeddings
        # llm_emb = BedrockEmbeddings(client=boto3_bedrock)
        llm_emb = BedrockEmbeddings(
          client=boto3_bedrock,
          model_id = "amazon.titan-embed-g1-text-02" # amazon.titan-e1t-medium, amazon.titan-embed-g1-text-02
        )        
        print("Bedrock Embeddings Model Loaded")
    elif is_KoSimCSERobert:
        LLMEmbHandler = KoSimCSERobertaContentHandler()
        endpoint_name_emb = endpont_name
        llm_emb = SagemakerEndpointEmbeddingsJumpStart(
            endpoint_name=endpoint_name_emb,
            region_name=aws_region,
            content_handler=LLMEmbHandler,
        )        
        print("KoSimCSERobert Embeddings Model Loaded")
    else:
        llm_emb = None
        print("No Embedding Model Selected")
    
    return llm_emb

    
############################################################    
# OpenSearch Client
############################################################    
    
from typing import List, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection

def create_aws_opensearch_client(region: str, host: str, http_auth: Tuple[str, str]) -> OpenSearch:
    '''
    오픈서치 클라이언트를 제공함.
    '''
    aws_client = OpenSearch(
        hosts = [{'host': host.replace("https://", ""), 'port': 443}],
        http_auth = http_auth,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    
    return aws_client

def create_index(aws_client, index_name, index_body):    
    '''
    인덱스 생성
    '''
    response = aws_client.indices.create(
        index_name,
        body=index_body
    )
    print('\nCreating index:')
    print(response)

    
def check_if_index_exists(aws_client, index_name):
    '''
    인덱스가 존재하는지 확인
    '''
    exists = aws_client.indices.exists(index_name)
    print(f"index_name={index_name}, exists={exists}")
    return exists


def add_doc(aws_client, index_name, document, id):
    '''
    # Add a document to the index.
    '''
    response = aws_client.index(
        index = index_name,
        body = document,
        id = id,
        refresh = True
    )

    print('\nAdding document:')
    print(response)

def search_document(aws_client, query, index_name):
    response = aws_client.search(
        body=query,
        index=index_name
    )
    print('\nSearch results:')
    # print(response)
    return response
    

def delete_index(aws_client, index_name):
    response = aws_client.indices.delete(
        index = index_name
    )

    print('\nDeleting index:')
    print(response)



def generate_opensearch_AndQuery(question):
    '''
    주어진 앱력을 키워드로 분리하고 AND 조건으로 바꾸어 주는 쿼리 생성
    '''
    keywords = question.split(' ')
    query = {
        "query": {
            "bool": {
                "must": []
            }
        }
    }
    
    for keyword in keywords:
        query["query"]["bool"]["must"].append({
            "match": {
                "text": keyword
            }
        })
    
    # return query
    return json.dumps(query, indent=2, ensure_ascii=False)


def parse_keyword_response(response, show_size=3):
    '''
    키워드 검색 결과를 보여 줌.
    '''
    length = len(response['hits']['hits'])
    if length >= 1:
        print("# of searched docs: ", length)
        print(f"# of display: {show_size}")        
        print("---------------------")        
        for idx, doc in enumerate(response['hits']['hits']):
            print("_id in index: " , doc['_id'])            
            print(doc['_score'])            
            print(doc['_source']['text'])
            print("---------------------")
            if idx == show_size-1:
                break
    else:
        print("There is no response")

    
    
############################################################    
# Hybrid Search
############################################################    


from itertools import chain as ch
from typing import Any, Dict, List
from langchain.schema import Document

def ensemble_results(doc_lists: List[List[Document]], weights, algorithm="RRF", c=60, k=5) -> List[Document]:

    assert algorithm in ["RRF", "simple_weighted"]

    # Create a union of all unique documents in the input doc_lists
    all_documents = set()

    for doc_list in doc_lists:
        for (doc, _) in doc_list:
            all_documents.add(doc.page_content)

    # Initialize the score dictionary for each document
    hybrid_score_dic = {doc: 0.0 for doc in all_documents}

    # Calculate RRF scores for each document
    for doc_list, weight in zip(doc_lists, weights):
        for rank, (doc, score) in enumerate(doc_list, start=1):
            if algorithm == "RRF": # RRF (Reciprocal Rank Fusion)
                score = weight * (1 / (rank + c))
            elif algorithm == "simple_weighted":
                score *= weight
            hybrid_score_dic[doc.page_content] += score

    # Sort documents by their scores in descending order
    sorted_documents = sorted(
        hybrid_score_dic.items(), key=lambda x: x[1], reverse=True
    )

    # Map the sorted page_content back to the original document objects
    page_content_to_doc_map = {
        doc.page_content: doc for doc_list in doc_lists for (doc, orig_score) in doc_list
    }

    sorted_docs = [
        (page_content_to_doc_map[page_content], hybrid_score) for (page_content, hybrid_score) in sorted_documents
    ]

    return sorted_docs[:k]



def interpolate_results(semantic, keyword, k, verbose=False):
    '''
    Semantic, Keyword 의 두개의 검색 셋트에서 Top K 개를 제공 함.
    '''
    semantic_set = set([doc.page_content for doc, score in semantic])
    keyword_set = set([doc.page_content for doc, score in keyword])
    common = semantic_set.intersection(keyword_set)

    results = []
    for doc, score in list(ch(semantic, keyword)):
        # print("######## doc, score ###########")
        # print(doc, score)
        # print("###############################")
        
        if doc.page_content in common:
            
            target = doc.page_content
            semantic_score = get_score(target, semantic, verbose=False)
            keyword_score = get_score(target, keyword, verbose=False)
            
            total_score = semantic_score + keyword_score
            
            if verbose:
                print("semantic_score: ", round(semantic_score, 4))
                print("keyword_score: ", round(keyword_score, 4))
                print("total_score: ", round(total_score, 4))

            is_processed = is_element(target, results, verbose=False)
            if is_processed: pass # 이미 중복된 것 한번 처리 했으면 스킵
            else: results.append((doc, round(total_score,4)))
        else:
            results.append((doc, score))

    if verbose:
        print("######## unique_set results ###########")
        print(results)
        print("###############################")

    top_result = sort_score_search_results(results)
    # print("top_result: \n", top_result)
    top_result = top_result[:k]

    return top_result

def get_score(page_content, document, verbose=False):
    '''
    주어진 page_content 이 document 안에 있으면 해당 스코어를 제공 함.
    '''
    # print("######## get_scroe ###########")

    for doc, socre in document:
        # print("document: \n", doc)
        if page_content == doc.page_content:
            if verbose:
                print("Identical: ")
                print("document: \n", doc.page_content)
                print("page_content: \n", page_content)
                print("socre: \n", socre)
            return socre
        else:
            if verbose:
                print("Not Identical")
    return None

def is_element(page_content, document, verbose=False):
    '''
    주어진 page_content 이 document 안에 있으면 True, 아니면 False
    '''

    for doc, score in document:
        # print("document: \n", doc)
        if page_content == doc.page_content:
            if verbose:
                print("Identical: ")
                print("document: \n", doc.page_content)
                print("page_content: \n", page_content)
            return True
    
        else:
            if verbose:
                print("Not Identical")
    return False

def sort_score_search_results(search_result):

    df = pd.DataFrame(search_result, columns=["document","score"])
    df = df.sort_values(by=["score"], ascending=False)

    top_result = []
    for index, row in df.iterrows():
        doc = row[0]
        score = row[1]
        doc.metadata['hybrid_socre'] = round(score,3)
        top_result.append(doc)

    return top_result

############################################################    
# Search Function
############################################################    

# def create_bool_filter(filter01, filter02):
#     boolean_filter = {
#         "bool": {
#           "must": [
#             {
#               "match": {
#                 "metadata.type": f"{filter01}"
#               }
#             }
#           ],
#           "filter": {
#             "term": {
#               "metadata.source": f"{filter02}"
#             }
#           }
#         }
#     }
    
#     return boolean_filter

from langchain.chains import RetrievalQA

def run_RetrievalQA(**kwargs):

    chain_types = ["stuff", "map_reduce", "refine"]

    assert "llm" in kwargs, "Check your llm"
    assert "query" in kwargs, "Check your query"
    assert "prompt" in kwargs, "Check your prompt"
    assert "vector_db" in kwargs, "Check your vector_db"
    assert kwargs.get("chain_type", "stuff") in chain_types, f'Check your chain_type, {chain_types}'

    qa = RetrievalQA.from_chain_type(
        llm=kwargs["llm"],
        chain_type=kwargs.get("chain_type", "stuff"),
        retriever=kwargs["vector_db"].as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": kwargs.get("k", 5),
                "boolean_filter": kwargs.get("boolean_filter", {})
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": kwargs["prompt"],
            "verbose": kwargs.get("verbose", False),
        },
        verbose=kwargs.get("verbose", False)
    )

    return qa(kwargs["query"])


# def run_RetrievalQA(query, boolean_filter, llm_text, vectro_db, PROMPT, verbose, is_filter, k):
    
#     if is_filter:
#         qa = RetrievalQA.from_chain_type(
#             llm=llm_text,
#             chain_type="stuff",
#             retriever=vectro_db.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={
#                     "k":k,
#                     "boolean_filter":boolean_filter
#                 }
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={
#                 "prompt":PROMPT,
#                 "verbose":verbose,
#             },
#             verbose=verbose
#         )
        
#     else:
#         qa = RetrievalQA.from_chain_type(
#             llm=llm_text,
#             chain_type="stuff",
#             retriever=vectro_db.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={
#                     "k":k
#                 }
#             ),
#             return_source_documents=True,
#             chain_type_kwargs={
#                 "prompt": PROMPT,
#                 "verbose": verbose,
#             },
#             verbose=verbose
#         )

#     result = qa(query)

#     return result


def run_RetrievalQA_kendra(query, llm_text, PROMPT, kendra_index_id, k, aws_region, verbose):
    qa = RetrievalQA.from_chain_type(
        llm=llm_text,
        chain_type="stuff",
        retriever=AmazonKendraRetriever(
            index_id=kendra_index_id,
            region_name=aws_region,
            top_k=k,
            attribute_filter = {
                "EqualsTo": {      
                    "Key": "_language_code",
                    "Value": {
                        "StringValue": "ko"
                    }
                },
            }
        ),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "verbose": verbose,
        },
        verbose=verbose
    )
    
    result = qa(query)

    return result 


# def create_keyword_bool_filter(query, filter01, filter02, k):
#     boolean_filter = {
#             "size": k,
#             "query": {
#                 "bool": {
#                   "must": [
#                     {
#                       "match": {
#                         "text": query
#                       }
#                     },
#                     {
#                       "match": {
#                         "metadata.type": filter01
#                       }
#                     }
#                   ],
#                   "filter": [
#                     {
#                       "term": {
#                         "metadata.source.keyword": filter02
#                       }
#                     }
#                   ]
#                 }
#             }            
#         }   
    
#     return boolean_filter


from pprint import pprint
from langchain.schema import Document
from utils.opensearch import opensearch_utils

# def get_similiar_docs_with_keyword_score(**kwargs):

#     assert "query" in kwargs, "Check your query"
#     assert "k" in kwargs, "Check your k"
#     assert "os_client" in kwargs, "Check your os_client"
#     assert "index_name" in kwargs, "Check your index_name"

#     def normalize_search_formula(score, max_score):

#         return score / max_score

#     def normalize_search_results(search_results):

#         hits = (search_results["hits"]["hits"])
#         max_score = search_results["hits"]["max_score"]
#         for hit in hits:
#             hit["_score"] = normalize_search_formula(hit["_score"], max_score)
#         search_results["hits"]["max_score"] = hits[0]["_score"]
#         search_results["hits"]["hits"] = hits
#         return search_results

#     query = opensearch_utils.get_query(
#         query=kwargs["query"],
#         minimum_should_match=kwargs.get("minimum_should_match", 0),
#         filter=kwargs.get("filter", [])
#     )
#     query["size"] = kwargs["k"]

#     print ("keyword search query: ")
#     pprint (query)

#     search_results = opensearch_utils.search_document(
#         os_client=kwargs["os_client"],
#         query=query,
#         index_name=kwargs["index_name"]
#     )

#     results = []
#     if search_results["hits"]["hits"]:
#         search_results = normalize_search_results(search_results)
#         for res in search_results["hits"]["hits"]:

#             metadata = res["_source"]["metadata"]
#             metadata["score"] = res["_score"]
#             metadata["id"] = res["_id"]

#             doc = Document(
#                 page_content=res["_source"]["text"],
#                 metadata=metadata
#             )
#             results.append((doc))

#     return results


import copy
from operator import itemgetter

def get_semantic_similar_docs(**kwargs):

    search_types = ["approximate_search", "script_scoring", "painless_scripting"]
    space_types = ["l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit"]

    assert "vector_db" in kwargs, "Check your vector_db"
    assert "query" in kwargs, "Check your query"
    assert kwargs.get("search_type", "approximate_search") in search_types, f'Check your search_type: {search_types}'
    assert kwargs.get("space_type", "l2") in space_types, f'Check your space_type: {space_types}'

    results = kwargs["vector_db"].similarity_search_with_score(
            query=kwargs["query"],
            k=kwargs.get("k", 5),
            search_type=kwargs.get("search_type", "approximate_search"),
            space_type=kwargs.get("space_type", "l2"),
            boolean_filter=kwargs.get("boolean_filter", {}),
            # fetch_k=3,
        )
    
    if kwargs.get("hybrid", False):
        max_score = results[0][1]
        new_results = []
        for doc in results:
            nomalized_score = float(doc[1]/max_score)
            new_results.append((doc[0], nomalized_score))
        results = copy.deepcopy(new_results)

    return results

def get_keyword_similar_docs(**kwargs):

    assert "query" in kwargs, "Check your query"
    assert "k" in kwargs, "Check your k"
    assert "os_client" in kwargs, "Check your os_client"
    assert "index_name" in kwargs, "Check your index_name"

    def normalize_search_formula(score, max_score):

        return score / max_score

    def normalize_search_results(search_results):

        hits = (search_results["hits"]["hits"])
        max_score = search_results["hits"]["max_score"]
        for hit in hits:
            hit["_score"] = normalize_search_formula(hit["_score"], max_score)
        search_results["hits"]["max_score"] = hits[0]["_score"]
        search_results["hits"]["hits"] = hits
        return search_results

    query = opensearch_utils.get_query(
        query=kwargs["query"],
        minimum_should_match=kwargs.get("minimum_should_match", 0),
        filter=kwargs.get("filter", [])
    )
    query["size"] = kwargs["k"]

    print ("keyword search query: ")
    pprint (query)

    search_results = opensearch_utils.search_document(
        os_client=kwargs["os_client"],
        query=query,
        index_name=kwargs["index_name"]
    )

    results = []
    if search_results["hits"]["hits"]:
        search_results = normalize_search_results(search_results)
        for res in search_results["hits"]["hits"]:

            metadata = res["_source"]["metadata"]
            #metadata["score"] = res["_score"]
            metadata["id"] = res["_id"]

            doc = Document(
                page_content=res["_source"]["text"],
                metadata=metadata
            )
            if kwargs.get("hybrid", False):
                results.append((doc, res["_score"]))
            else:
                results.append((doc))

    return results



def get_similiar_docs(query, vectro_db, is_filter, boolean_filter, weight_decay_rate=0, k=5):    
    '''
    weight_decay_rate: 벡터 서치를 통해서 나온 스코어에 표준화를 한 스코어의 가중치를 낮추기 위한 값임.
    수식은 (1 - weight_decay_rate) 임. 예를 들어서 weight_decay_rate = 0.1 이면 1 - 0.1 = 0.9 를 곱하여 기존 스코어를 감소 시킴.
    '''
    # query = f'{store}, {query}'
    # store = "*" + store.replace("이마트", "").strip() + "*"
    # print("query: ", query)
    # print (query, search_type, store)

    if is_filter:
        pre_similar_doc = vectro_db.similarity_search_with_score(
            query,
            k=k,
            # fetch_k=3,
            search_type="approximate_search", # approximate_search, script_scoring, painless_scripting
            space_type="l2",     #"l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit";
            boolean_filter= boolean_filter
        )
    else:
        pre_similar_doc = vectro_db.similarity_search_with_score(
            query,
            k=k,
            # fetch_k=3,
            search_type="approximate_search", # approximate_search, script_scoring, painless_scripting
            space_type="l2",     #"l2", "l1", "linf", "cosinesimil", "innerproduct", "hammingbit";
        )
        

    
    # print("################################")
    # print("similar_docs: \n", similar_docs)
    # print("################################")    
    
    similar_docs_copy = copy.deepcopy(pre_similar_doc)
    
#    print("similar_docs_copy: ", similar_docs_copy)
    
    # 전체 결과의 스코어에 대해서 표준화를 하여 새로운 점수를 할당 함. 시행 함.
    weight_decay_value =  1 - weight_decay_rate
    if len(similar_docs_copy) != 0:
        max_score = max(similar_docs_copy, key=itemgetter(1))[1]
        similar_docs_copy = [(doc[0], ( doc[1] * weight_decay_value ) / max_score) for doc in similar_docs_copy]
        
    return similar_docs_copy

def get_similiar_docs_with_keywords(query, aws_client, index_name, weight_decay_rate=0, is_filter=True, filter01=None, filter02=None, k=10):

    def normalize_search_formula(score, max_score, weight_decay_rate):
        weight_decay_value = 1 - weight_decay_rate
        return ( score * weight_decay_value ) / max_score

    def normalize_search_results(search_results, weight_decay_rate):
        hits = (search_results["hits"]["hits"])
        max_score = search_results["hits"]["max_score"]
        for hit in hits:
            hit["_score"] = normalize_search_formula(hit["_score"], max_score, weight_decay_rate)
        search_results["hits"]["max_score"] = hits[0]["_score"]
        search_results["hits"]["hits"] = hits
        return search_results

    if is_filter:
        search_query = create_keyword_bool_filter(query, filter01, filter02, k)
        # print("search_query: \n", search_query)
        
    else:
        search_query = {
            "size": k,
            "query": {
                "match": {
                    "text": query
                }
            },
            "_source": ["text"],
        }
    
    search_results = aws_client.search(body=search_query, index=index_name)
    # print("###############")
    # print("search_query: \n", search_query)    
    # print("search_results: \n", search_results)
    # print("###############")    
    
    results = []
    if search_results["hits"]["hits"]:
        search_results = normalize_search_results(search_results, weight_decay_rate)
        for res in search_results["hits"]["hits"]:
            source = res["_source"]["text"].rsplit("\n", 2)[-1].split("Source: ")[-1]
            doc = Document(
                page_content=res["_source"]["text"],
                metadata={'source': source}
            )
            results.append((doc, res["_score"]))

    return results

from langchain.schema import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

class BM25OpenSearchRetriever(BaseRetriever):
    
    client: Any
    index_name: str
    is_filter: bool
    filter01: str
    filter02: str
    k: int
            
    def create_keyword_bool_filter(self, query, filter01, filter02, k):
        
        boolean_filter = {
                "size": k,
                "query": {
                    "bool": {
                      "must": [
                        {
                          "match": {
                            "text": query
                          }
                        },
                        {
                          "match": {
                            "metadata.type": filter01
                          }
                        }
                      ],
                      "filter": [
                        {
                          "term": {
                            "metadata.source.keyword": filter02
                          }
                        }
                      ]
                    }
                }            
            }   

        return boolean_filter

    def normalize_search_formula(self, score, max_score):
        return score / max_score

    def normalize_search_results(self, search_results):
        hits = (search_results["hits"]["hits"])
        max_score = search_results["hits"]["max_score"]
        for hit in hits:
            hit["_score"] = self.normalize_search_formula(hit["_score"], max_score)
        search_results["hits"]["max_score"] = hits[0]["_score"]
        search_results["hits"]["hits"] = hits
        return search_results
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        
        if self.is_filter: search_query = self.create_keyword_bool_filter(query, self.filter01, self.filter02, self.k)
        else:
            search_query = {
                "size": k,
                "query": {
                    "match": {
                        "text": query
                    }
                },
                "_source": ["text"],
            }
            
        search_results = self.client.search(
            body=search_query,
            index=self.index_name
        )
        
        results = []
        if search_results["hits"]["hits"]:
            search_results = self.normalize_search_results(search_results)
            for res in search_results["hits"]["hits"]:
                source = res["_source"]["text"].rsplit("\n", 2)[-1].split("Source: ")[-1]
                doc = Document(
                    page_content=res["_source"]["text"],
                    metadata={'source': source, 'score': res["_score"]}
                )
                results.append((doc))
                
        
        return results


###########################################    
### Chatbot Functions
###########################################    

# turn verbose to true to see the full logs and documents
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import BaseMessage


# We are also providing a different chat history retriever which outputs the history as a Claude chat (ie including the \n\n)
_ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}
def _get_chat_history(chat_history):
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "\n\nHuman: " + dialogue_turn[0]
            ai = "\n\nAssistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer


import ipywidgets as ipw
from IPython.display import display, clear_output

class ChatUX:
    """ A chat UX using IPWidgets
    """
    def __init__(self, qa, retrievalChain = False):
        self.qa = qa
        self.name = None
        self.b=None
        self.retrievalChain = retrievalChain
        self.out = ipw.Output()


    def start_chat(self):
        print("Starting chat bot")
        display(self.out)
        self.chat(None)


    def chat(self, _):
        if self.name is None:
            prompt = ""
        else: 
            prompt = self.name.value
        if 'q' == prompt or 'quit' == prompt or 'Q' == prompt:
            print("Thank you , that was a nice chat !!")
            return
        elif len(prompt) > 0:
            with self.out:
                thinking = ipw.Label(value="Thinking...")
                display(thinking)
                try:
                    if self.retrievalChain:
                        result = self.qa.run({'question': prompt })
                    else:
                        result = self.qa.run({'input': prompt }) #, 'history':chat_history})
                except:
                    result = "No answer because some errors occurredr"
                thinking.value=""
                print_ww(f"AI:{result}")
                self.name.disabled = True
                self.b.disabled = True
                self.name = None

        if self.name is None:
            with self.out:
                self.name = ipw.Text(description="You:", placeholder='q to quit')
                self.b = ipw.Button(description="Send")
                self.b.on_click(self.chat)
                display(ipw.Box(children=(self.name, self.b)))