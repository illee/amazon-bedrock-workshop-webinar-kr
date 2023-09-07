from utils import print_ww
import pandas as pd

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

def check_if_index_exists(aws_client, index_name):
    '''
    인덱스가 존재하는지 확인
    '''
    exists = aws_client.indices.exists(index_name)
    print(f"index_name={index_name}, exists={exists}")
    return exists

def delete_index(aws_client, index_name):
    response = aws_client.indices.delete(
        index = index_name
    )

    print('\nDeleting index:')
    print(response)
    