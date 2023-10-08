from typing import List, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection

class opensearch_utils():
    


    @classmethod
    def create_aws_opensearch_client(cls, region: str, host: str, http_auth: Tuple[str, str]) -> OpenSearch:

        client = OpenSearch(
            hosts=[
                {'host': host.replace("https://", ""),
                 'port': 443
                }
            ],
            http_auth=http_auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

        return client

    @classmethod
    def create_index(cls, os_client, index_name, index_body):
        '''
        인덱스 생성
        '''
        response = os_client.indices.create(
            index_name,
            body=index_body
        )
        print('\nCreating index:')
        print(response)

    @classmethod
    def check_if_index_exists(cls, os_client, index_name):
        '''
        인덱스가 존재하는지 확인
        '''
        exists = os_client.indices.exists(index_name)
        print(f"index_name={index_name}, exists={exists}")

        return exists

    @classmethod
    def add_doc(cls, os_client, index_name, document, id):
        '''
        # Add a document to the index.
        '''
        response = os_client.index(
            index = index_name,
            body = document,
            id = id,
            refresh = True
        )

        print('\nAdding document:')
        print(response)

    @classmethod
    def search_document(cls, os_client, query, index_name):
        response = os_client.search(
            body=query,
            index=index_name
        )
        print('\nSearch results:')
        return response

    @classmethod
    def delete_index(cls, os_client, index_name):
        response = os_client.indices.delete(
            index=index_name
        )

        print('\nDeleting index:')
        print(response)

    @classmethod
    def parse_keyword_response(cls, response, show_size=3):
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
                print(doc['_source']['metadata'])
                print("---------------------")
                if idx == show_size-1:
                    break
        else:
            print("There is no response")

    @classmethod
    def get_query(cls, **kwargs):

        # Reference:
        # OpenSearcj boolean query:
        #  - https://opensearch.org/docs/latest/query-dsl/compound/bool/
        # OpenSearch match qeury:
        #  - https://opensearch.org/docs/latest/query-dsl/full-text/index/#match-boolean-prefix

        min_shoud_match = 0
        if "minimum_should_match" in kwargs:
            min_shoud_match = kwargs["minimum_should_match"]

        QUERY_TEMPLAGE = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text": {
                                    "query": f'{kwargs["query"]}',
                                    "minimum_should_match": f'{min_shoud_match}%',
                                    "operator":  "or",
                                    # "fuzziness": "AUTO",
                                    # "fuzzy_transpositions": True,
                                    # "zero_terms_query": "none",
                                    # "lenient": False,
                                    # "prefix_length": 0,
                                    # "max_expansions": 50,
                                    # "boost": 1
                                }
                            }
                        },
                    ],
                    "filter": [
                    ]
                }
            }
        }

        if "filter" in kwargs:
            QUERY_TEMPLAGE["query"]["bool"]["filter"].extend(kwargs["filter"])

        return QUERY_TEMPLAGE
