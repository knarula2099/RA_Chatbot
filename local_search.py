import os
import pandas as pd
import tiktoken
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

INPUT_DIR = "output/20240909-090920/artifacts"
LANCEDB_URI = "lancedb"

COMMUNITY_REPORT_TABLE = "create_final_community_reports"
ENTITY_TABLE = "create_final_nodes"
ENTITY_EMBEDDING_TABLE = "create_final_entities"
RELATIONSHIP_TABLE = "create_final_relationships"
COVARIATE_TABLE = "create_final_covariates"
TEXT_UNIT_TABLE = "create_final_text_units"
COMMUNITY_LEVEL = 2


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path} Current working directory: {os.getcwd()}, visible files: {os.listdir()}")


def load_data():
    entity_file = os.path.join(INPUT_DIR, f"{ENTITY_TABLE}.parquet")
    check_file_exists(entity_file)
    entity_df = pd.read_parquet(entity_file)

    entity_embedding_file = os.path.join(
        INPUT_DIR, f"{ENTITY_EMBEDDING_TABLE}.parquet")
    check_file_exists(entity_embedding_file)
    entity_embedding_df = pd.read_parquet(entity_embedding_file)

    entities = read_indexer_entities(
        entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    description_embedding_store = LanceDBVectorStore(
        collection_name="entity_description_embeddings")
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    entity_description_embeddings = store_entity_semantic_embeddings(
        entities=entities, vectorstore=description_embedding_store)

    relationship_file = os.path.join(
        INPUT_DIR, f"{RELATIONSHIP_TABLE}.parquet")
    check_file_exists(relationship_file)
    relationship_df = pd.read_parquet(relationship_file)
    relationships = read_indexer_relationships(relationship_df)

    covariates_file = os.path.join(INPUT_DIR, f"{COVARIATE_TABLE}.parquet")
    check_file_exists(covariates_file)
    covariates_df = pd.read_parquet(covariates_file)
    claims = read_indexer_covariates(covariates_df)
    covariates = {"claims": claims}

    report_file = os.path.join(INPUT_DIR, f"{COMMUNITY_REPORT_TABLE}.parquet")
    check_file_exists(report_file)
    report_df = pd.read_parquet(report_file)
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    text_unit_file = os.path.join(INPUT_DIR, f"{TEXT_UNIT_TABLE}.parquet")
    check_file_exists(text_unit_file)
    text_unit_df = pd.read_parquet(text_unit_file)
    text_units = read_indexer_text_units(text_unit_df)

    return entities, relationships, reports, text_units, description_embedding_store, entity_description_embeddings, covariates


def create_search_engine(api_key, llm_model='gpt-4o-mini', embedding_model='text-embedding-3-small'):
    entities, relationships, reports, text_units, description_embedding_store, \
        entity_descriptions_embeddings, covariates = load_data()
    llm = ChatOpenAI(
        api_key=api_key,
        model=llm_model,
        api_type=OpenaiApiType.OpenAI,
        max_retries=20,
    )
    token_encoder = tiktoken.get_encoding("cl100k_base")
    text_embedder = OpenAIEmbedding(
        api_key=api_key,
        api_base=None,
        api_type=OpenaiApiType.OpenAI,
        model=embedding_model,
        deployment_name=embedding_model,
        max_retries=20,
    )
    context_builder = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        covariates=covariates,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )
    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }
    llm_params = {
        "max_tokens": 2_000,
        "temperature": 0.0,
    }
    search_engine = LocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type="Multiple Paragraphs",
    )
    return search_engine


def main():
    api_key = os.environ["GRAPHRAG_API_KEY"]
    search_engine = create_search_engine(api_key)
    query = input("Enter your query: ")
    response = search_engine.search(query)
    print(response.response)

    # print("Entities:")
    # entities = response.context_data['entities']
    # entities.to_csv("entities.csv")
    # print(entities)

    # print("Relationships:")
    # relationships = response.context_data['relationships']
    # relationships.to_csv("relationships.csv")
    # print(relationships)

    # print("Sources:")
    # sources = response.context_data['sources']
    # sources.to_csv("sources.csv")
    # print(sources)


if __name__ == '__main__':
    main()
