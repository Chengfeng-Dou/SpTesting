import time
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import config


def _split_document(name):
    loader = DirectoryLoader(
        f'script/{name}',
        glob='**/*.json',
        loader_cls=TextLoader,
        show_progress=True
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128, chunk_overlap=0)
    all_splits = text_splitter.split_documents(docs)

    return all_splits


def read_vector_store(name):
    start_time = time.time()
    if not os.path.exists(f'cache/{name}'):
        vector_store = Chroma.from_documents(
            documents=_split_document(name),
            embedding=OpenAIEmbeddings(
                openai_api_base=config.OPEN_AI_BASE,
                openai_api_key=config.OPEN_AI_KEY
            ),
            persist_directory=f'cache/{name}'
        )
    else:
        vector_store = Chroma(
            persist_directory=f'cache/{name}',
            embedding_function=OpenAIEmbeddings(
                openai_api_base=config.OPEN_AI_BASE,
                openai_api_key=config.OPEN_AI_KEY
            )
        )
        print(f'find vector cache.')
    print(f'load finish within {time.time() - start_time} seconds')
    return vector_store


if __name__ == '__main__':
    vs = read_vector_store('01_severe_pneumonia')
    docs = vs.similarity_search("你之前有没有做过什么检查？")
    for doc in docs:
        print(doc)
