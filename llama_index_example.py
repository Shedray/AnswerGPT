import os
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex,LLMPredictor,ServiceContext
from langchain import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLama:

    def __init__(self) -> None:
        
        self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",max_tokens=1800))
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)

    # 查询本地索引
    def query_index(self,prompt,index_path="./index.json"):

        # 加载索引
        local_index = GPTSimpleVectorIndex.load_from_disk(index_path)
        # 查询索引
        res = local_index.query(prompt)

        print(res)


    # 建立本地索引
    def create_index(self,dir_path="./data"):


        # 读取data文件夹下的文档
        documents = SimpleDirectoryReader(dir_path).load_data()

        index = GPTSimpleVectorIndex.from_documents(documents,service_context=self.service_context)

        print(documents)

        # 保存索引
        index.save_to_disk('./index.json')


if __name__ == '__main__':
    
    llama = LLama()

    # 建立索引
    llama.create_index()

    # 查询索引
    llama.query_index("列出所有 API")
