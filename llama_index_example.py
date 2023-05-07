import os
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex,LLMPredictor,ServiceContext
from langchain import OpenAI
from dotenv import load_dotenv

load_dotenv()

class LLama:

    def __init__(self) -> None:
        
        self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",max_tokens=1800000))
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)

    # 查询本地索引
    def query_index(self,prompt,index_path="./index.json"):

        # 加载索引
        local_index = GPTSimpleVectorIndex.load_from_disk(index_path)
        # 查询索引
        res = local_index.query(prompt,mode="embedding")
        resp = {"result":res}
        print(resp)


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
    # llama.create_index()

    # 查询索引
    llama.query_index("白鹿原的作者是谁")
    llama.query_index("白孝文的性格是怎样的")
    llama.query_index("田小娥怎样死的")
    llama.query_index("女性角色有多少名，他们的命运如何")
    llama.query_index("帮我生成一个快排")




