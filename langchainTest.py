from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os
from dotenv import load_dotenv

load_dotenv()

class LangChain:

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings()
 
    def create_index(self,dir_path="./data",persist_directory="db"):
        # 加载文件夹中的所有txt类型的文件
        loader = DirectoryLoader(dir_path, glob='**/*.txt')
        # 将数据转成 document 对象，每个文件会作为一个 document
        documents = loader.load()
        # 初始化加载器
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        # 切割加载的 document
        split_docs = text_splitter.split_documents(documents)
        # # 持久化数据
        embeddings = OpenAIEmbeddings()
 
        docsearch = Chroma.from_texts([t.page_content for t in split_docs], embedding= self.embeddings, persist_directory=persist_directory)
        docsearch.persist()
        print("保存完成")

    def query_index(self,prompt,persist_directory="db"):
        docsearch = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,return_source_documents=True)
        result = qa({"query": prompt})
        print(result["result"])
if __name__ == '__main__':
        langchain =  LangChain()
        # langchain.create_index()


        langchain.query_index(prompt="白鹿原的作者是谁？")
        langchain.query_index(prompt="田小饿被谁杀死了，怎么杀死的")
        langchain.query_index(prompt="白孝文的性格是怎样的")
        langchain.query_index(prompt="请用一句话描述白鹿原里的女性角色")
        langchain.query_index(prompt="白鹿原里的黑娃的形象是怎样的")
        langchain.query_index(prompt="白鹿原里的白嘉轩的腰被谁打折的")
        langchain.query_index(prompt="白嘉轩一共有几任老婆，分别是谁")
        langchain.query_index(prompt="请帮我实现一个快排算法")
        langchain.query_index(prompt="白嘉轩是谁？他的儿媳妇又是谁")



        
        
