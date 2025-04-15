from query_processor import QueryProcessor
from asyncio import run
from langchain_openai import ChatOpenAI
import re

async def main():
    query_processor = QueryProcessor(subgraph_distance=2)
    # question = "What is HLC?"
    # answer = await query_processor.ask_question(question)
    # print(answer)
    prompt = open("final_prompt.txt", "r").read()
    response = query_processor.llm.invoke(prompt)
    final_answer = response.content
    final_answer = re.sub(
            r'(\d+\. \*\*[^:]+\*\*): ', 
            r'\n### \1\n', 
            final_answer
        )
    print(final_answer)
if __name__ == "__main__":
    run(main())