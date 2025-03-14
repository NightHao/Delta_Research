from query_processor import QueryProcessor
from asyncio import run

async def main():
    query_processor = QueryProcessor(subgraph_distance=1)
    question = "What is EIM?"
    # Async
    result, renewed_question = await query_processor.ask_question(question)
    print("Final result:", result)
    print("Renewed question:", renewed_question)

if __name__ == "__main__":
    run(main())