from query_processor import QueryProcessor
from asyncio import run
import json

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)
    
def write_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

async def main():
    query_processor = QueryProcessor(subgraph_distance=1, graph_path="./optimized_entity_graph.json")
    QA_dict = load_json("./QA.json")['qa_pairs']
    for QA in QA_dict:
        question = QA['question']
        answer = await query_processor.ask_question(question)
        if answer is None:
            continue
        print("Question:\n", question)
        print("Answer:\n", answer)
        print("\n-----------------------------------\n")
        answers_dict = load_json("./answers.json")
        answers_dict[question] = answer
        write_json("./answers.json", answers_dict)
    # answer = await query_processor.ask_question("What is EIM?")
    # print(answer)
    # answer = await query_processor.ask_question("What is the difference between V2G and XYZZSW?")
    # print(answer)
if __name__ == "__main__":
    run(main())