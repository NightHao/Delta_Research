from query_processor import QueryProcessor
from asyncio import run
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval import evaluate
from langchain_core.prompts import PromptTemplate
from deepeval.test_case import LLMTestCase
from langchain_openai import ChatOpenAI
import json, re, asyncio
from pathlib import Path

class AgenticFlowEval:
    def __init__(self):
        self.query_processor = QueryProcessor(subgraph_distance=1, graph_path="./processed_entity_graph.json")

    def load_json(self, path: str):
        with open(path, "r") as f:
            return json.load(f)
    
    def write_json(self, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def eval_correctness(self, question, actual_answer, expected_answer):
        correctness_metric = GEval(
            name="Correctness",
            #criteria="Determine whether the actual output is factually correct based on the expected output.",
            # NOTE: you can only provide either criteria or evaluation_steps, and not both
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            #criteria="Determine whether the actual output is factually correct and complete based on the expected output. Additional relevant detail beyond the expected output should be considered a positive attribute, not a negative one.",
            # evaluation_steps=[
            #     "Check whether the facts in 'actual output' contradict any facts in 'expected output'. Factual contradictions should be severely penalized.",
            #     "Additional detail in the actual output beyond what's in the expected output should be considered a positive attribute, not a negative one, as long as the additional information is accurate and relevant.",
            #     "Evaluate the core information from the expected output - is it all present in the actual output? Missing key information should be penalized.",
            #     "Consider whether the additional detail enhances understanding or provides useful context - this should be rewarded.",
            #     "Vague language or contradicting opinions are acceptable as long as they don't misrepresent factual information."
            # ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            #model=azure_openai,  # Replace with your actual model instance if different
            model="o4-mini",
            async_mode=False
        )
        test_case = LLMTestCase(
            input=question,
            actual_output=actual_answer,
            expected_output=expected_answer
        )

        # Evaluate the test case
        evaluation_result = evaluate([test_case], [correctness_metric])
        if evaluation_result and evaluation_result.test_results:
            test_result = evaluation_result.test_results[0]  # Get first test result
            if test_result.metrics_data:
                metric_data = test_result.metrics_data[0]  # Get first metric data
            else:
                print(f"No metrics data available for entry {question}")
        else:
            print(f"No evaluation results for entry {question}")
        #print(metric_data)
        result_entry = {
            "question": question,
            "score": metric_data.score,
            "reason": metric_data.reason
        }
        print(result_entry)
        # self.write_json("./tmp_res.json", result_entry)


    def evaluate_correctness_with_checkpoint(self, ground_truth_QA_path, generated_answers_path, eval_res_correctness_path):
        """
        Evaluates the correctness of LLM outputs against expected outputs with checkpointing.

        Parameters:
        - ground_truth_QA_path (str): Path to the ground truth QA JSON file.
        - generated_answers_path (str): Path to the generated answers JSON file.
        - eval_res_path (str): Path to the evaluation results JSON file.
        """
        ground_truth_QA = self.load_json(ground_truth_QA_path)
        generated_answers = self.load_json(generated_answers_path)
        if not ground_truth_QA or not generated_answers:
            print("No matched QA data to process.")
            return
        
        # Load existing evaluation results to avoid reprocessing
        eval_results = self.load_json(eval_res_correctness_path)
        processed_questions = set([result['question'] for result in eval_results])
        # Initialize the correctness metric
        correctness_metric = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            #NOTE: you can only provide either criteria or evaluation_steps, and not both
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            # criteria="Determine whether the actual output is factually correct and complete based on the expected output. Additional relevant detail beyond the expected output should be considered a positive attribute, not a negative one.",
            # evaluation_steps=[
            #     "Check whether the facts in 'actual output' contradict any facts in 'expected output'. Factual contradictions should be severely penalized.",
            #     "Additional detail in the actual output beyond what's in the expected output should be considered a positive attribute, not a negative one, as long as the additional information is accurate and relevant.",
            #     "Evaluate the core information from the expected output - is it all present in the actual output? Missing key information should be penalized.",
            #     "Consider whether the additional detail enhances understanding or provides useful context - this should be rewarded.",
            #     "Vague language or contradicting opinions are acceptable as long as they don't misrepresent factual information."
            # ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            #model=azure_openai,  # Replace with your actual model instance if different
            model="gpt-4.1",
            #model="o1",
            async_mode=False
        )
        
        total = len(generated_answers)
        print(f"Total entries to process: {total}")
        processed = len(processed_questions)
        print(f"Already processed entries: {processed}")
        res = []
        idx = 0
        try:
            for question, answer in generated_answers.items():
                print(question)
                expected_output = ground_truth_QA.get(question)
                idx += 1
                if not question:
                    print(f"Entry {idx} has no question. Skipping.")
                    continue
                if not expected_output:
                    print(f"Entry {idx} has no expected_output. Skipping.")
                    continue
                if not answer or answer.lower() == "null":
                    print(f"Entry {idx} has no valid answer. Skipping.")
                    continue
                if question in processed_questions:
                    print(f"Entry {idx}: '{question}' already processed. Skipping.")
                    continue
                
                # Create the test case
                test_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    expected_output=expected_output
                )

                # Evaluate the test case
                evaluation_result = evaluate([test_case], [correctness_metric])
                if evaluation_result and evaluation_result.test_results:
                    test_result = evaluation_result.test_results[0]  # Get first test result
                    if test_result.metrics_data:
                        metric_data = test_result.metrics_data[0]  # Get first metric data
                    else:
                        print(f"No metrics data available for entry {idx}")
                else:
                    print(f"No evaluation results for entry {idx}")
                    
                # evaluate([test_case], [correctness_metric])
                # result = evaluate([test_case], [correctness_metric])
                # metric_data = result.test_result[0].metrics_data[0]
                result_entry = {
                    "question": question,
                    "score": metric_data.score,
                    "reason": metric_data.reason
                }
                res.append(result_entry)
                # Save the result incrementally
                self.write_json(eval_res_correctness_path, res)
                
                # Update processed questions
                processed_questions.add(question)
                
                print(f"Entry {idx}/{total} evaluated and saved.")
        
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user. Progress saved to eval_res.json.")

        except Exception as e:
            print(f"\nAn error occurred during evaluation: {e}")
            print("Progress saved up to the last successfully processed entry.")

        finally:
            print(f"\nEvaluation completed. Results saved to {eval_res_correctness_path}.")

    async def generate_judgement(self, generated_answer_by_A, generated_answer_by_B, question):
        judgement_prompt = f"""
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user's instructions and answers the user's question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide an explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.

[User question]
{question}
[The Start of Assistant A's response]
{generated_answer_by_A}
[The End of Assistant A's response]
[The Start of Assistant B's response]
{generated_answer_by_B}
[The End of Assistant B's response]
"""
        llm = ChatOpenAI(model="o3-mini", reasoning_effort="low", seed = 42)
        prompt = PromptTemplate.from_template(judgement_prompt)
        chain = prompt | llm
        res = chain.invoke({"generated_answer_by_A": generated_answer_by_A, "generated_answer_by_B": generated_answer_by_B, "question": question})
        response_text = res.content
        verdict_match = re.search(r'\[\[([A-C])\]\]', response_text)
        verdict = verdict_match.group(0) if verdict_match else "No verdict found"
        explanation = response_text
        
        print(f"Question: {question}")
        print(f"Verdict: {verdict}")
        print("--------------------------------")
        return question, verdict, explanation
        
    async def judge_by_llm(self, generated_answer_by_A_path, generated_answer_by_B_path):
        generated_answer_by_A_dict = self.load_json(generated_answer_by_A_path)
        generated_answer_by_B_dict = self.load_json(generated_answer_by_B_path)
        A_name = f"{generated_answer_by_A_path.split('/')[-1].split('.')[0]}"
        B_name = f"{generated_answer_by_B_path.split('/')[-1].split('.')[0]}"
        judgements = {}
        tasks = []
        
        # Create all tasks first without awaiting them
        for question, answer_by_A in generated_answer_by_A_dict.items():
            if question in generated_answer_by_B_dict:
                task = self.generate_judgement(answer_by_A, generated_answer_by_B_dict[question], question)
                tasks.append(task)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        for question, verdict, explanation in results:
            if verdict == '[[A]]':
                verdict_name = A_name
            elif verdict == '[[B]]':
                verdict_name = B_name
            elif verdict == '[[C]]':
                verdict_name = 'Tie'
            else:
                verdict_name = 'No verdict'
            judgements[question] = {
                "verdict": verdict_name,
                "explanation": explanation
            }

        self.write_json("./llm_judgements.json", judgements)
        print("Judgements saved to ./llm_judgements.json")

    async def main(self):
        # Remember to clear entities chunks before running ag
        query_processor = QueryProcessor(subgraph_distance=1, graph_path="./optimized_entity_graph.json")
        # query_processor = QueryProcessor(subgraph_distance=1, graph_path="./processed_entity_graph.json")
        # query_processor = QueryProcessor(subgraph_distance=1, graph_path="./experiments/graphs/lightrag_graph.json")
        # query_processor = QueryProcessor(subgraph_distance=1, graph_path="./experiments/graphs/graphrag_graph.json")
        # query_processor = QueryProcessor(subgraph_distance=2, graph_path="./entity_graph.json")
        # ===Generate answers by not using the agentic flow===
        # QA_dict = self.load_json("./experiments/ground_truth_QA.json")
        # questions = ["VEHICLE-TO-GRID", "HOMEPLUG GREEN PHY", "SLAC", "LOGICAL NETWORK", 'CCO', ['DATA SAP', 'DATA LINK CONTROL SAP'], ["BASIC SIGNALING", "HIGH-LEVEL COMMUNICATION"], ["MAIN TEST COMPONENT", "PARALLEL TEST COMPONENT"], "ABSTRACT TEST SUITE", "TEST SUITE STRUCTURE", "EIM", "CM_SLAC_PARM.REQ", "CM_SLAC_PARAM.CNF", "CM_START_ATTEN_CHAR.IND", "CM_MNBC_SOUND.IND", "CM_ATTEN_CHAR.IND", "CM_ATTEN_CHAR.RSP", "CM_ATTEN_PROFILE.IND", "CM_VALIDATE.REQ", "CM_VALIDATE.CNF", "CM_SLAC_MATCH.REQ", "CM_SLAC_MATCH.CNF", "CM_SET_KEY.REQ", "CM_SET_KEY.CNF", "CM_AMP_MAP.REQ", "CM_AMP_MAP.CNF", "D-LINK_READY.indication", "D-LINK_TERMINATE.request", "D-LINK_ERROR.request", "D-LINK_PAUSE.request"]
        # complete_questions = ["What is the system architecture of Vehicle-to-Grid?", "What is HPGP?", "What is SLAC?", "What is Logical Network?", "What is CCo?", "What is the difference between Data SAP and Data link control SAP?", "What is the difference between basic signaling and high-level communication?", "What is the difference between MTC and PTC?", "What is ATS?", "What is TSS?", "What is EIM?", "What is CM_SLAC_PARAM.REQ?", "What is CM_SLAC_PARAM.CNF?", "What is CM_START_ATTEN_CHAR.IND?", "What is CM_MNBC_SOUND.IND?", "What is CM_ATTEN_CHAR.IND?", "What is CM_ATTEN_CHAR.RSP?", "What is CM_ATTEN_PROFILE.IND?", "What is CM_VALIDATE.REQ?", "What is CM_VALIDATE.CNF?", "What is CM_SLAC_MATCH.REQ?", "What is CM_SLAC_MATCH.CNF?", "What is CM_SET_KEY.REQ?", "What is CM_SET_KEY.CNF?", "What is CM_AMP_MAP.REQ?", "What is CM_AMP_MAP.CNF?", "What is D-LINK_READY.indication?", "What is D-LINK_TERMINATE.request?", "What is D-LINK_ERROR.request?", "What is D-LINK_PAUSE.request?"]
        # idx = 0
        # entity_graph = self.load_json("./entity_graph.json")
        # # prod_graph = self.load_json("./processed_entity_graph.json")
        # # entity_graph = self.load_json("./optimized_entity_graph.json")
        # # entity_graph = self.load_json("./experiments/graphs/lightrag_graph.json")
        # entity_answers = {}
        # # prod_answers = {}
        # # optimized_answers = {}
        # total_tokens = 0
        # for idx, question in enumerate(questions):
        #     entity_description = ""
        #     # prod_description = ""
        #     # optimized_description = ""
        #     if isinstance(question, list):
        #         for q in question:
        #             if q in entity_graph:
        #                 # entity_description += entity_graph[q]["description"]
        #                 subgraph = query_processor.flow_constructor.flow_operations.get_subgraph(entity_graph, q, 2)
        #                 # print(subgraph)
        #                 for node in subgraph:
        #                     entity_description += subgraph[node]["description"]
        #             # if q in prod_graph:
        #             #     prod_description += prod_graph[q]["description"]
        #             # if q in optimized_graph:
        #             #     optimized_description += optimized_graph[q]["description"]
        #     else:
        #         if question in entity_graph:
        #             subgraph = query_processor.flow_constructor.flow_operations.get_subgraph(entity_graph, question, 2)
        #             # print(subgraph)
        #             for node in subgraph:
        #                 entity_description += subgraph[node]["description"]
        #         # if questions[idx] in prod_graph:
        #         #     prod_description = prod_graph[questions[idx]]["description"]
        #         # if questions[idx] in optimized_graph:
        #         #     optimized_description = optimized_graph[questions[idx]]["description"]
        #     retrieved_chunks = self.load_json("./retrieved_chunks_15.json")
        #     print(retrieved_chunks[idx]['question'])
        #     entity_description += retrieved_chunks[idx]['prompt']
        #     if entity_description is None:
        #         continue
        #     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed = 42)
        #     prompt = PromptTemplate.from_template("""
        #     [Description]
        #     {description}
        #     [Instruction]
        #     You need to answer the following question as more details as possible based on the provided information above
        #     [Question]
        #     {question}
        #     """)
        #     chain = prompt | llm
        #     response = chain.invoke({"description": entity_description, "question": question})
        #     entity_answer = response.content
        #     total_tokens += response.usage_metadata["total_tokens"]
        #     # prod_answer = chain.invoke({"description": prod_description, "question": question}).content
        #     # optimized_answer = chain.invoke({"description": optimized_description, "question": question}).content
        #     # print(entity_answer)
        #     print(f"Finished processing question: {question}")
        #     entity_answers[complete_questions[idx]] = entity_answer
        #     # prod_answers[question] = prod_answer
        #     # optimized_answers[question] = optimized_answer
        # self.write_json("./experiments/Answers/refrag_twohop_ans.json", entity_answers)
        # print(f"Total tokens: {total_tokens}")
        # self.write_json("./prod_answers.json", prod_answers)
        # self.write_json("./optimized_answers.json", optimized_answers)
        # ===Generate answers by executing the agentic flow===
        # QA_dict = self.load_json("./experiments/ground_truth_QA.json")
        # # QA_dict = {"What is the commonality between Data SAP and Data link control SAP?": "No answer",
        # #            "What is the commonality between basic signaling and high-level communication?": "No answer",
        # #            "What is the commonality between MTC and PTC?": "No answer"}
        # for question, _ in QA_dict.items():
        #     #check if the question is already in the entities_chunks.json
        #     query_processor.flow_constructor.flow_operations.init_replaced_term()
        #     answer = await query_processor.ask_question(question)
        #     if answer is None:
        #         continue
        #     # print("Question:\n", question)
        #     # print("Answer:\n", answer)
        #     # print("\n-----------------------------------\n")
        #     print(f"Finished processing question: {question}")
        #     # answers_dict = self.load_json("./optimized_answers.json")
        #     # answers_dict = self.load_json("./prod_answers.json")
        #     # answers_dict = self.load_json("./experiments/Answers/lightrag_ans_w_ag.json")
        #     # answers_dict = self.load_json("./experiments/Answers/graphrag_ans_w_ag.json")
        #     answers_dict = self.load_json("./experiments/Answers/4omini_ourgraph_ag.json")
        #     # answers_dict = self.load_json("./experiments/Answers/o4mini_golden_ans.json")
        #     # answers_dict = self.load_json("./experiments/subgraph_extraction_ans/com/twohop_com.json")
        #     answers_dict[question] = answer
        #     # self.write_json("./optimized_answers.json", answers_dict)
        #     # self.write_json("./prod_answers.json", answers_dict)
        #     # self.write_json("./experiments/Answers/lightrag_ans_w_ag.json", answers_dict)
        #     # self.write_json("./experiments/Answers/graphrag_ans_w_ag.json", answers_dict)
        #     self.write_json("./experiments/Answers/4omini_ourgraph_ag.json", answers_dict)
        #     # self.write_json("./experiments/Answers/o4mini_golden_ans.json", answers_dict)
        #     # self.write_json("./experiments/subgraph_extraction_ans/com/twohop_com.json", answers_dict)

        # answer = await query_processor.ask_question("What is EIM?")
        # print(answer)
        # answer = await query_processor.ask_question("What is the difference between V2G and XYZZSW?")
        # print(answer)
        # answer = await query_processor.ask_question("What is the difference between EIM and MTC?")
        # print(answer)

        # ===Generate answers by executing the prompts===
        # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, seed = 42)
        # llm = ChatOpenAI(model="o4-mini", seed = 42)
        # merged_prompt_answers = {}
        # # Load existing answers if they exist
        # # target_file = "./experiments/Answers/4omini_all_info_ans.json"
        # target_file = "./experiments/Answers/o4mini_all_info_ans.json"
        # try:
        #     existing_answers = self.load_json(target_file)
        #     merged_prompt_answers = existing_answers
        #     print(f"Loaded {len(merged_prompt_answers)} existing answers")
        # except:
        #     print(f"No existing answers file found, creating new one")
            
        # # merged_prompts_dir = ("./merged_prompts")
        # # merged_prompts_dir = ("/home/yeskey525/Research_CODE/experiments/golden_answers_exe/complete_prompts")
        # # merged_prompts_dir = ("/home/yeskey525/Research_CODE/merged_prompts")
        # merged_prompts_dir = ("/home/yeskey525/Research_CODE/new_prompts")
        # merged_prompts_dir = Path(merged_prompts_dir)
        # if not merged_prompts_dir.exists():
        #     print(f"Directory {merged_prompts_dir} does not exist.")
        # else:
        #     for prompt_file in merged_prompts_dir.glob("*.txt"):
        #         question = prompt_file.stem
        #         print(question)
        #         # Skip if already processed
        #         if question in merged_prompt_answers:
        #             print(f"Skipping already processed question: {question}")
        #             continue
                    
        #         print(f"Processing question: {question}")
        #         try:
        #             with open(prompt_file, 'r', encoding='utf-8') as f:
        #                 prompt_content = f.read()
        #             prompt = PromptTemplate.from_template("""{prompt_content}""")
        #             chain = prompt | llm
        #             answer = chain.invoke({"prompt_content": prompt_content}).content
        #             merged_prompt_answers[question] = answer
        #             print(f"Finished processing question: {question}")
        #             # Save after each successful processing to avoid losing progress
        #             self.write_json(target_file, merged_prompt_answers)
        #         except Exception as e:
        #             print(f"Error processing {prompt_file}: {e}")

        #     self.write_json(target_file, merged_prompt_answers)
        #     print(f"Saved {len(merged_prompt_answers)} answers to o4mini_all_info_ans.json")

        # ===Evaluate answers===
        # o4mini_golden_ans
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/graphrag_ans_wo_ag.json", "./experiments/correctness_eval_ag/graphrag_ans_wo_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/graphrag_ans_w_ag.json", "./experiments/correctness_eval_ag/graphrag_ans_w_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/lightrag_naive.json", "./experiments/correctness_eval_ag/lightrag_naive_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/lightrag_local.json", "./experiments/correctness_eval_ag/lightrag_local_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/lightrag_global.json", "./experiments/correctness_eval_ag/lightrag_global_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/lightrag_hybrid.json", "./experiments/correctness_eval_ag/lightrag_hybrid_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/lightrag_mix.json", "./experiments/correctness_eval_ag/lightrag_mix_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/lightrag_ans_w_ag.json", "./experiments/correctness_eval_ag/lightrag_ans_w_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/4omini_ourgraph_wo_ag.json", "./experiments/correctness_eval_ag/4omini_ans_wo_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/4omini_ourgraph_ag.json", "./experiments/correctness_eval_ag/4omini_ans_w_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/4omini_ourgraph_ag_backup.json", "./experiments/correctness_eval_ag/4omini_ans_w_ag_correctness_backup.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/4o_ourgraph_ag_answers.json", "./experiments/correctness_eval_ag/4o_ans_w_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./o1_answers.json", "./experiments/Answers/4omini_ourgraph_ag.json", "./experiments/correctness_eval_ag/4omini_ans_w_ag_correctness_o1.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/refrag_onehop_ans.json", "./experiments/correctness_eval_ag/refrag_onehop_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/refrag_twohop_ans.json", "./experiments/correctness_eval_ag/refrag_twohop_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/refrag_onehop_ans_pro.json", "./experiments/correctness_eval_ag/refrag_onehop_pro_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/Answers/refrag_twohop_ans_pro.json", "./experiments/correctness_eval_ag/refrag_twohop_pro_correctness.json")
        # Complete golden answers
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/graphrag_ans_wo_ag.json", "./experiments/correctness_eval_complete_ans/graphrag_ans_wo_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/graphrag_ans_w_ag.json", "./experiments/correctness_eval_complete_ans/graphrag_ans_w_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/lightrag_naive.json", "./experiments/correctness_eval_complete_ans/lightrag_naive_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/lightrag_local.json", "./experiments/correctness_eval_complete_ans/lightrag_local_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/lightrag_global.json", "./experiments/correctness_eval_complete_ans/lightrag_global_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/lightrag_hybrid.json", "./experiments/correctness_eval_complete_ans/lightrag_hybrid_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/lightrag_mix.json", "./experiments/correctness_eval_complete_ans/lightrag_mix_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/lightrag_ans_w_ag.json", "./experiments/correctness_eval_complete_ans/lightrag_ans_w_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/4omini_ourgraph_wo_ag.json", "./experiments/correctness_eval_complete_ans/4omini_ans_wo_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/4omini_ourgraph_ag.json", "./experiments/correctness_eval_complete_ans/4omini_ans_w_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/4o_ourgraph_ag_answers.json", "./experiments/correctness_eval_complete_ans/4o_ans_w_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/4omini_ourgraph_ag.json", "./experiments/correctness_eval_complete_ans/4omini_ans_w_ag_correctness_o1.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/o4mini_golden_ans.json", "./experiments/correctness_eval_complete_ans/golden_ans_ag_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/4omini_all_info_ans.json", "./experiments/correctness_eval_complete_ans/4omini_all_info_ans_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/Answers/o4mini_all_info_ans.json", "./experiments/correctness_eval_complete_ans/o4mini_all_info_ans_correctness.json")

        # Subgraph extraction answers
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/complete_golden_answers.json", "./experiments/subgraph_extraction_ans/full_two_hop_ans.json", "./experiments/subgraph_extraction_ans/full_two_hop_ans_correctness_op.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/subgraph_extraction_ans/2u2c_cmp.json", "./experiments/subgraph_extraction_ans/2u2c_cmp_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/subgraph_extraction_ans/2u1c_cmp.json", "./experiments/subgraph_extraction_ans/2u1c_cmp_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/subgraph_extraction_ans/1u1c_cmp.json", "./experiments/subgraph_extraction_ans/1u1c_cmp_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/subgraph_extraction_ans/1u_cmp.json", "./experiments/subgraph_extraction_ans/1u_cmp_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/Answers/o4mini_golden_ans.json", "./experiments/subgraph_extraction_ans/anchor.json", "./experiments/subgraph_extraction_ans/anchor_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/subgraph_extraction_ans/com/1u2c_o4mini_com.json", "./experiments/subgraph_extraction_ans/com/2u2c_com.json", "./experiments/subgraph_extraction_ans/com/2u2c_com_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/subgraph_extraction_ans/com/1u2c_o4mini_com.json", "./experiments/subgraph_extraction_ans/com/1u2c_com.json", "./experiments/subgraph_extraction_ans/com/1u2c_com_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/subgraph_extraction_ans/com/1u2c_o4mini_com.json", "./experiments/subgraph_extraction_ans/com/1u1c_com.json", "./experiments/subgraph_extraction_ans/com/1u1c_com_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/subgraph_extraction_ans/com/1u2c_o4mini_com.json", "./experiments/subgraph_extraction_ans/com/1c_com.json", "./experiments/subgraph_extraction_ans/com/1c_com_correctness.json")
        # self.evaluate_correctness_with_checkpoint("./experiments/subgraph_extraction_ans/com/1u2c_o4mini_com.json", "./experiments/subgraph_extraction_ans/com/twohop_com.json", "./experiments/subgraph_extraction_ans/com/twohop_com_correctness.json")
        # cmp_full = self.load_json("./experiments/subgraph_extraction_ans/full_two_hop_ans_correctness_op.json")
        # cmp_2u2c = self.load_json("./experiments/subgraph_extraction_ans/2u2c_cmp_correctness.json")
        # cmp_2u1c = self.load_json("./experiments/subgraph_extraction_ans/2u1c_cmp_correctness.json")
        # cmp_1u1c = self.load_json("./experiments/subgraph_extraction_ans/1u1c_cmp_correctness.json")
        # cmp_1u = self.load_json("./experiments/subgraph_extraction_ans/1u_cmp_correctness.json")
        # cmp_anchor = self.load_json("./experiments/subgraph_extraction_ans/anchor_correctness.json")
        # print(f"full_two_hop_ans_correctness: {sum([eval_res['score'] for eval_res in cmp_full]) / len(cmp_full)}")
        # com_2u2c = self.load_json("./experiments/subgraph_extraction_ans/com/2u2c_com_correctness.json")
        # com_1u2c = self.load_json("./experiments/subgraph_extraction_ans/com/1u2c_com_correctness.json")
        # com_1u1c = self.load_json("./experiments/subgraph_extraction_ans/com/1u1c_com_correctness.json")
        # com_1c = self.load_json("./experiments/subgraph_extraction_ans/com/1c_com_correctness.json")
        # twohop_com = self.load_json("./experiments/subgraph_extraction_ans/com/twohop_com_correctness.json")
        # print(f"2u2c_cmp_correctness: {sum([eval_res['score'] for eval_res in cmp_2u2c]) / len(cmp_2u2c)}")
        # print(f"2u1c_cmp_correctness: {sum([eval_res['score'] for eval_res in cmp_2u1c]) / len(cmp_2u1c)}")
        # print(f"1u1c_cmp_correctness: {sum([eval_res['score'] for eval_res in cmp_1u1c]) / len(cmp_1u1c)}")
        # print(f"1u_cmp_correctness: {sum([eval_res['score'] for eval_res in cmp_1u]) / len(cmp_1u)}")
        # print(f"anchor_correctness: {sum([eval_res['score'] for eval_res in cmp_anchor]) / len(cmp_anchor)}")
        # print(f"2u2c_com_correctness: {sum([eval_res['score'] for eval_res in com_2u2c]) / len(com_2u2c)}")
        # print(f"1u2c_com_correctness: {sum([eval_res['score'] for eval_res in com_1u2c]) / len(com_1u2c)}")
        # print(f"1u1c_com_correctness: {sum([eval_res['score'] for eval_res in com_1u1c]) / len(com_1u1c)}")
        # print(f"1c_com_correctness: {sum([eval_res['score'] for eval_res in com_1c]) / len(com_1c)}")
        # print(f"twohop_com_correctness: {sum([eval_res['score'] for eval_res in twohop_com]) / len(twohop_com)}")
        # await self.judge_by_llm("./prod_answers.json", "./optimized_answers.json")
        # await self.judge_by_llm("./prod_answers.json", "./optimized_answers.json")
        # judgements = self.load_json("./llm_judgements.json")
        # verdicts_count = {}
        # for question, judgement in judgements.items():
        #     verdict = judgement.get("verdict", "No verdict")
        #     verdicts_count[verdict] = verdicts_count.get(verdict, 0) + 1
        # print(verdicts_count)

        # prod_correctness = self.load_json("./prod_eval_res_correctness.json")
        # optimized_correctness = self.load_json("./optimized_eval_res_correctness.json")
        # prod_dict = {eval_res.get("question", "No question"): eval_res.get("score", 0) for eval_res in prod_correctness}
        # optimized_dict = {eval_res.get("question", "No question"): eval_res.get("score", 0) for eval_res in optimized_correctness}
        # better_count = {"prod": 0, "optimized": 0}
        # for question, score in prod_dict.items():
        #     if question in optimized_dict:
        #         if score > optimized_dict[question]:
        #             better_count["prod"] += 1
        #         else:
        #             better_count["optimized"] += 1
        #     else:
        #         print(f"Question {question} not in optimized_dict")
        # print(better_count)

        # graphrag_ans_wo_ag_correctness = self.load_json("./experiments/correctness_eval_ag/graphrag_ans_wo_ag_correctness.json")
        # graphrag_ans_w_ag_correctness = self.load_json("./experiments/correctness_eval_ag/graphrag_ans_w_ag_correctness.json")
        # lightrag_naive_correctness = self.load_json("./experiments/correctness_eval_ag/lightrag_naive_correctness.json")
        # lightrag_local_correctness = self.load_json("./experiments/correctness_eval_ag/lightrag_local_correctness.json")
        # lightrag_global_correctness = self.load_json("./experiments/correctness_eval_ag/lightrag_global_correctness.json")
        # lightrag_hybrid_correctness = self.load_json("./experiments/correctness_eval_ag/lightrag_hybrid_correctness.json")
        # lightrag_mix_correctness = self.load_json("./experiments/correctness_eval_ag/lightrag_mix_correctness.json")
        # lightrag_ans_w_ag_correctness = self.load_json("./experiments/correctness_eval_ag/lightrag_ans_w_ag_correctness.json")
        # gpt4omini_ans_wo_ag_correctness = self.load_json("./experiments/correctness_eval_complete_ans/4omini_ans_wo_ag_correctness.json")
        # gpt4omini_ans_w_ag_correctness = self.load_json("./experiments/correctness_eval_ag/4omini_ans_w_ag_correctness.json")
        # gpt4omini_ans_w_ag_correctness_backup = self.load_json("./experiments/correctness_eval_ag/4omini_ans_w_ag_correctness_backup.json")
        # gpt4o_ans_w_ag_correctness = self.load_json("./experiments/correctness_eval_ag/4o_ans_w_ag_correctness.json")
        # golden_ans_ag_correctness = self.load_json("./experiments/correctness_eval_complete_ans/golden_ans_ag_correctness.json")
        # gpt4omini_ans_w_ag_correctness_o1 = self.load_json("./experiments/correctness_eval_ag/4omini_ans_w_ag_correctness_o1.json")
        # gpt4omini_all_info_ans_correctness = self.load_json("./experiments/correctness_eval_complete_ans/4omini_all_info_ans_correctness_v2.json")
        # o4mini_all_info_ans_correctness = self.load_json("./experiments/correctness_eval_complete_ans/o4mini_all_info_ans_correctness_backup.json")
        refrag_onehop_correctness = self.load_json("./experiments/correctness_eval_ag/refrag_onehop_correctness.json")
        refrag_twohop_correctness = self.load_json("./experiments/correctness_eval_ag/refrag_twohop_correctness.json")
        refrag_onehop_correctness_pro = self.load_json("./experiments/correctness_eval_ag/refrag_onehop_pro_correctness.json")
        refrag_twohop_correctness_pro = self.load_json("./experiments/correctness_eval_ag/refrag_twohop_pro_correctness.json")
        
        # # print the average score of the correctness
        # print(f"graphrag_ans_wo_ag_correctness: {sum([eval_res['score'] for eval_res in graphrag_ans_wo_ag_correctness]) / len(graphrag_ans_wo_ag_correctness)}")
        # print(f"graphrag_ans_w_ag_correctness: {sum([eval_res['score'] for eval_res in graphrag_ans_w_ag_correctness]) / len(graphrag_ans_w_ag_correctness)}")
        # print(f"lightrag_naive_correctness: {sum([eval_res['score'] for eval_res in lightrag_naive_correctness]) / len(lightrag_naive_correctness)}")
        # print(f"lightrag_local_correctness: {sum([eval_res['score'] for eval_res in lightrag_local_correctness]) / len(lightrag_local_correctness)}")
        # print(f"lightrag_global_correctness: {sum([eval_res['score'] for eval_res in lightrag_global_correctness]) / len(lightrag_global_correctness)}")
        # print(f"lightrag_hybrid_correctness: {sum([eval_res['score'] for eval_res in lightrag_hybrid_correctness]) / len(lightrag_hybrid_correctness)}")
        # print(f"lightrag_mix_correctness: {sum([eval_res['score'] for eval_res in lightrag_mix_correctness]) / len(lightrag_mix_correctness)}")
        # print(f"lightrag_ans_w_ag_correctness: {sum([eval_res['score'] for eval_res in lightrag_ans_w_ag_correctness]) / len(lightrag_ans_w_ag_correctness)}")
        # print(f"gpt4omini_ans_wo_ag_correctness: {sum([eval_res['score'] for eval_res in gpt4omini_ans_wo_ag_correctness]) / len(gpt4omini_ans_wo_ag_correctness)}")
        # print(f"gpt4omini_ans_w_ag_correctness: {sum([eval_res['score'] for eval_res in gpt4omini_ans_w_ag_correctness]) / len(gpt4omini_ans_w_ag_correctness)}")
        # print(f"gpt4omini_ans_w_ag_correctness_backup: {sum([eval_res['score'] for eval_res in gpt4omini_ans_w_ag_correctness_backup]) / len(gpt4omini_ans_w_ag_correctness_backup)}")
        # print(f"gpt4o_ans_w_ag_correctness: {sum([eval_res['score'] for eval_res in gpt4o_ans_w_ag_correctness]) / len(gpt4o_ans_w_ag_correctness)}")
        # print(f"golden_ans_ag_correctness: {sum([eval_res['score'] for eval_res in golden_ans_ag_correctness]) / len(golden_ans_ag_correctness)}")
        # print(f"gpt4omini_ans_w_ag_correctness_o1: {sum([eval_res['score'] for eval_res in gpt4omini_ans_w_ag_correctness_o1]) / len(gpt4omini_ans_w_ag_correctness_o1)}")
        # print(f"4omini_all_info_ans_correctness: {sum([eval_res['score'] for eval_res in gpt4omini_all_info_ans_correctness]) / len(gpt4omini_all_info_ans_correctness)}")
        # print(f"o4mini_all_info_ans_correctness: {sum([eval_res['score'] for eval_res in o4mini_all_info_ans_correctness]) / len(o4mini_all_info_ans_correctness)}")
        print(f"refrag_onehop_correctness: {sum([eval_res['score'] for eval_res in refrag_onehop_correctness]) / len(refrag_onehop_correctness)}")
        print(f"refrag_twohop_correctness: {sum([eval_res['score'] for eval_res in refrag_twohop_correctness]) / len(refrag_twohop_correctness)}")
        print(f"refrag_onehop_correctness_pro: {sum([eval_res['score'] for eval_res in refrag_onehop_correctness_pro]) / len(refrag_onehop_correctness_pro)}")
        print(f"refrag_twohop_correctness_pro: {sum([eval_res['score'] for eval_res in refrag_twohop_correctness_pro]) / len(refrag_twohop_correctness_pro)}")
        # # Compare graphrag_4o and graphrag for questions with scores >= 0.3 in graphrag_4o
        # print("\nComparison between graphrag_4o and graphrag (scores >= 0.3 in graphrag_4o):")
        
        # # Create dictionaries to map questions to scores for easier comparison
        # graphrag_4o_dict = {eval_res['question']: eval_res['score'] for eval_res in graphrag_4o_correctness}
        # graphrag_dict = {eval_res['question']: eval_res['score'] for eval_res in graphrag_correctness}
        
        # # Filter questions with scores >= 0.3 in graphrag_4o
        # filtered_questions = [q for q, score in graphrag_4o_dict.items() if score >= 0.3 and q in graphrag_dict]
        
        # if filtered_questions:
        #     # Calculate average scores for the filtered questions
        #     graphrag_4o_avg = sum(graphrag_4o_dict[q] for q in filtered_questions) / len(filtered_questions)
        #     graphrag_avg = sum(graphrag_dict[q] for q in filtered_questions) / len(filtered_questions)
            
        #     print(f"Number of questions with score >= 0.3 in graphrag_4o: {len(filtered_questions)}")
        #     print(f"Average score for graphrag_4o (filtered): {graphrag_4o_avg:.4f}")
        #     print(f"Average score for graphrag (same questions): {graphrag_avg:.4f}")
            
            # Count how many times each model performs better
            # graphrag_4o_better = sum(1 for q in filtered_questions if graphrag_4o_dict[q] > graphrag_dict[q])
            # graphrag_better = sum(1 for q in filtered_questions if graphrag_dict[q] > graphrag_4o_dict[q])
            # tie = sum(1 for q in filtered_questions if graphrag_4o_dict[q] == graphrag_dict[q])
            
            # print(f"graphrag_4o better: {graphrag_4o_better} questions")
            # print(f"graphrag better: {graphrag_better} questions")
            # print(f"Tie: {tie} questions")
            
            # # Optional: Show score differences for specific questions
            # print("\nIndividual question comparison (score differences):")
            # for q in filtered_questions:
            #     diff = graphrag_4o_dict[q] - graphrag_dict[q]
            #     if abs(diff) >= 0.2:  # Only show significant differences
            #         print(f"Question: {q}")
            #         print(f"  graphrag_4o score: {graphrag_4o_dict[q]:.2f}")
            #         print(f"  graphrag score: {graphrag_dict[q]:.2f}")
            #         print(f"  Difference: {diff:.2f}")
        # else:
        #     print("No questions found with scores >= 0.3 in graphrag_4o data.")

        # ===Generate judgement===
        # prod_answers = self.load_json("./experiments/prod_answers_ag.json")
        # optimized_answers = self.load_json("./optimized_answers.json")
        # judgement_prompts = {}
        # for question, answer in prod_answers.items():
        #     if question in optimized_answers:
        #         judgement_prompt = self.generated_judgement_prompt(answer, optimized_answers[question], question)
        #         judgement_prompts[question] = judgement_prompt
        # self.write_json("./judgement_prompts.json", judgement_prompts)
#         self.eval_correctness(
#             question="What is HLC?",
#             actual_answer="""High-Level Communication (HLC) is a sophisticated communication technology utilized primarily in the context of electric vehicles (EVs) and their interaction with charging infrastructure. It serves as a critical component in the communication systems that facilitate advanced interactions between electric vehicles and Electric Vehicle Supply Equipment (EVSE). Here are the key aspects of High-Level Communication:


# ### 1. **Definition and Purpose**
# High-Level Communication refers to a set of protocols and technologies that enable complex data exchanges and command interactions between electric vehicles and charging stations. It enhances the communication capabilities beyond basic signaling, allowing for more advanced features such as bidirectional energy flow, real-time data exchange, and automated processes.


# ### 2. **Components**
# HLC encompasses various protocols, including the HLC-C protocol, which is specifically designed for high-level communication during the charging process. This protocol is essential for managing the charging operations, ensuring that the EV and EVSE can communicate effectively regarding charging status, current limits, and other operational parameters.


# ### 3. **Relation to Basic Signaling**
# HLC builds upon basic signaling, which provides the foundational signals necessary for establishing a communication link. While basic signaling is crucial for initial connections and simple commands, HLC allows for more intricate interactions, such as negotiating charging rates, managing energy distribution, and facilitating vehicle-to-grid (V2G) communication.


# ### 4. **Standards and Requirements**
# The implementation of HLC is guided by various standards, including the V2G3 requirements, which mandate the use of HLC to enhance communication capabilities. For instance, the V2G3-M06-02 requirement specifies the necessity of HLC for enabling bidirectional communication and offering additional features in the charging process.


# ### 5. **Impact on Charging Processes**
# The presence of HLC in electric vehicles significantly affects the control pilot duty cycle values, which determine the maximum charge current that can be delivered to the vehicle. This capability allows for more efficient and flexible charging solutions, adapting to the needs of both the vehicle and the grid.


# ### 6. **Fallback Mechanisms**
# In scenarios where HLC fails, basic charging (BC) serves as a backup, indicating a fallback relationship between the two systems. This ensures that even if advanced communication fails, the vehicle can still charge using standard methods.


# ### 7. **Integration with Other Technologies**
# HLC is often integrated with other technologies such as Electricity Information Management (EIM) and Plug and Charge (PnC) systems, which further streamline the charging process by automating authentication and data exchange.


# ### 8. **Vehicle-to-Grid (V2G) Communication**
# HLC plays a vital role in V2G setups, where electric vehicles not only draw power from the grid but can also return energy. This bidirectional communication is essential for managing energy resources effectively and supporting grid stability.

# In summary, High-Level Communication is a crucial technology that enhances the interaction between electric vehicles and charging infrastructure, enabling advanced features, efficient energy management, and compliance with evolving standards in electric vehicle technology.""",
#             expected_answer="""High-Level Communication (often abbreviated as **HLC**) is an **advanced communication layer or protocol** used during the electric vehicle (EV) charging process **beyond** the basic signaling defined by standards like IEC 61851‑1. Whereas basic signaling manages fundamental on/off and current-limit signals via the control pilot line, HLC adds **more sophisticated, bidirectional data exchange** between the vehicle and the charging infrastructure. This allows for richer functionality, such as:

# 1. **Advanced Control of Charging**  
#    - Through HLC, the charging station (**EVSE**) and electric vehicle (**EV**) can exchange detailed information about **maximum current, charging schedules, and session states** (e.g., entering sleep mode, resuming from pause, stopping the session).
#    - The EVSE may apply the **HLC-C protocol** (High-Level Communication Control) to negotiate or dynamically manage charging parameters.  

# 2. **Integration with Authorization and Identification**  
#    - HLC allows for advanced features such as **Plug and Charge (PnC)** or **External Identification Means (EIM)**. These methods provide **automatic authentication and authorization** of charging sessions without requiring manual card swipes or RFID scans.  
#    - By exchanging identification data (e.g., contract certificates, keys), the vehicle and charger can **establish a secure session** seamlessly.

# 3. **Fallback to Basic Charging**  
#    - If HLC fails or is unavailable, many systems are designed to **fall back to "basic charging"** (BC) as defined by IEC 61851‑1. In that scenario, the control pilot's duty cycle alone dictates allowable current, but it lacks advanced communication or authentication features.

# 4. **Relationship to Basic Signaling**  
#    - Basic signaling is the **foundation** for ensuring safe current flow and certain minimal handshake steps.  
#    - HLC then **builds on top** of that basic layer, giving the EV and EVSE a broader data channel to exchange more granular charging instructions, statuses, and advanced control messages.

# 5. **Bidirectional / Vehicle-to-Grid (V2G) Communication**  
#    - High-Level Communication is also the **mechanism** by which V2G features become possible. V2G requires **coordinated data exchange** (e.g., how much power the grid needs or can absorb, pricing signals, vehicle commands) – all of which rely on an advanced protocol rather than only the pilot signal.

# 6. **Examples of HLC Use Cases**  
#    - **Controlling the actual charging loop**: An EV might start charging only after concluding certain negotiations with the EVSE using HLC-C.  
#    - **PNC / EIM**: The EV automatically identifies itself to the station, obtains authorization, and sets up a secure session.  
#    - **Error Handling**: If a data link needs to be paused or restarted, HLC messages coordinate how the EV and EVSE proceed.

# In short, **High-Level Communication** is the **enhanced, protocol-driven data exchange** layer between an EV and a charging station that goes far beyond the low-level pilot signal, enabling features like **automatic authentication**, **dynamic charge management**, **advanced security**, and (in many cases) the foundation for **vehicle-to-grid** (V2G) interactions."""
#         )

if __name__ == "__main__":
    agentic_flow_eval = AgenticFlowEval()
    run(agentic_flow_eval.main())