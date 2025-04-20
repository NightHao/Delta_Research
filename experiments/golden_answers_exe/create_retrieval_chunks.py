import pandas as pd
import json
# 讀取CSV
df = pd.read_csv('./thesis_experiment_result_update.csv')

# 初始化結果字典
qa_dict = {}

# 遍歷每一列資料
for _, row in df.iterrows():
    question_number = int(row['Question number'])  # 轉換為整數確保排序
    question = row['Question']
    
    # 分割 Table 和 Figure 成 list
    tables = row['Table'].split("\n") if pd.notna(row['Table']) else []
    figures = row['Figure'].split("\n") if pd.notna(row['Figure']) else []
    entity = row['Entity'].split("\n") if pd.notna(row['Entity']) else []
    
    # 過濾出 Answer 欄位
    answers = [row[col] for col in df.columns if 'Answer' in col and pd.notna(row[col])]

    # 建立結構
    qa_dict[question_number] = {
        "Question number": question_number,
        "Question": question,
        "Entity": entity,
        "Table": tables,
        "Figure": figures,
        "Answers": answers
    }

# 依照 Question number 進行排序
qa_dict_update = dict(sorted(qa_dict.items()))

# Write to a JSON file
with open('qa_data.json', 'w', encoding='utf-8') as f:
    json.dump(qa_dict_update, f, ensure_ascii=False, indent=4)

print(f"Data saved to qa_data.json")