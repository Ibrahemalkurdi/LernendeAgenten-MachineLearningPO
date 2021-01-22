import json
import sys
print(sys.getdefaultencoding())
with open('train-v2.0.json', 'r') as f:
    json_file = json.load(f)
question_context_list = []
answer_start_list_of_lists = []
answer_end_list_of_lists = []
for title_paragraphs in json_file['data']:
    for qas_context in title_paragraphs['paragraphs']:
        for qas in qas_context['qas']:
            question_context_list.append(qas['question'] + ":" + qas_context['context'])
            if len(qas['answers']) == 0:
                answer_start_list_of_lists.append([])
                answer_end_list_of_lists.append([])
            else:
                answers_start = []
                answers_end = []
                for answer in qas['answers']:
                    answers_start.append(answer['answer_start'])
                    answers_end.append(answer['answer_start'] + len(answer['text']) - 1)
                answer_start_list_of_lists.append(answers_start)
                answer_end_list_of_lists.append(answers_end)


#print(question_context_list)
#print("\n")
#print(answer_start_list_of_lists)
#print("\n")
#print(answer_end_list_of_lists)