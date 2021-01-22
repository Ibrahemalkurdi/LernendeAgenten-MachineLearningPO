import json
import sys

print(sys.getdefaultencoding())
with open('train-v2.0.json', 'r') as f:
    json_file = json.load(f)
pars_file = {'context': [], 'qas': []}
file = open("myfile.txt", "w", encoding="utf-8")
for title in json_file['data']:
    file.write('title: ' + title['title'] + '\n')
    for qas_cont in title['paragraphs']:
        pars_file['context'].append(qas_cont['context'])
        dict_que_ans = {'question': [], 'answers': []}
        file.write((5 * ' ') + 'context: ' + qas_cont['context'] + '\n')
        file.write((10 * ' ') + 'qas: \n')
        for que_ans in qas_cont['qas']:
            dict_que_ans['question'].append(que_ans['question'])
            file.write((15 * ' ') + 'question: ' + que_ans['question'] + '\n')
            file.write((15 * ' ') + 'answers: ')
            i = 0
            for ans in que_ans['answers']:
                dict_que_ans['answers'].append(ans['text'])
                file.write(i.__str__() + ' text: ' + ans['text'] + '\n')
        pars_file['qas'].append(dict_que_ans)
file.close()