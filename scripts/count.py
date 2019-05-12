import json


path = 'data/redial/test_data.jsonl'
instances = []
with open(path) as json_file:
    for line in json_file.readlines():
        instances.append(json.loads(line))

cnt = 0
for instance in instances:
    initiator_id = instance["initiatorWorkerId"]
    respondent_id = instance["respondentWorkerId"]
    messages = instance["messages"]
    for message in messages:
        # if message['senderWorkerId'] == respondent_id:
        cnt += message['text'].count('@')
print(cnt)

