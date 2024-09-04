import json
import os


project = 'Chaos'

files = [dir[:-4] for dir in os.listdir("data\\"+project+"\\meshes")]
num = int(len(files)*0.7)

train_data = {'Chaos':{'normalized_samples':files[:num]}}
test_data = {'Chaos':{'normalized_samples':files[num:]}}

# json_train = json.dumps(train_data)
# json_test = json.dumps(test_data)

with open('splits/Chaos_train.json', 'w') as outfile:
    json.dump(train_data, outfile)

with open('splits/Chaos_test.json', 'w') as outfile:
    json.dump(test_data, outfile)

# with open("splits\\chaos_train.json", "r") as f:
#     train_split = json.load(f)
#     train_data = train_split["CHAOS"]["normalized_samples"]
#     train_list = []
#     for data in train_data:
#         train_list.append(data)
#         train_list.append(data+"_rotate1")
#
#     save_data = {'autoPET_512':{'normalized_samples':train_list}}
#     with open('splits/autoPET_aug_train.json', 'w') as outfile:
#         json.dump(save_data, outfile)
