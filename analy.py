import json
happy = open('happy3.json', 'r')
con = open('contrast3.json', 'r')
pred = json.load(happy)
raw = json.load(con)
happy.close()
con.close()

for i in range(len(pred)):
    print(f'pred batch step:{len(pred[i])},raw batch step:{max(raw[i])}\n')