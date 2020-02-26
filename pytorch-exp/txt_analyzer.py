import math

data = []
count = 0
max_count = 3
cur_data = [[], [], []]

# aggregates data together
with open('execution.txt') as in_file:
    line_count = 0
    for line in in_file:
        line = line.replace('\n', '')
        if line == '---':
            data.append(cur_data)
            cur_data = [[], [], []]
            count = 0
        else:
            cur_data[count].append(line)
            if count < (max_count - 1):
                count = count + 1
            else:
                count = 0
        line_count = line_count + 1

for d in data:
    cur_epoch = []
    count1 = 0
    count2 = 0
    count3 = 0

    correct1 = 0
    correct2 = 0
    correct3 = 0

    conf_right1 = 0
    conf_right2 = 0
    conf_right3 = 0

    conf_wrong1 = 0
    conf_wrong2 = 0
    conf_wrong3 = 0

    for i in range(len(d[0])):
        if d[0][i] == 'L1':
            count1 = count1 + 1
            if d[2][i] == '1':
                correct1 = correct1 + 1
                conf_right1 = conf_right1 + math.exp(float(d[1][i]))
            else:
                conf_wrong1 = conf_wrong1 + math.exp(float(d[1][i]))
        elif d[0][i] == 'L2':
            count2 = count2 + 1
            if d[2][i] == '1':
                correct2 = correct2 + 1
                conf_right2 = conf_right2 + math.exp(float(d[1][i]))
            else:
                conf_wrong2 = conf_wrong2 + math.exp(float(d[1][i]))
        elif d[0][i] == 'L3':
            count3 = count3 + 1
            if d[2][i] == '1':
                correct3 = correct3 + 1
                conf_right3 = conf_right3 + math.exp(float(d[1][i]))
            else:
                conf_wrong3 = conf_wrong3 + math.exp(float(d[1][i]))

    cur_epoch.append(count1)
    cur_epoch.append(count2)
    cur_epoch.append(count3)

    cur_epoch.append(-1)

    cur_epoch.append(correct1 / count1 if count1 != 0 else 0)
    cur_epoch.append(correct2 / count2 if count2 != 0 else 0)
    cur_epoch.append(correct3 / count3 if count3 != 0 else 0)

    cur_epoch.append(-1)

    cur_epoch.append(conf_right1 / correct1 if correct1 != 0 else 0)
    cur_epoch.append(conf_right2 / correct2 if correct2 != 0 else 0)
    cur_epoch.append(conf_right3 / correct3 if correct3 != 0 else 0)

    cur_epoch.append(-1)

    cur_epoch.append(conf_wrong1 / (count1 - correct1) if (count1 - correct1) != 0 else 0)
    cur_epoch.append(conf_wrong2 / (count2 - correct2) if (count2 - correct2) != 0 else 0)
    cur_epoch.append(conf_wrong3 / (count3 - correct3) if (count3 - correct3) != 0 else 0)

    print(cur_epoch)
