import os

file_path = os.path.join(os.getcwd(), "data\\corpus_POS.txt")
start_c = {}  # 开始概率，就是一个字典，state:chance=Word/lines
transport_c = {}  # 转移概率  num = num(state1)/num(statess)
emit_c = {}  # 观测概率   num = num(word)/num(words)
Count_dic = {}  # 一个属性下的所有单词，为了求解 emit
state_list = ['Ag', 'a', 'ad', 'an', 'Bg', 'b', 'c', 'Dg',
              'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l',
              'Mg', 'm', 'Ng', 'n', 'nr', 'ns', 'nt', 'nx',
              'nz', 'o', 'p', 'q', 'Rg', 'r', 's', 'na',
              'Tg', 't', 'u', 'Vg', 'v', 'vd', 'vn', 'vvn',
              'w', 'Yg', 'y', 'z']
lineCount = -1  # 句子总数，为了求出开始概率
for state0 in state_list:
    transport_c[state0] = {}
    for state1 in state_list:
        transport_c[state0][state1] = 0.0
    emit_c[state0] = {}
    start_c[state0] = 0.0
vocabs = []
classify = []
class_count = {}
for state in state_list:
    class_count[state] = 0.0

with open(file_path, encoding="gbk") as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        lineCount += 1  # 应该在有内容的行处加 1
        words = line.split(" ")  # 分解为多个单词
        for word in words:
            position = word.index('/')  # 如果是[中国人民/n]
            if '[' in word and ']' in word:
                vocabs.append(word[1:position])
                vocabs.append(word[position + 1:-1])
                break
            if '[' in word:
                vocabs.append(word[1:position])
                classify.append(word[position + 1:])
                break
            if ']' in word:
                vocabs.append(word[:position])
                classify.append(word[position + 1:-1])
                break
            vocabs.append(word[:position])
            classify.append(word[position + 1:])
        if len(vocabs) != len(classify):
            print('词汇数量与类别数量不一致')
            break  # 不一致退出程序
        else:
            for n in range(0, len(vocabs)):
                class_count[classify[n]] += 1.0
                if vocabs[n] in emit_c[classify[n]]:
                    emit_c[classify[n]][vocabs[n]] += 1.0
                else:
                    emit_c[classify[n]][vocabs[n]] = 1.0
                if n == 0:
                    start_c[classify[n]] += 1.0
                else:
                    transport_c[classify[n - 1]][classify[n]] += 1.0
        vocabs = []
        classify = []
for state in state_list:
    start_c[state] = start_c[state] * 1.0 / lineCount
    for li in emit_c[state]:
        emit_c[state][li] = emit_c[state][li] / class_count[state]
    for li in transport_c[state]:
        transport_c[state][li] = transport_c[state][li] / class_count[state]


def hmm_viterbi(obs, states, start_p, trans_p, emit_p):
    path = {}
    V = [{}]  # 记录第几次的概率
    for state in states:
        V[0][state] = start_p[state] * emit_p[state].get(obs[0], 0)
        path[state] = [state]
    for n in range(1, len(obs)):
        V.append({})
        newpath = {}
        for k in states:
            pp, pat = max([(V[n - 1][j] * trans_p[j].get(k, 0) * emit_p[k].get(obs[n], 0), j) for j in states])
            V[n][k] = pp
            newpath[k] = path[pat] + [k]
        path = newpath
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return prob, path[state]


test_strs = ["今天 天气 特别 好",
             "欢迎 大家 的 到来",
             "请 大家  喝茶",
             "你 的 名字 是 什么"
             ]
for li in range(0, len(test_strs)):
    test_strs[li] = test_strs[li].split()
for li in test_strs:
    p, out_list = hmm_viterbi(li, state_list, start_c, transport_c, emit_c)
    print(list(zip(li, out_list)))
