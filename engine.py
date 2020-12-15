import bisect

input1 = "Hello, World!"
input2 = [(0, 13), (0, 12), (12, 13), (1, 12)]


def getLastIndex(list_of_elems, elem):
    return len(list_of_elems) - list_of_elems[::-1].index(elem) - 1


def parantez(mtn, pairs):
    # pairs = sorted(pairs, key=lambda x: x[0])
    print(pairs)
    sortedList = []
    current_string = list(mtn)
    for pair in range(len(pairs)):
        start = input2[pair][0]
        end = input2[pair][1]
        # sortedList.append(start)
        # sortedList.sort()
        bisect.insort(sortedList, start)
        # sortedList.append(end)
        # sortedList.sort()
        bisect.insort(sortedList, end)
        start = start + getLastIndex(sortedList, start)
        end = end + getLastIndex(sortedList, end)
        current_string.insert(start, "(")
        current_string.insert(end, ")")
    current_string = ''.join(current_string)
    print(str(current_string))


def parantez_v2(mtn, pairs):
    chars = []
    current_string = list(mtn)
    starts = list(map(lambda record: record[0], pairs))
    starts.sort()
    ends = list(map(lambda record: record[1], pairs))
    ends.sort()
    for char in range(len(current_string) + 1):
        end_num = ends.count(char)
        for i in range(end_num):
            chars.append(")")
        start_num = starts.count(char)
        for i in range(start_num):
            chars.append("(")
        if i < len(current_string):
            chars.append(current_string[char])
    current_string = ''.join(chars)
    print(str(current_string))






parantez_v2(input1, input2)