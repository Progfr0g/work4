import os
import re
from collections import Counter


# def text_analyze(array):
#     lines = [i.lower() for i in array]
#     code_list = sorted(codes())
#     lines = [i.replace(' ', '') for i in lines]
#     lines = [re.sub(r'\W', '', i) for i in lines]
#     r = re.compile(r'\w\w\w{2}(\d{3}|\d{2})')
#     newlist = list(filter(r.match, lines))
#     # print(newlist)
#     for num, element in enumerate(newlist):
#         element = list(element)
#         if len(element) > 5:
#             if element[0] == '0' or element[0] == 0:
#                 element[0] = 'o'
#             if element[1] == 'o':
#                 element[1] = '0'
#     #         for t in range(4, 6):
#     #             if element[t] == '0' or element[t] == 0:
#     #                 element[t] = 'o'
#     #         if len(element) > 8:
#     #             for t in range(6, 9):
#     #                 if element[t] == 'o':
#     #                     element[t] = '0'
#     #         else:
#     #             for t in range(6, 8):
#     #                 if element[t] == 'o':
#     #                     element[t] = '0'
#     #     pattern = re.compile(r'(\w\d{3}\w{2})')
#     #
#     #     element = pattern.split(''.join(element))
#     #     del element[0]
#     #     element[1] = re.sub(r'\D', '', element[1])
#     #     # if list(element[1])[0] == '0':
#     #     #     element[1] = element[1][1:]
#     #
#     #     newlist[num] = ''.join(element)
#
#     newlist = [i.upper() for i in lines]
#
#     newlist = [[x, newlist.count(x)] for x in set(newlist)]
#     rr = 0
#     if len(newlist) != 0:
#         mk = sorted(newlist, key=lambda x: x[1], reverse=True)[0]
#         rr = mk[0]
#     return rr
#
# def text_analyze2(file):
#     with open('./output.txt', 'r') as f:
#
#         lines = [i.strip().lower() for i in f.readlines()]
#         lines = [i.replace(' ', '') for i in lines]
#         lines = [re.sub(r'\W', '', i) for i in lines] (^\D\d{3})|(^\d{3})
#
#         r = re.compile('\w\d\d\d\w\w\d\d\d')
#         newlist = list(filter(r.match, lines)) (\d{3})*(\d{3})$
#
#         return newlist (\D\D)(\d{3}|\d{2})
def codes():
    f = open('/home/cucumber/somputer-vision/python/cars/codes.txt')
    lines = [i.strip().lower() for i in f.readlines()]
    lines = [re.findall(r'\d*', i) for i in lines]
    lines = lines[0]
    new_lines = []
    for i in lines:
        if i != '':
            if len(i) < 4:
                new_lines.append(int(i))

    return new_lines


codes_array = codes()


def analyze(number_plates):
    result_plate = {'first_ser': {}, 'reg_num': {}, 'second_ser': {}, 'region_code': {}}
    matches = {'first_ser': {}, 'reg_num': {}, 'second_ser': {}, 'region_code': {}}
    result = []
    plate = ''
    for element in number_plates:
        temp = re.findall("(^\D\d{3})|(^\d{3})", element)
        if temp is not None and len(temp) != 0:
            temp = list(temp[0])
            for elem in temp:
                if len(elem) > 3 and elem != '':
                    m1 = elem[:1]
                    m2 = elem[1:]
                    if m1 not in matches['first_ser']:
                        matches['first_ser'][m1] = 1
                    else:
                        matches['first_ser'][m1] += 1
                    if m2 not in matches['reg_num']:
                        matches['reg_num'][m2] = 1
                    else:
                        matches['reg_num'][m2] += 1
                elif len(elem) == 3 and elem != '':
                    m1 = elem
                    if m1 not in matches['reg_num']:
                        matches['reg_num'][m1] = 1
                    else:
                        matches['reg_num'][m1] += 1

        temp = re.findall("(\D\D\d{3})$", element)

        if temp is not None and len(temp) != 0:
            for elem in temp:
                if len(elem) > 4:
                    m1 = elem[:2]
                    m2 = elem[2:]
                    if m1 not in matches['second_ser']:
                        matches['second_ser'][m1] = 1
                    else:
                        matches['second_ser'][m1] += 1
                    if m2 not in matches['region_code']:
                        matches['region_code'][m2] = 1
                    else:
                        matches['region_code'][m2] += 1

    for i in matches:
        ll = sorted(matches[i].items(), key=lambda x: x[1], reverse=True)
        if len(ll) != 0:
            plate = plate + ll[0][0]
    result.append(plate)
    # plate = ''.join([sorted(matches[i].items(), key=lambda x: x[1], reverse=True)[0][0] for i in matches])

    return result
    # result[0] = ''.join(result[0])

# print(sorted(codes()))
