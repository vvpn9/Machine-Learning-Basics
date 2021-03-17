import os


def fileWalker(path):
    fileArray = []
    for roots, dirs, files in os.walk(path):
        for fn in files:
            eachpath = str(roots + '/' + fn)
            fileArray.append(eachpath)
    # print(fileArray)
    return fileArray


# fileWalker('/Users/zw/Desktop/Data Learning/Machine Learning Basics/Spam Filter')

def readText(path, encoding):
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    # print(lines)
    return lines


# readText('/Users/zw/Desktop/Data Learning/Machine Learning Basics/Spam Filter/ham/1.txt',
#          encoding='utf-8')


def email_parser(email_path):
    punctuations = """,.<>()*&^%$#@!'";~`[]{}|、\\/~+_-=?"""
    content_list = readText(email_path, 'utf-8')
    content = (' '.join(content_list)).replace('\r\n', ' ').replace('\t', ' ')  # LF, CR, HT
    clean_word = []
    for punctuation in punctuations:
        content = (' '.join(content.split(punctuation))).replace('  ', ' ')
        clean_word = [word.lower for word in content.split(' ') if len(word) >= 2]
    # print(clean_word)
    return clean_word


# email_parser('/Users/zw/Desktop/Data Learning/Machine Learning Basics/Spam Filter/ham/1.txt')

def get_word(email_file):
    word_list = []
    word_set = []
    email_paths = fileWalker(email_file)
    for each_email_path in email_paths:
        clean_word = email_parser(each_email_path)
        word_list.append(clean_word)
        word_set.extend(clean_word)
    return word_list, set(word_set)

get_word('/Users/zw/Desktop/Data Learning/Machine Learning Basics/Spam Filter/ham/1.txt')