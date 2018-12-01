import sys
from jieba import posseg
from pyhanlp import HanLP

def jieba_seg(content):
    segments = posseg.cut(content)
    for seg in segments:
        if seg.flag != 'x':
            print(seg.word + '/' + seg.flag, end=' ')

def HanLP_seg(content):
    segments = HanLP.segment(content)
    for term in segments:
        if str(term.nature) != 'w':
            print('{}/{}'.format(term.word, term.nature), end=' ')

def main():
    with open('article.txt', 'r') as f:
        content = f.read()

    print("\n\n===== Jieba =====\n")
    jieba_seg(content)

    print("\n\n===== HanLP =====\n")
    HanLP_seg(content)

if __name__ == "__main__":
    main()
