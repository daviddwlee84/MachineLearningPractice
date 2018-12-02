# Document Similarity and Latent Semantic Indexing

## Dataset

[Corpus of the People's Daily / 中國人民日報標注語料庫 (PFR)](http://dx.doi.org/10.18170/DVN/SEYRX5)

### Structure

1. 年月日-版號-篇章號-段號
2. Interval of each paragraph is two empty lines

### Tagset

[The PKU tagset](http://www.lancaster.ac.uk/fass/projects/corpus/babel/PKU_tagset.htm)

Tag|Meaning
---|---------
a  |adjective
ad |adjective as adverbial
ag |adjective morpheme
an |adjective with nominal function
b  |non-predicate adjective
bg |non-predicate adjective morpheme
c  |conjunction
cg |conjunction morpheme
d  |adverb
dg |adverb morpheme
e  |interjection
ew |sentential puncuation
f  |directional locality
fg |locality morpheme
g  |morpheme
h  |prefix
i  |idiom
j  |abbreviation
k  |suffix
l  |fixed expressions
m  |numeral
mg |numeric morpheme
n  |common noun
ng |noun morpheme
nr |personal name
ns |place name
nt |organization name
nx |nominal charachter string
nz |other proper noun
o  |onomatope
p  |preposition
pg |preposition morpheme
q  |classifier
qg |classifier morpheme
r  |pronoun
rg |pronoun morpheme
s  |space word
t  |time word
tg |time word morpheme
u  |auxiliary
v  |verb
vd |verb as adverbial
vg |verb morpheme
vn |verb with nominal function
w  |symbol and non-sentential punctuation
x  |unclassified items
y  |modal particle
yg |modal particle morpheme
z  |descriptive
zg |descriptive morpheme

代碼 |名稱
----|-----
Ag  |形語素
a   |形容詞
ad  |副形詞
an  |名形詞
Bg  |區別語素
b   |區別詞
c   |連詞
Dg  |副語素
d   |副詞
e   |嘆詞
f   |方位詞
g   |語素
h   |前接成分
i   |成語
j   |簡略語
k   |後接成分
l   |習用語
Mg  |數語素
m   |數詞
Ng  |名語素
n   |名詞
nr  |人名
ns  |地名
nt  |機構團體
nx  |外文字符
nz  |其它專名
o   |擬聲詞
p   |介詞
Qg  |量語素
q   |量詞
Rg  |代語素
r   |代詞
s   |處所詞
Tg  |時間語素
t   |時間詞
Ug  |助語素
u   |助詞
Vg  |動語素
v   |動詞
vd  |副動詞
vn  |名動詞
w   |標點符號
x   |非語素字
Yg  |語氣語素
y   |語氣詞
z   |狀態詞

## Result

### VSM

Output diagonal element is 1. (self-similarity)

```txt
In : df.shape
Out: (8880, 8880)

In : df.head()
Out:
       0         1         2         3         4         5         6     \
0  1.000000  0.247370  0.031764  0.011568  0.069333  0.000000  0.104921
1  0.247370  1.000000  0.074034  0.093247  0.114389  0.118067  0.099436
2  0.031764  0.074034  1.000000  0.199128  0.240452  0.000000  0.318746
3  0.011568  0.093247  0.199128  1.000000  0.121334  0.024070  0.152113
4  0.069333  0.114389  0.240452  0.121334  1.000000  0.000000  0.148902
```

### SVD

```txt
In : df.shape
Out: (8880, 8880)

In : df.head()
Out:

(2-dimensional LSI space)

       0         1         2         3         4         5         6     \
0  1.000000  0.972027  0.986259  0.965273  0.980496  0.753523  0.988651
1  0.972027  1.000000  0.997472  0.999629  0.999230  0.886853  0.996280
2  0.986259  0.997472  1.000000  0.995168  0.999492  0.851778  0.999885
3  0.965273  0.999629  0.995168  1.000000  0.997791  0.899103  0.993565
4  0.980496  0.999230  0.999492  0.997791  1.000000  0.868036  0.998895

(5-dimensional LSI space)

       0         1         2         3         4         5         6     \
0  1.000000  0.983483  0.848290  0.796135  0.629115  0.691262  0.791310
1  0.983483  1.000000  0.894295  0.839724  0.696409  0.727351  0.818970
2  0.848290  0.894295  1.000000  0.907489  0.878193  0.536798  0.933318
3  0.796135  0.839724  0.907489  1.000000  0.910081  0.370954  0.861982
4  0.629115  0.696409  0.878193  0.910081  1.000000  0.406662  0.728450

(100-dimensional LSI space)

       0         1         2         3         4         5         6     \
0  1.000000  0.653180  0.215969  0.222961  0.154828  0.149586  0.406351
1  0.653180  1.000000  0.376675  0.380387  0.363116  0.256816  0.463112
2  0.215969  0.376675  1.000000  0.475014  0.592771 -0.006707  0.613748
3  0.222961  0.380387  0.475014  1.000000  0.350812 -0.039246  0.369790
4  0.154828  0.363116  0.592771  0.350812  1.000000 -0.022277  0.382058
```

## Links

### Dataset explanation

* [簡書 - 人民日報標注語料庫（PFR）](https://www.jianshu.com/p/30fa95e143bf)

### Similar project

* [使用VSM及LSI分別對人民日報標注語料庫（PFR）進行文章相似度分析](http://www.pkudodo.com/2018/11/09/%e4%bd%bf%e7%94%a8vsm%e5%8f%8alsi%e5%88%86%e5%88%ab%e5%af%b9%e4%ba%ba%e6%b0%91%e6%97%a5%e6%8a%a5%e6%a0%87%e6%b3%a8%e8%af%ad%e6%96%99%e5%ba%93%ef%bc%88pfr%ef%bc%89%e8%bf%9b%e8%a1%8c%e6%96%87%e7%ab%a0/)
* [關於VSM性能優化的思考](https://wyydsb.xin/other/vsm.html)
