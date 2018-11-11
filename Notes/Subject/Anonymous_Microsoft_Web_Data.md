# Anonymous Microsoft Web Data Data Set

## Dataset

[Anonymous Microsoft Web Data Data Set](https://archive.ics.uci.edu/ml/datasets/Anonymous+Microsoft+Web+Data)
([R language](https://rdrr.io/github/mhahsler/recommenderlab/man/MSWeb.html))

### Data Set Information

We created the data by sampling and processing the www.microsoft.com logs. The data records the use of www.microsoft.com by 38000 anonymous, randomly-selected users. For each user, the data lists all the areas of the web site (Vroots) that user visited in a one week timeframe.

Users are identified only by a sequential number, for example, User #14988, User #14989, etc. The file contains no personally identifiable information. The 294 Vroots are identified by their title (e.g. "NetShow for PowerPoint") and URL (e.g. "/stream"). The data comes from one week in February, 1998.

### Abstract

-|-
-|-
Data Set Characteristics |N/A
Attribute Characteristics|Categorical
Number of Attributes     |294
Number of Instances      |37711
Associated Tasks         |Recommender-Systems

### Source

* Creator:

    Jack S. Breese, David Heckerman, Carl M. Kadie
    Microsoft Research, Redmond WA, 98052-6399, USA
    breese '@' microsoft.com, heckerma '@' microsoft.com, carlk '@' microsoft.com

* Donor:

    Breese:, Heckerman, & Kadie

### Relevant Information

```txt
We created the data by sampling and processing the www.microsoft.com logs.
The data records the use of www.microsoft.com by 38000 anonymous,
randomly-selected users. For each user, the data lists all the areas of
the web site (Vroots) that user visited in a one week timeframe.

Users are identified only by a sequential number, for example, User #14988,
User #14989, etc. The file contains no personally identifiable information.
The 294 Vroots are identified by their title (e.g. "NetShow for PowerPoint")
and URL (e.g. "/stream"). The data comes from one week in February, 1998.

Dataset format:
-- The data is in an ASCII-based sparse-data format called "DST".
    Each line of the data file starts with a letter which tells the line's type.
    The three line types of interest are:
        -- Attribute lines:
        For example, 'A,1277,1,"NetShow for PowerPoint","/stream"'
                Where:
                'A' marks this as an attribute line,
                '1277' is the attribute ID number for an area of the website
                            (called a Vroot),
                '1' may be ignored,
                '"NetShow for PowerPoint"' is the title of the Vroot,
                '"/stream"' is the URL relative to "http://www.microsoft.com"
        -- Case and Vote Lines:
            For each user, there is a case line followed by zero or more vote lines.
                For example:
                    C,"10164",10164
                    V,1123,1
                    V,1009,1
                    V,1052,1
                Where:
                    'C' marks this as a case line,
                    '10164' is the case ID number of a user,
                    'V' marks the vote lines for this case,
                    '1123', 1009', 1052' are the attributes ID's of Vroots that a
                        user visited.
                    '1' may be ignored.
```

* Attribute (A): the description of the website area
* Case (C): the case for each user, containing its ID
* Vote (V): the vote lines for the case

## Result

The ratings matrix is preprocessed by independent R program

Model           |Energy|# of sigma
----------------|------|----------
SVD From Scratch|90%   |165
SVD From Scratch|80%   |118

You can get a recommendation result as a table (and also information about how we recommend in percentage)

e.g.

```txt
We recommend X1005 (100.0 % recommend)
description    "Norway"
url            "/norge"
```

```txt
Recommendation Table
                    description         url
X1001         "Support Desktop"  "/support"
X1002  "End User Produced View"   "/athome"
X1003          "Knowledge Base"       "/kb"
X1004    "Microsoft.com Search"   "/search"
X1005                  "Norway"    "/norge"
```

## Book

* Building a Recommendation System with R - Ch5 Case Study - Building Your Own Recommendation Engine

## Link

### Correlated project

* [amirkrifa/ms-web-dataset - Recommendation & clustering algorithms](https://github.com/amirkrifa/ms-web-dataset)
* [perwagner/link_recommender](https://github.com/perwagner/link_recommender)
