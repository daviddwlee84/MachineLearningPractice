## FP-Growth Frequent Itemset Mining From Scratch Version
#
# Author: David Lee
# Create Date: 2018/10/23
#
# Detail:
#   Total Data = 88163

import numpy as np
import pandas as pd # Read csv

class FPTreeNode:
    def __init__(self, item=None, numOccur=1):
        self.item = item # 'Value' of the item (root maybe None)
        self.count = numOccur # Number of times the item occurs in a transaction
        self.nodeLink = None
        self.children = {} # Child nodes in the FP Growth Tree

    def increaseOccur(self, numOccur):
        self.count += numOccur
    
    def displayNodes(self, indent=1):
        print("%s%s:%s" % ('  '*indent, self.item, self.count))
        for child in self.children.values():
            child.displayNodes(indent+1)

class FPGrowth:
    def __init__(self, min_sup=1):
        self.__min_sup = min_sup # The minimum of transactions in an itemset need to occur in to be deemed frequent
        self.tree_root = None

    ## Build FP tree helper

    # Count the number of transactions that contains item
    def __calculate_support(self, item, transactions):
        count = 0
        for transaction in transactions:
            if item in transaction:
                count += 1
        return count
    
    # Returns a set of frequent items
    # an item is determined to be frequent if there are at least min_sup transactions that contains it
    def __get_frequent_items(self, transactions):
        # Get all unique items in the transactions
        unique_items = set(item for transaction in transactions for item in transaction)
        items = []

        # Calculate support for each item
        for item in unique_items:
            support = self.__calculate_support(item, transactions)
            if support >= self.__min_sup:
                # If it's a frequent item than add to list
                items.append([item, support])
            
        # Sort by support in descending order
        items.sort(key=lambda item: item[1], reverse=True)
        frequent_items = [[element[0]] for element in items]

        # Only return the frequent items
        return frequent_items
    
    # Recursively add nodes to the tree
    def __insert_tree(self, node, children):
        if not children:
            # If no more items to add
            return
        # Create new node as the first item in children list (a transaction)
        child_item = children[0]
        child = FPTreeNode(item=child_item)
        if child_item in node.children:
            # If parent already contains item => increase the support
            node.children[child.item].count += 1
        else:
            # Else add new child
            node.children[child.item] = child
        # Recursively call on the rest of the children list from the new node
        self.__insert_tree(node.children[child.item], children[1:])

    # Construct FP tree
    def __construct_tree(self, transactions, frequent_items=None):
        if not frequent_items:
            # Get frequent items sorted by support
            frequent_items = self.__get_frequent_items(transactions)
        
        unique_frequent_items = list(set(item for itemset in frequent_items for item in itemset))
        
        # Construct the root of the FP Growth tree
        root = FPTreeNode()
        for transaction in transactions:
            # Remove items that are not frequent according to unique_frequent_items
            transaction = [item for item in transaction if item in unique_frequent_items]
            transaction.sort(key=lambda item: frequent_items.index([item]))
            self.__insert_tree(root, transaction)
        
        return root
    

    def print_tree(self, node=None, indent=0):
        #self.tree_root.displayNodes() # Same usage
        if not node:
            node = self.tree_root
        print("%s%s:%s" % ('  '*indent, node.item, node.count))
        for child_key in node.children:
            child = node.children[child_key]
            self.print_tree(child, indent+1)

    # Build FP tree
    def fit(self, transactions):
        self.__transactions = transactions
        # Build the FP Growth Tree
        self.tree_root = self.__construct_tree(transactions)

    # Mine frequent itemsets from the FP-tree
    def find_frequent_itemsets(self):
        pass

def loadSimpleData():
    simpleData = [['r', 'z', 'h', 'j', 'p'],
                  ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                  ['z'],
                  ['r', 'x', 'n', 'o', 's'],
                  ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                  ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpleData

def main():
    # Test simple data of the example on the book Machine Learning in Action
    data = loadSimpleData()
    FPGrowthModel = FPGrowth(min_sup=3)
    FPGrowthModel.fit(data)
    FPGrowthModel.print_tree()
    
if __name__ == '__main__':
    main()
