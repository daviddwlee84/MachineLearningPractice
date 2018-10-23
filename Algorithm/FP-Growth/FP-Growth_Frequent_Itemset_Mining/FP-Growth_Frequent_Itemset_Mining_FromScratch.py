## FP-Growth Frequent Itemset Mining From Scratch Version
#
# Author: David Lee
# Create Date: 2018/10/23
#
# Detail:
#   Total Data = 88163

class FPTreeNode:
    def __init__(self, item=None, numOccur=1):
        self.item = item # 'Value' of the item (root maybe None)
        self.count = numOccur # Number of times the item occurs in a transaction
        self.nodeLink = None
        self.children = {} # Child nodes in the FP Growth Tree
    
    def displayNodes(self, indent=1):
        print("%s%s:%s" % ('  '*indent, self.item, self.count))
        for child in self.children.values():
            child.displayNodes(indent+1)

class FPGrowth:
    def __init__(self, min_sup=1):
        self.__min_sup = min_sup # The minimum of transactions in an itemset need to occur in to be deemed frequent
        self.tree_root = None
        # Prefixes of itemsets in the FP Growth Tree
        self.prefixes = {}
        self.__frequent_itemsets = []

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

    ## Traversal FP tree helper

    # Make sure that the first item in itemset is a child of node
    # and that every following item in itemset is reachable via that path
    def __is_prefix(self, itemset, node):
        for item in itemset:
            if not item in node.children:
                return False
            else:
                node = node.children[item]
        return True

    # Determines the look of the hashmap key for self.prefixes
    # List of more strings than one gets joined by '-'
    def __get_itemset_key(self, itemset):
        if len(itemset) > 1:
            itemset_key = "-".join(itemset)
        else:
            itemset_key = str(itemset[0])
        return itemset_key

    # Recursive method that add prefixes to the itemset by traversing the FP Growth Tree
    def __determine_prefixes(self, itemset, node, prefixes=None):
        if not prefixes:
            prefixes = []
        
        # If the current node is a prefix to the itemset
        # add the current prefixes value as prefix to the itemset
        if self.__is_prefix(itemset, node):
            itemset_key = self.__get_itemset_key(itemset) # Transfer an itemset to a string as key
            if not itemset_key in self.prefixes:
                self.prefixes[itemset_key] = []
            self.prefixes[itemset_key] += [{"prefix": prefixes, "support": node.children[itemset[0]].count}]
            
        for child_key in node.children:
            child = node.children[child_key]
            # Recursive call with child as new node
            # Add the child item as potential prefix
            self.__determine_prefixes(itemset, child, prefixes + [child.item])
    
    # Calculate new frequent items from the conditional pattern base of suffix
    # Create Conditional FP-trees
    def __determine_frequent_itemsets(self, conditional_pattern_base, suffix=None):
        # Calculate new frequent items from the conditional database of suffix
        frequent_items = self.__get_frequent_items(conditional_pattern_base)

        cond_fp_tree = None

        if suffix:
            cond_fp_tree = self.__construct_tree(conditional_pattern_base, frequent_items)
            # Output new frequent itemset as the suffix added to the frequent items
            self.__frequent_itemsets += [element + suffix for element in frequent_items]
        
        # Find larger frequent itemset by finding prefixes of the frequent items
        # in the FP Growth Tree for the conditional database
        self.prefixes = {}
        for itemset in frequent_items:
            # If no suffix (first run)
            if not cond_fp_tree:
                cond_fp_tree = self.tree_root
            # Determine prefixes to itemset
            self.__determine_prefixes(itemset, cond_fp_tree)
            conditional_pattern_base = []
            itemset_key = self.__get_itemset_key(itemset)
            # Build new conditional database
            if itemset_key in self.prefixes:
                for element in self.prefixes[itemset_key]:
                    # If support = 4 => add 4 of the corresponding prefix set
                    for _ in range(element["support"]):
                        conditional_pattern_base.append(element["prefix"])
                
                # Create new suffix
                new_suffix = itemset + suffix if suffix else itemset
                self.__determine_frequent_itemsets(conditional_pattern_base, new_suffix)

    # Mine frequent itemsets from the FP-tree
    def find_frequent_itemsets(self):
        if not self.__frequent_itemsets:
            self.__determine_frequent_itemsets(self.__transactions)
        return self.__frequent_itemsets

def loadSimpleData():
    simpleData = [['r', 'z', 'h', 'j', 'p'],
                  ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
                  ['z'],
                  ['r', 'x', 'n', 'o', 's'],
                  ['y', 'r', 'x', 'z', 'q', 't', 'p'],
                  ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpleData


def loadRetailData():
    retail_data = []
    with open("Datasets/retail.csv", 'r') as retail_file:
        lines = retail_file.read().splitlines() # Get rid of '\n'
        for line in lines:
            retail_data.append(line.split(','))
    return retail_data

def main():
    # Test simple data of the example on the book Machine Learning in Action
    print("==== Example of book ====")
    data = loadSimpleData()
    FPGrowthModel = FPGrowth(min_sup=3)
    FPGrowthModel.fit(data)
    FPGrowthModel.print_tree()
    print(FPGrowthModel.find_frequent_itemsets())
    
    from datetime import datetime
    # Retail Market Basket Data Set
    print("==== Retail Market Basket Data Set ====")
    retail_data = loadRetailData()
    FPGrowthModel = FPGrowth(min_sup=5000)
    print("Building the tree...")
    startTime = datetime.now()
    FPGrowthModel.fit(retail_data)
    print("Take %s to build the tree" % (str(datetime.now() - startTime)))

    #FPGrowthModel.print_tree()
    
    print("Mining the tree...")
    startTime = datetime.now()
    print(FPGrowthModel.find_frequent_itemsets())
    print("Take %s to mine the tree" % (str(datetime.now() - startTime)))
    
if __name__ == '__main__':
    main()
