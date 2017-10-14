class Node:

    def __init__(self, zerocount, onecount, feature, entropy, number, side, inputfile, classlabel, headerlist):
        self.left = None
        self.feature = feature
        self.right = None
        self.zerocount = zerocount
        self.onecount = onecount
        self.entropy = entropy
        self.number = number
        self.side = side
        self.inputfile = inputfile
        self.classlabel = classlabel
        self.headerlist = headerlist

class Tree:

    def createNode(self, zerocount, onecount, feature, entropy, number, side, inputfile, classlabel, headerlist):
        return Node(zerocount, onecount, feature, entropy, number, side, inputfile, classlabel, headerlist)

    def insert(self, node, zerocount, onecount, feature, entropy, number, side, inputfile, classlabel, headerlist):
        # if tree is empty , return a root node
        if node is None:
            return self.createNode(zerocount, onecount, feature, entropy, number, side, inputfile, classlabel, headerlist)
        if(side=="l"):
            node.left = self.insert(node.left, zerocount, onecount, feature, entropy, number, side, inputfile, classlabel, headerlist)
            return node.left
        else:
            node.right = self.insert(node.right, zerocount, onecount, feature, entropy, number, side, inputfile, classlabel, headerlist)
            return node.right

    def print(self,root,count):
        if root is not None:
            if(root.feature!= 'Class'):
                print("%s %s = %s" %('| '*count,root.feature[:2],root.feature[2:])) if(root.left is not None and root.right is not None) else  print("%s %s = %s : %s" %('| '*count,root.feature[:2],root.feature[2:],root.classlabel))
            count+=1
            self.print(root.left,count)
            self.print(root.right,count)

    def deleteNode(self,number,root):
        if(root is None):
            return False
        elif(root.left is not None and root.left.number == number):
            root.left = None
            return True
        elif(root.right is not None and root.right.number == number):
            root.right = None
            return True

        found = self.deleteNode(number,root.left)
        if (found!=True):
            return self.deleteNode(number, root.right)
        else:
            return found
