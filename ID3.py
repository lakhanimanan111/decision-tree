import csv,math,pandas,random,sys
from asyncio.futures import _TracebackLogger

from DecisonTreeNode import Tree,Node
from queue import Queue

tree = Tree()

training_file_path = sys.argv[1]
#training_file_path = "C:/Users/manan/Documents/Courses/Semester3/CS6375.001_MachineLearning/Assignment2/data_sets1/training_set.csv"
training_file = pandas.read_csv(training_file_path)
training_file = training_file.dropna()

validation_path = sys.argv[2]
# validation_path = "C:/Users/manan/Documents/Courses/Semester3/CS6375.001_MachineLearning/Assignment2/data_sets1/validation_set.csv"
validationfile = pandas.read_csv(validation_path)

test_file_path = sys.argv[3]
# test_file_path = "C:/Users/manan/Documents/Courses/Semester3/CS6375.001_MachineLearning/Assignment2/data_sets1/test_set.csv"
testfile = pandas.read_csv(test_file_path)

q = Queue()
nodecount =0

def entropy(zerocount,tcount):

    if(tcount != 0):
        p1 = zerocount/tcount
    else:
        p1 = 0

    p2 = 1 - p1
    e=0
    if (p1!=0 and p2!=0):
        e= p1*math.log(1/p1, 2) + p2*math.log(1/p2, 2)
    return e

def information_gain(e1,e2):
   gain = e1-e2
   return gain


def utility(rootNode):
    maxInformationGain = 0
    maxAttribute = ""
    leftZeroCount = 0
    leftTotalCount = 0
    leftEntropy = 0
    rightZeroCount = 0
    rightTotalCount = 0
    rightEntropy = 0
    rootNodeFile = rootNode.inputfile
    headerList = rootNode.headerlist[:]

    for header in headerList:
        # Entropy of records with rootNode classified as 0
        zerofile = rootNodeFile[rootNodeFile[header] == 0]
        classzero_file0 = zerofile[zerofile.Class == 0].Class.count()
        tzeroCount = zerofile.Class.count()
        e0 = entropy(classzero_file0, tzeroCount)

        # Entropy of records with rootnode classified as 1
        onefile = rootNodeFile[rootNodeFile[header] == 1]
        classzero_file1 = onefile[onefile.Class == 0].Class.count()
        toneCount = onefile.Class.count()
        e1 = entropy(classzero_file1, toneCount)

        #Weighted Entropy
        total = toneCount + tzeroCount
        weightedentropy = e0 * (tzeroCount / total) + e1 * (toneCount / total)

        # Calculating Information Gain
        ig = information_gain(rootNode.entropy, weightedentropy)
        # print(header,ig,e0,e1,weightedentropy)
        #Finding attribute with maximum information gain
        if (ig > maxInformationGain):
            maxInformationGain = ig
            maxAttribute = header
            leftZeroCount = classzero_file0
            leftTotalCount = tzeroCount-classzero_file0
            leftEntropy = e0
            rightZeroCount = classzero_file1
            rightTotalCount = toneCount-classzero_file1
            rightEntropy = e1

    # Labeling internal nodes based on number of zero classfied records and one classified record
    lclasslabel = 0  if leftZeroCount > leftTotalCount - leftZeroCount else 1
    rclasslabel = 0 if rightZeroCount > rightTotalCount - rightZeroCount else 1

    # Removing the attribute with max information gain from list and creating 2 nodes namely left and right. Also adding the nodes to the queue

    if maxAttribute!="":
        headerList.remove(maxAttribute)
        global nodecount
        nodecount+=1
        q.put(tree.insert(rootNode, leftZeroCount, leftTotalCount, maxAttribute + "0", leftEntropy, nodecount, "l", rootNodeFile[rootNodeFile[maxAttribute] == 0], lclasslabel, headerList))
        nodecount+=1
        q.put(tree.insert(rootNode, rightZeroCount, rightTotalCount, maxAttribute + "1", rightEntropy, nodecount, "r", rootNodeFile[rootNodeFile[maxAttribute] == 1], rclasslabel, headerList))

# Method to verify our model on various set
def test(inputfile,rootNode):
    truecount =0;   #truecount is count of rows in dataset which are corrrectly classified by model
    falsecount =0;
    for index,row in inputfile.iterrows():
        if (testhelper(rootNode,row)==True):
            truecount += 1
        else:
            falsecount +=1
    return(truecount/(truecount+falsecount))

def testhelper(rootNode,row):
    if(rootNode is None):
        return False
    elif(rootNode.left is  None and rootNode.right is None):
        if rootNode.classlabel == row['Class']:
            return True
        else:
            return False
    elif(rootNode.left is not None):
        feature = rootNode.left.feature[:2]
        value = rootNode.left.feature[2:]
    elif (rootNode.right is not None):
        feature = rootNode.right.feature[:2]
        value = rootNode.right.feature[2:]

    # print(str(row[feature])[:1],value)
    if(str(row[feature])[:1] == value):
       match = testhelper(rootNode.left,row)
       if(match==False):
            return testhelper(rootNode.right, row)
       else:
           return match
    else:
        return testhelper(rootNode.right, row)

# Print summary of our model
def print_summary(inputfile,accuracy,rootNode,type):
    print("Number of %s instance = %s" %(type,inputfile.Class.count()))
    print("Number of %s attributes = %s" %(type,(len(inputfile.columns))-1)) #-1 because we are not counting class as attribute
    if(type=="training set"):
        print("Total number of Nodes in the Tree = %s" % (nodecount))
        print("Total number of leaf nodes in the tree = %s" %(count_leafnode(rootNode)))
    print("Accuracy of the model on %s = %.2f" %(type,accuracy*100))
    print()

# Recursive method to calculate total number of leaf nodes
def count_leafnode(rootNode):
    if(rootNode is None):
        return 0
    elif(rootNode.left is None and rootNode.right is None):
        return 1
    return count_leafnode(rootNode.left) + count_leafnode(rootNode.right)

def pruning(pruning_factor,rootNode,totalnodes):
    todelete = int(pruning_factor*totalnodes)
    for x in range(todelete):
        tree.deleteNode(random.randint(1,totalnodes+1),rootNode)

def main():

        # Calculate entropy of first node
        zeroc = training_file[training_file.Class == 0].Class.count()
        tcount = training_file.Class.count()
        root_entropy = entropy(zeroc,tcount)

        # Get all the column headers in a list
        headerList = [column.rstrip('\t') for column in training_file.columns]

        # Remove Class as it not a feature
        headerList.remove('Class')

        # Create a root node with this entropy
        root = None
        root = tree.insert(root, zeroc, tcount - zeroc, 'Class', root_entropy, 0,"", training_file, -1, headerList)

        # Put the root in a Queue
        q.put(root)

        # While Queue is not empty, keep calling utility function to further divide the dataset based on features in Queue
        while (q.empty() == False):
            utility(q.get())

        # Print tree in desired format
        tree.print(root,-1)

        # Testing model on training set and caculating accuracy
        print("Pre Pruned accuracy\n--------------------------------------")
        trainingaccuracy =  test(training_file, root)
        print_summary(training_file,trainingaccuracy,root,"training set")

        validationaccuracy = test(validationfile, root)
        print_summary(validationfile, validationaccuracy, root, "validation set")

        testaccuracy =  test(testfile,root)
        print_summary(testfile, testaccuracy, root, "test set")

        pruning(float(sys.argv[4]),root,nodecount)
        # pruning(0.2, root, nodecount)
        print("Post Pruned accuracy\n--------------------------------------")
        trainingaccuracy = test(training_file, root)
        print_summary(training_file, trainingaccuracy, root, "training set")

        validationaccuracy = test(validationfile, root)
        print_summary(validationfile, validationaccuracy, root, "validation set")

        testaccuracy = test(testfile, root)
        print_summary(testfile, testaccuracy, root, "test set")

main()