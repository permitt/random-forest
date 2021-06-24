# Random Forest From Scratch

Implementation of Random Forest in Python and Go. Both sequential and parallel versions are developed. Visualization and performance comparison is done using Pharo. We'll compare the results based on increased datasets and execution time as well as incread number of Decision Trees and execution time.

### Definition

This project is focused on Random Forests for classification purposes.
Random Forest is an ensemble method. It is a form of bagging (bootstrap aggregating) algorithm which means that it consists of **multiple weak classificators** and makes a final prediction based on the majority of votes. It's main advantage is that it reduces variance - decreases overfitting. 
The base estimator for Random Forest is the Decision Tree.  

### Decision Tree
![alt text](https://lh4.googleusercontent.com/v9UQUwaQTAXVH90b-Ugyw2_61_uErfYvTBtG-RNRNB_eHUFq9AmAN_2IOdfOETnbXImnQVN-wPC7_YzDgf7urCeyhyx5UZmuSwV8BVsV8VnHxl1KtgpuxDifJ4pLE23ooYXLlnc)<br/>
_source: https://www.aitimejournal.com/@akshay.chavan/a-comprehensive-guide-to-decision-tree-learning_

The tree is built with the root note, decision nodes and leaf nodes. Root node and decision nodes consist of conditions.
We start with each example and run it through the tree, checking the conditions and navigating the tree accordingly. The final prediction is stored in the leaf node.

### Parallelization
As Random Forest consists of multiple weak classificators it is a great candidate for paralellization. (**bagging** generally is great for this purpose) Every classificator is run on the subset of data on different threads. As we have larger and larger datasets, the advantage of parallel approach is obvious.
* The complexity of sequential version is O(t * v * n * logn)
  * t is number of trees 
  * v is number of features we selected
  * n is number of samples we take into consideration

![alt_text](https://cdn1.bbcode0.com/uploads/2021/6/24/f5d0be789c85e5f513b541eedda71ecf-full.png)
<br/>
_source: https://cse.buffalo.edu/faculty/miller/Courses/CSE633/Hu-Fall-2012-CSE633.pdf_

### How to run the program


### Results and conclusion
