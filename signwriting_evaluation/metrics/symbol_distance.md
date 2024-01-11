# Evaluation metric for SignWriting
### Introduction
This code introduces a novel metric for assessing the similarity of two phrases written
in Formal SignWriting (FSW). Unlike generic string comparison methods like BLEU and CHRF, our approach
is tailored to the unique characteristics and rules of SignWriting, offering a task-specific evaluation.

### Evaluation Method
Our method addresses key aspects of SignWriting, such as:

- Symbols are organized in the FSW dictionary to reflect their types (e.g., hand signals, motion, touch), with proximity
indicating visual and semantic closeness.
- Symbols forming a sign can be written in different orders, representing the same visual output.
- Each symbol part has distinct meaning and importance, emphasizing aspects like symbol type, facing direction, angle, and position.

### Main concept
The evaluation process is built on three main stages, each with its own intent and purposes:
1. Symbol Distance Function: Evaluates similarity between two symbols based on SignWriting rules, considering custom 
weights for different symbol differences.
2. Distance Normalization: Normalizes distance values using the following non-linear function for better representation.

![Graph of f(x) = x^{\frac{1}{5}}](/assets/equations/graph1.png)

$$
f(x) = x^{\frac{1}{5}}
$$

3. Matching and Grading: Utilizes symbol distances to generalize similarity for entire signs. The Hungarian algorithm
matches similar parts, and using a weight calculated using the formula below, the weighted mean accounts for length differences.

![Graph of f(x) = x^{\frac{3}{2}}](/assets/equations/graph2.png)


$$
f(x) = x^{\frac{3}{2}}
$$
