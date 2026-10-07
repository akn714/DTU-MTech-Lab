I am an MTech AI student and this is my AIML lab assignment.
I have attached TWO files:
1. Experiment 5.pdf — the official assignment instructions
2. adult.zip — the dataset provided for the assignment
I want you to complete this assignment as a single Jupyter Notebook (.ipynb).
MOST IMPORTANT REQUIREMENT
The final deliverable MUST be:
Experiment_5_Decision_Tree.ipynb
Do NOT make the main implementation a .py file.
Do NOT split the implementation into multiple Python files.
Do NOT create unnecessary supporting files.
Everything — preprocessing, decision tree implementation, experiments, evaluation, pruning, comparisons, plots/tables, and conclusion — should be contained inside the .ipynb notebook.
The notebook should be directly usable in Jupyter Notebook, JupyterLab, or Google Colab after the dataset path is set correctly.
 
⸻
 
1. FOLLOW THE PDF EXACTLY
First carefully read Experiment 5.pdf.
Treat the PDF as the source of truth for the assignment.
Do not add requirements that are not present in the PDF.
Do not remove any requirement from the PDF.
Do not turn this into a large production-level machine learning project.
The goal is a simple, clean, understandable student lab implementation that satisfies the assignment and that I can explain during a viva.
 
⸻
 
2. INSPECT THE DATASET FIRST
Before writing the final notebook, inspect adult.zip.
Determine:
• what files are inside the ZIP
• which file(s) contain the actual dataset
• column names
• data types
• missing-value representation
• target-column representation
• approximate dataset size
Do not blindly assume filenames or column names.
Use the dataset I provided rather than downloading a different version from the internet.
If the ZIP contains multiple relevant files, explain briefly which one is being used and why.
 
⸻
 
3. IMPORTANT: FLAG AMBIGUITIES
If you find anything ambiguous or inconsistent in the assignment PDF, do not silently make up a solution.
For example, if the train/validation/test percentages appear inconsistent, explicitly point that out before implementing it.
For the dataset split, the PDF currently appears to state:
• 80% training
• 20% validation
• 20% test
These percentages total more than 100%, so do not silently choose your own interpretation.
Tell me what the PDF says and identify the ambiguity.
If a reasonable interpretation is necessary, clearly state the interpretation used in the notebook rather than pretending the requirement was unambiguous.
 
⸻
 
4. REQUIRED ASSIGNMENT
Implement a Decision Tree Classifier from scratch using NumPy and apply it to the Adult Income Dataset.
The assignment requires:
Data Preparation
• Handle missing values.
• Encode categorical variables into numeric values.
• Split the dataset into training, validation, and test sets according to the assignment’s intended procedure.
• Use the validation set for tuning depth and pruning.
Decision Tree From Scratch
Implement the actual decision tree yourself.
The implementation should:
• build the tree recursively
• calculate Gini Impurity
• calculate Entropy
• calculate weighted impurity of child nodes
• calculate information gain
• select the best split
• stop according to the required stopping conditions
• predict labels for new samples
The actual tree implementation must NOT simply call sklearn’s DecisionTreeClassifier.
Use NumPy for the core implementation.
Pre-Pruning
Implement the required pre-pruning techniques, including:
• maximum depth
• minimum number of samples required to split
• minimum impurity decrease only if appropriate/required
Experiment with:
• depth = 2
• depth = 4
• depth = 6
• unlimited depth
Post-Pruning
Implement the assignment’s Reduced Error Pruning.
The procedure should be:
1. Grow a full tree.
2. Consider replacing an internal node with a leaf containing the majority class.
3. Evaluate the change using the validation set.
4. Keep the pruning if validation accuracy does not decrease, as specified in the assignment.
5. Repeat until no further useful pruning can be performed.
Evaluation
Report:
• Accuracy
• Precision
• Recall
• F1-score
• Confusion Matrix
The final selected model should ultimately be evaluated on the test set.
Required Comparisons
Include:
1. Gini vs Entropy
2. Different tree depths:  
 • 2
• 4
• 6
• unlimited
1. Full tree vs pre-pruned vs post-pruned
2. Comparison with sklearn.tree.DecisionTreeClassifier
3. Identify important/top features, particularly the features appearing near the top of the tree.
 
⸻
 
5. STRICT RULES FOR THE IMPLEMENTATION
The from-scratch tree MUST NOT use:
sklearn.tree.DecisionTreeClassifier
• Random Forest
• XGBoost
• LightGBM
• CatBoost
• any other pre-built decision-tree implementation
Sklearn may ONLY be used for:
• the required comparison model
• standard evaluation metrics/utilities if appropriate
The actual decision tree algorithm must be implemented manually.
Do not hide the tree implementation behind another library.
 
⸻
 
6. KEEP THE CODE SIMPLE
This is extremely important.
I am a student and I need to understand the code.
Prefer:
simple + readable + correct
over:
complex + highly optimized + unnecessarily sophisticated
DO NOT add:
• GridSearchCV
• cross-validation unless explicitly required
• hyperparameter optimization frameworks
• complicated pipelines
• unnecessary classes
• excessive abstraction
• advanced Python tricks
• decorators
• unnecessary design patterns
• huge amounts of logging
• unnecessary data augmentation
• unnecessary visualizations
• unnecessary libraries
• unrelated ML techniques
• neural networks
• ensemble models
• unnecessary web scraping
• unrelated theory
Do not try to make the assignment look more impressive than it needs to be.
 
⸻
 
7. NOTEBOOK STRUCTURE
Create a clean notebook with approximately this structure:
1. Title
Experiment 5 — Decision Trees from Scratch
2. Objective
A short explanation based on the PDF.
3. Imports
Only import libraries that are actually needed.
4. Load Dataset
Load the supplied Adult dataset.
5. Dataset Inspection
Show:
• shape
• columns
• data types
• missing values
• a few sample rows
• target distribution
Keep this concise.
6. Data Preprocessing
Handle missing values and categorical variables.
Clearly explain what was done.
7. Train / Validation / Test Split
Follow the interpretation of the assignment discussed earlier.
Use a fixed random seed so results are reproducible.
8. Gini Impurity
Implement the Gini impurity calculation manually.
9. Entropy
Implement entropy manually.
10. Information Gain / Best Split
Implement the calculations needed to select the best split.
11. Decision Tree Node
Create a simple representation for tree nodes if needed.
12. Decision Tree From Scratch
Implement training recursively.
13. Prediction
Implement prediction for individual samples / batches.
14. Full Tree
Train and evaluate the baseline full tree.
15. Pre-Pruning
Run the required depth experiments:
• 2
• 4
• 6
• unlimited
16. Post-Pruning
Implement reduced-error pruning using the validation set.
17. Evaluation
Calculate:
• Accuracy
• Precision
• Recall
• F1
• Confusion Matrix
18. Gini vs Entropy
Provide a simple comparison table.
19. Depth Comparison
Provide a simple results table.
20. Full vs Pre-Pruned vs Post-Pruned
Provide a comparison table.
21. Important Features
Show the important/top features based on the actual implemented tree.
22. sklearn Comparison
Train:
sklearn.tree.DecisionTreeClassifier
and compare its performance with the from-scratch implementation.
23. Final Conclusion
Briefly summarize the results.
Do not write a huge essay.
 
⸻
 
8. NOTEBOOK STYLE
Use Markdown cells for short explanations and Code cells for implementation.
Do not put enormous blocks of prose into the notebook.
Each section should explain:
• what we are doing
• why we are doing it
• then the code
Keep explanations at an MTech lab-report level.
Do not make the notebook look like a research paper.
 
⸻
 
9. DATA LEAKAGE
Be careful about data leakage.
Any preprocessing that learns information from the dataset should be handled appropriately with respect to the train/validation/test split.
The validation set should be used for:
• depth selection
• pruning decisions
The test set should be used only for final evaluation.
Do not use the test set to select the best hyperparameters.
 
⸻
 
10. PERFORMANCE
The Adult dataset is reasonably large.
Make the implementation practical enough to actually run.
If the completely naive implementation would take an unreasonable amount of time, optimize the implementation only where necessary.
However:
Do NOT replace the from-scratch implementation with sklearn or another decision-tree library just to make it faster.
The algorithm must still genuinely implement the decision tree from scratch.
If you make any performance-oriented simplification, explain it briefly.
 
⸻
 
11. REPRODUCIBILITY
Use a fixed random seed wherever randomness is involved.
Make sure the notebook runs from top to bottom in the correct order.
Do not rely on variables that were created in some earlier hidden step.
Do not assume the user has manually executed cells in a particular order.
 
⸻
 
12. FINAL QUALITY CHECK
Before giving me the final .ipynb, check the notebook carefully.
Verify:
• all imports work
• dataset loading works
• preprocessing works
• categorical encoding works
• train/validation/test split works
• Gini works
• Entropy works
• information gain works
• tree construction works
• prediction works
• full tree works
• pre-pruning works
• post-pruning works
• evaluation works
• comparison tables work
• sklearn comparison works
• notebook cells are in the correct order
• there are no undefined variables
• there are no unnecessary libraries
• there are no placeholder sections
• there are no fake results
• there are no invented dataset values
• there is no accidental use of sklearn to implement the actual tree
Most importantly, make sure the notebook actually follows the assignment PDF.
 
⸻
 
13. DO NOT INVENT RESULTS
Do not fabricate accuracy, precision, recall, F1, confusion matrices, feature importance, or any other results.
Actually run the code and use the resulting values.
If something cannot be executed in your environment, clearly tell me instead of making up an output.
 
⸻
 
14. FINAL DELIVERABLE
After completing everything, give me the actual:
Experiment_5_Decision_Tree.ipynb
This is the file I will use as my lab submission.
Do not give me a .py file instead.
Do not make me manually copy code into Jupyter.
Do not create unnecessary extra files.
The notebook should be clean, straightforward, executable, and easy for a student to understand and explain in a viva.
Before generating the final notebook, briefly state any ambiguity you found in the PDF and how you handled it.