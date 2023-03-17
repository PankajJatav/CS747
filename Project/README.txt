Files:

0. AskMe.pdf
Paper that was used as basis for this project

1. log.log
Output of training; for each task, each epoch reports two numbers. The first number is the training loss and the second number is the evaluation loss.  *NOTE*: the log does not report the task outputs in order -- the order is defined by the ordering of the accuracy report at the bottom of the file.

2. Report
Final report

3. fetch_babi_data.sh
Script used to download the training and evaluation data.  The dataset used is Facebook's bAbI task data and can be found in many places.  The dataset can also be manually downloaded from: https://research.facebook.com/downloads/babi/

4. Plots
Plots for trianing and evaluation results
- 4 plots of training/evaluation loss vs epochs grouped by 5 tasks (20 total task)
- 20 individual plots of training/evaltion loss vs epochs
- 1 plot for accuracy by task
- 1 plot for accuracy by task (vs paper's accuracy)

5. src/__OLD__
***THIS CAN BE IGNORED***
Previous iterations of the final code

6. src/DMN.py
Code that creates the model definition

7. src/helper.py
Code that preprocesses the data and preforms the training and evaluation process

8. postprocess.py
Code that parses the log file and generates the plots
