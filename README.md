<h1>
    <center>CS4023D Artificial Intelligence</center>
    <center>Assignment 2</center>
</h1>
<p><center>By Dev Sony, B180297CS</center></p>
The question, report and source code can be found here.

[Github Repo]()

## Solution 1

Based on the formula given:
![Formula](Qn1/Formula.png)

The function has been defined:
![DF](Qn1/DF.png)

The variables can be configured based on the scenario: 
![Vars](Qn1/Vars.png)

The input is the sample dataset, each set separated by the class they belong to as given below:
![Data](Qn1/data.png)

In order to classify the sample data, we first run the function through our sample dataset, classwise. On each sample, we find the class which gives the maximum output from its discriminant function. 

A count and total count is maintained in order to find the success rates.

![Main](Qn1/main.png)

Assuming that all classes have an equal prior probability (as per the configuration in the example picture), the following output is produced:

![Output](Qn1/Output.png)

