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

The variables can be configured based on the scenario. Here, it's assumed that prior probabilities are equally distributed and all features are taken: 
![Vars](Qn1/Vars.png)

The input is the sample dataset, each set separated by the class they belong to as given below:
![Data](Qn1/data.png)

In order to classify the sample data, we first run the function through our sample dataset, classwise. On each sample, we find the class which gives the maximum output from its discriminant function. 

A count and total count is maintained in order to find the success and failiure rates.

![Main](Qn1/main.png)

Assuming that all classes have an equal prior probability (as per the configuration in the example picture), the following output is produced:

![Output](Qn1/Output.png)

## Solution 2

### Part (a) and (b)
In order to match the question, the configuration variables are altered. 

- (data-1) for n indicates that only 2 classes will be considered (the final class would not be considered as its Prior probability is 0, implying that it wouldn't appear.)
- The d value is changed to 1, indicating that only 1 feature will be used. (which is x<sub>1</sub>)

![NewVars](Qn2/Vars.png)

The configuration parameters being passed are also changed.

- x\[0] indicates that only x^1 will be used.
- means\[i]\[0] indiciates that we need the mean only for x<sub>1</sub>).
- cov\[i]\[0]\[0] indicates the variance of feature x<sub>1</sub>).

![Main](Qn2/Main.png)

This results in the following output:

![Output1](Qn2/ab.png)

### Part (c)

Here, the configuration parameters are changed slightly.

- 'd's value is changed to 2, as now we are considering the first and second features.
- The matrix paramateres passed now include necessary values for the same reason.

![NewVars](Qn2/Varsc.png)

This results in the following output:
![Output2](Qn2/c.png)

### Part (d)

Here again, the configurations are changed in a similiar fashion as in (c).

- 'd' values is changed to 3 as all three features are now considered.
- The matrix paramaeteres are now passed without slicing as all values are important.

![NewVars](Qn2/Varsd.png)

The resuls in the following output:

![Output2](Qn2/d.png)

### Part (e)

On comparing the three outputs, using one or three features give more accurate results than using the first and second features.

![Output3](Qn2/e.png)

The reason for this could be because the covariance with the third feature is much higher than the ones associated with the second feature.

![Variance](Qn2/CoVar.png)

### Part (f)

In order to consider the possible configurations mentioned, the code takes an input vector and goes through all of them.
![Vars](Qn2/Varsf.png)

Here are the outputs for the 4 input vectors mentioned in the question:
![Otuput4](Qn2/f.png)
