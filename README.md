# IntegerProgramming
+ Uses scipy.optimize.linprog to calculate discrete optimization values. Allows all real-values greater than or equal to zero as discrete constraints (e.g. 1.5 unit variables).
+ Feasible variable values are examined in a specific order. Initially, we look at
	variables with the largest intervals. Variables with large intervals have fewer
	values in the feasible region, so 
	feasible region
	 vector in order of the largest discretionary interval to the smallest since larger
	 intervals inherently have fewer values in their feasible region. In the event of a
	 tie, the variable with the largest c-vector coefficient will be searched first
	 since the Integer Programming algorithm dismisses subtrees with profits <= the
	 current optimal profit. 
+ Integer Programming Algorithm based upon Section 3.4 of "Mathematical Modeling" by
	Mark M. Meerschaert as well as lectures by Professor Meerschaert.
+ Tested with Python 3.3.2 and SciPy 0.15.1
+ Usage (include in the same folder, there is no setup.py):
		from discreteopt import intprog
+ See discreteopt.py for intprog function definition and parameter explanation

Future Improvement(s):
1. Add a 'binary' constraint for the di field
