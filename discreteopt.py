from scipy.optimize import linprog
import math

# requires: Ax <= b, A_eq*x == b_eq, and x >= 0. These are all python lists
# param di: The discretionary interval for the variable set.
#						di=0 or di=None: Treat as regular continuous linear program
#						di=1: Corresponding variable must be an integer value
#						di may be a list of len(c) numerical values or a single numerical value.
#						The list indicates the corresponding variable's interval constraint.
#						A single numerical value indicates that all variables share the same
#						interval.
# param answer: Used for debugging.
# param callback: same attributes as the linprog parameter. Used for debugging.
# returns:	The optimal x-vector and the value of the objective function

def intprog(c, A, b, A_eq=None, b_eq=None, di=None, answer=None, callback=None):
	global data
	data = { # Optimal Solution Information (Hill-climbing behavior throught the program
					 "profit":None,
					 "optimal x":[0] * len(c),
					 "solution":answer,
					 "callback":callback,
					 # Problem Information (Immutable)
					 # Note that scipy.optimize.linprog aims to minimize the function
					 "c":[ -z for z in c ],
					 "var count":len(c),
					 "constraints":None,
					 # The precision-level allows for simplicity when comparing imprecise floating values to one
					 # another. Typically, we want to have PRECISION <= di * 0.01, where di is the required
					 # discretionary interval
					 "precision":list()

					 # Current Solution Information (Mutable, but must be cleared before end of a function
					 "A":A.copy(),
					 "b":b.copy(),
					 "A_eq":A_eq,
					 "b_eq":b_eq,
					 # bounds are INCLUSIVE on both ends
					 "bounds":[ [0,None] ] * len(c),
					 # So as to avoid stack overflow and wasting time rechecking the same
					 # tableaus
					 "past solutions":list(),
					 # for debugging purposes
					 "num tabs":-1 }

	if type(di) is list:
		if len(di) != data["var count"]:
			raise TypeError("di must have the same length as c")
		data["constraints"] = [ abs(i) for i in di ]
	elif type(di) is int or type(di) is float:
		data["constraints"] = [abs(di)] * data["var count"]
	elif di == None:
		data["constraints"] = [0] * data["var count"]
	else:
		raise TypeError("di must be a list, an int, or None")
	
	for i in range(data["var count"]):
		data["precision"].append(_precision(i))
	
	_iphelper()
	return data["optimal x"], data["profit"]

def _iphelper(callback=None):
	global data
	data["num tabs"] += 1

	soln = linprog(data["c"], A_ub=data["A"], b_ub=data["b"], A_eq=data["A_eq"],
								 b_eq=data["b_eq"], callback=data["callback"])

	if not soln.success:
		return
	
	var_range = range(data["var count"])
	x = tuple([ soln.x[i] for i in var_range ])

	if x in data["past solutions"]:
		return
	data["past solutions"].append(x)

	z = _dot_product(data["c"], x)

	if data["profit"] is not None and z > data["profit"]:
		return

	valid = [ _constrained(x[i], i) for i in var_range ]
	if all(valid):
		x = tuple([ _round(soln.x[i], i) for i in var_range ])
		z = _dot_product(data["c"], x)
		if data["profit"] is None or z <= data["profit"]:
			data["optimal x"] = x
			data["profit"] = z
			return

	for i in var_range:
		if not valid[i]:
			lower = _floor(x[i], i)
			upper = _ceil(x[i], i)

			# first look at the values less than this solution value
			bounds = data["bounds"][i]
			if lower == bounds[0]:
				new_a = [ 1 if j == i else 0 for j in range(data["var count"]) ]
				if data["A_eq"] is None:
					data["A_eq"] = list()
					data["b_eq"] = list()
				data["A_eq"].append(new_a)
				data["b_eq"].append(lower)
				_iphelper()
				data["A_eq"].pop()
				data["b_eq"].pop()
				if len(data["A_eq"]) == 0:
					data["A_eq"] = None
					data["b_eq"] = None
			elif lower > bounds[0]:
				new_a = [ 1 if j == i else 0 for j in range(data["var count"]) ]
				old_bound = data["bounds"][i][1]
				data["A"].append(new_a)
				data["b"].append(lower)
				data["bounds"][i][1] = lower
				_iphelper()
				data["A"].pop()
				data["b"].pop()
				data["bounds"][i][1] = old_bound
	
			# the upper modulus:
			if upper == bounds[1]:
				new_a = [ 1 if j == i else 0 for j in range(data["var count"]) ]
				if data["A_eq"] is None:
					data["A_eq"] = list()
					data["b_eq"] = list()
				data["A_eq"].append(new_a)
				data["b_eq"].append(upper)
				_iphelper()
				data["A_eq"].pop()
				data["b_eq"].pop()
				if len(data["A_eq"]) == 0:
					data["A_eq"] = None
					data["b_eq"] = None
			elif bounds[1] is None or  upper < bounds[1]:
				new_a = [ -1 if j == i else 0 for j in range(data["var count"]) ]
				old_bound = data["bounds"][i][0]
				data["A"].append(new_a)
				data["b"].append(-upper)
				data["bounds"][i][0] = upper
				_iphelper()
				data["A"].pop()
				data["b"].pop()
				data["bounds"][i][0] = old_bound
	data["num tabs"] -= 1

# must both be 1-d lists/arrays of the same length
def _dot_product(c, x):
	summation = 0
	for i in range(len(c)):
		summation += (c[i] * x[i])
	return summation

# param var_list: a list of the valid variable indices. Orders the indices in order
def _order(var_list):
	global data

# returns a boolean value indicating whether x satisfies its interval constraint
def _constrained(x, idx):
	global data
	constraints = data["constraints"]
	epsilon = data["epsilon"]

	if constraints[idx] == None or constraints[idx] == 0:
		return True

	eps = x - _floor(x, idx)

	return eps <= epsilon

# gets the lower interval bound of the given float
def _floor(x, idx):
	global data
	constraint = data["constraints"][idx]

	if constraint is None or constraint == 0:
		return x

	div = math.floor(x / constraint)
	return div * constraint

def _ceil(x, idx):
	global data

	return _floor(x, idx) + data["constraints"][idx]

def _precision(idx):
	global data
	c = data["constraints"][idx]
	if type(c) is not str:
		return c / 100.
	return 0.01

def _round(x, idx):
	global data
	constraint = data["constraints"][idx]

	floor = _floor(x, idx)
	if (x - floor) < (constraint / 2):
		return floor
	return floor + constraint
