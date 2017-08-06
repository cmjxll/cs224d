import numpy as np
#已知cita的情形下，求出每个观测softmax值。
def ql_softmax(X):
	"""
		softmax 函数
	"""
	X_m=np.exp(np.max(X))
	X_s=np.sum(np.exp(X))
	X=X_m/X_s
	#print(X)
	return(X)
