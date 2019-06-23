import math

class Knn(object):

	def __init__(self,k=3):
		self.k = k

	def distEuclid(self,x1,x2):
		s = 0.0
		for i in range(len(x1)):
			s += math.sqrt((float(x1[i]) - float(x2[i])) ** 2)
		return s

	def fit(self,x,y):
		self.x = x
		self.y = y

	def predict(self,teste):
		dist = []
		for i in range(len(self.x)):
			dist.append(self.distEuclid(self.x[i],teste))
		result = []
		for i in range(self.k):
			result.append( self.y[dist.index(min(dist))] )
			dist.pop(dist.index(min(dist)))
		return max(set(result), key=result.count)
