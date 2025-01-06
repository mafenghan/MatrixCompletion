#!/usr/bin/env python
#coding:utf-8
import vasp
import pandas as pd
import numpy as np
from scipy.linalg import norm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize,basinhopping
import time
import functools
from pylanczos.pylanczos import PyLanczos


PT = pd.read_csv('/home10/msc/bin/PeriodicTable.csv',index_col = 0)

def bb_optimize(func, grad, x0, tol=5e-5, max_iter=200):
	print('\nStart Barzilai-Borwein descent optimization\n')
	x = x0
	g = grad(x)
	iter_count = 0
	while np.linalg.norm(g) > tol and iter_count < max_iter:
		if iter_count == 0:
			alpha = 1.0
		else:
			s = x - x_old
			y = g - g_old
			alpha = np.dot(s, s) / np.dot(s, y)
		x_old = x
		g_old = g

		x = x - alpha * g
		g = grad(x)
		iter_count += 1
		print(f"Iteration %d: f(x) = %.6f, g(x) =%12.6e, alpha = %12.6e" %(iter_count,func(x),np.linalg.norm(g),alpha))
	return x


def cal_D2(P):
	n = P.shape[0]
	N = np.dot(P,P.T)
	diag = np.diag(N).reshape(-1,1)
	c1 = np.array([1]*n).reshape(-1,1)
	X = np.dot(c1,diag.T) + np.dot(diag,c1.T) - 2 * N
	return X

def cal_dist_from_PPT(data):
	x = data[0]
	y = data[1]
	return x[y[0],y[0]] + x[y[1],y[1]] - x[y[0],y[1]] - x[y[1],y[0]]

def cal_dist_jac_from_PPT(data):
	X = data[0]
	ix = data[1]
	Y = data[2]
	X[ix,ix+1:] = -(Y[ix,ix+1:] + Y[ix+1:,ix])
	X[ix+1:,ix] = X[ix,ix+1:]
	X[ix,ix] = np.sum(Y[ix,:] + Y[:,ix])
	return X

def cal_proj(omega,x,Ncore = 4):
#	data = [[x,y.astype(int)] for y in omega]
#	proj = vasp.ParaRun(cal_dist_from_PPT,data,Ncore = Ncore)
	proj = []
	for y in omega:
		y = y.astype(int)
		proj.append(x[y[0],y[0]] + x[y[1],y[1]] - x[y[0],y[1]] - x[y[1],y[0]])
	return np.array(proj)

def cal_proj_adj_mat(x,omega,n,Ncore = 4):
	row,col = list(omega[:,0]),list(omega[:,1])
	Y = coo_matrix((x,(row,col)),shape = (n,n)).toarray()
	X = np.zeros((n,n))
	for ix in range(n):
		X[ix,ix+1:] = -(Y[ix,ix+1:] + Y[ix+1:,ix])
		X[ix+1:,ix] = X[ix,ix+1:]
		X[ix,ix] = np.sum(Y[ix,:] + Y[:,ix])
	return X
#	data = [[X,i,Y] for i in range(n)]
#	total = vasp.ParaRun(cal_dist_jac_from_PPT,data,Ncore = Ncore)
#	total = functools.reduce(lambda a,b:a+b, total)
#	return np.array(total)

def cal_MDS(dist,n_dims):
	'''
	cal multi-dimensional scaling
	'''
	n = dist.shape[0]
	T1 = np.ones((n,n))*np.sum(dist)/n**2
	T2 = np.sum(dist,axis = 1,keepdims = True)/n
	T3 = np.sum(dist,axis = 0,keepdims = True)/n
	B = -(T1 - T2 - T3 + dist)/2

	engine = PyLanczos(B,True, n_dims+1)
	eig_val,eig_vector = engine.run()
#	eig_val,eig_vector = eigsh(B,n_dims+1)
	index = np.argsort(-eig_val)[:n_dims]
	picked_eig_val = eig_val[index].real
	print('Picked Eigen-values: %s' %-np.sort(-eig_val.real)[:n_dims])
	print('Fourth Eigen-values: %s' %-np.sort(-eig_val.real)[n_dims], flush = True)
	picked_eig_vector = eig_vector[:,index].real
	return picked_eig_vector * picked_eig_val**(1/2)

class OptimizerWithDynamicParams():
	def __init__(self,b, omega, n,r,q,lambda_,threshold_rmse,threshold_maxd,Ncore):
		self.b = b
		self.omega = omega
		self.n = n
		self.r = r
		self.q = q
		self.lambda_ = lambda_
		self.threshold_rmse = threshold_rmse
		self.threshold_maxd = threshold_maxd
		self.start_time = None
		self.results = []
		self.best_x = None
		self.best_f = float('inf')
		self.Ncore = Ncore

	def L_P(self, P):
		P_2 = P.reshape((self.n, self.q))
		
		d_2 = np.dot(P_2,P_2.T)
		a_2 = cal_proj(self.omega,d_2,self.Ncore) - self.b


		L = np.sum(np.diag(d_2)) + self.r /2 * norm(a_2)**2

		a_p = np.sum(self.lambda_ * a_2)
		L += a_p

		return L 

	def grad(self, P):
		P_2 = P.reshape((self.n, self.q))

		d_2 = np.dot(P_2,P_2.T)
		a_2 = cal_proj(self.omega,d_2,self.Ncore)

		l = a_2 - self.b
		b_2 = cal_proj_adj_mat(l,self.omega,self.n,self.Ncore)

		g_2 = 2 * P_2 + 2 * self.r * np.dot(b_2, P_2)
		
		c_2 = cal_proj_adj_mat(self.lambda_, self.omega,self.n,self.Ncore)
		g_2 += 2 * np.dot(c_2, P_2)
		
		return g_2.flatten()
	
	def print_fun(self, x):
		P_2 = x.reshape((self.n, self.q))
		d_mat = np.dot(P_2,P_2.T)
		dd = np.sqrt(cal_proj(self.omega,d_mat,self.Ncore)) - np.sqrt(self.b)
		rmse = norm(dd-np.mean(dd),2)/self.n
		maxd = max(abs(dd))
		return rmse, maxd
	def __call__(self, x, f, accept):
		P_2 = x.reshape((self.n, self.q))
		d_mat = np.dot(P_2,P_2.T)
		aa = cal_proj(self.omega,d_mat,self.Ncore)
		dd = np.sqrt(aa) - np.sqrt(self.b)
		rmse = norm(dd-np.mean(dd),2)/self.n
		maxd = max(abs(dd))
		self.lambda_ = self.lambda_ + self.r * (aa - self.b)
		Ek = self.r/2 * norm(aa - self.b)**2/self.n
#		print('\nlambda: %s' %self.lambda_)
		print('\nr/2*||Ax-b||_2^2 / n = %12.5f' %Ek)
		print('Maximum error of distance: %12.5f ' %maxd)
		print('         RMSE of distance: %12.5f \n' %rmse)
		if self.start_time is not None:
			elapsed_time = time.time() - self.start_time
			print(f"Iteration time: %.4f seconds\n" %elapsed_time, flush = True)

		self.start_time = time.time()
		self.best_x = x
		return (rmse < self.threshold_rmse) & (maxd < self.threshold_maxd)

def initializeP(b,omega,q,n):
	row,col = omega[:,0],omega[:,1]
	D = coo_matrix((b,(row,col)),shape = (n,n)).toarray()
	ns = len(np.where(D[:,0] > 0)[0])
	for i in range(n):
		for j in range(i+1,n,1):
			if D[i,j] == 0:
				D[i,j] = np.random.uniform(np.max(b),np.max(b)*n/ns,1)
				D[j,i] = D[i,j]
	return cal_MDS(D,q)

def re_initializeP(X,b,omega,q):
	row,col = omega[:,0],omega[:,1]
	for i in range(len(b)):
		X[row[i],col[i]] = b[i]
	return cal_MDS(X,q)

def dropDistviaNumber(dist,ns):
	n = dist.shape[0]
	for i in range(n):
		a = np.argsort(dist[:,i])[ns:]
		for j in a:
			dist[j,i] = np.nan
	return dist

def dropDistviaCut(dist,rcut):
	n = dist.shape[0]
	mask = dist > rcut
	dist[mask] = np.nan
	return dist

def initializeParameters(dist,q = 3,r = 1.0):
	n = dist.shape[0]
	row_i, col_j,x_x = [],[],[]
	for i in range(n):
		for j in range(n):
			if np.isnan(dist[i,j]) == False:
				row_i.append(i)
				col_j.append(j)
				x_x.append(dist[i,j])
				row_i.append(j)
				col_j.append(i)
				x_x.append(dist[i,j])
	
	df = pd.DataFrame({'row_i': row_i, 'col_j': col_j, 'x_x': x_x})
	df.drop_duplicates(inplace = True,ignore_index = True)
	df = df[(df['row_i'] - df['col_j'] != 0) & (df['x_x'] > 0)]
	omega = df.loc[:,['row_i', 'col_j']].to_numpy()
	b = df.loc[:,'x_x'].to_numpy()
	lambda_ = [0.0] * len(b)
	print('Values of q: %d (Rank number)' %q)
	print('Values of n: %d (Atom number)' %n)
	print('Values of r: %.3f (Penalty weight)' %r)
	print('Length of b: %d (Input distance value)' %(len(b)))
	print('Length of omega: %d (Input distance i,j index)' %(len(omega)))
	return b, omega, n,r,q,lambda_

def MatrixReconstruction(dist,ns = None,rcut=None,threshold_rmse = 0.1,threshold_maxd = 0.005,maxiter = 10,abc = None, Ncore = 4):
	cycle = 0
	method = 'L-BFGS-B'
	
	print('',flush = True)
	print('Exact Reconstruction of Euclidean Distance Using Low-Rank Matrix Completion')
	print('                                                        writen by Sicong Ma\n')
	starttime = time.time()
	print('Initialize Parameters!\n',flush = True)
	if ns != None and rcut == None:
		dist = dropDistviaNumber(dist,ns = ns)
		print('Values of ns: %d (Neighbor number)' %ns)
	elif ns == None and rcut != None:
		dist = dropDistviaCut(dist,rcut = rcut)
		print('Values of rcut: %.2f (distance cut)' %rcut)
	elif ns != None and rcut != None:
		print('Error!!! Please either provide "ns" value or "rcut" value. We do not support that both values are provided.')
		return

	b, omega, n,r,q,lambda_ = initializeParameters(dist)
	optimizer = OptimizerWithDynamicParams(b, omega, n,r,q,lambda_,threshold_rmse,threshold_maxd * 4,Ncore)
	print('Convergence criterion: Maximum error of distance < %.4f; RMSE of distance < %.4f' %(threshold_maxd,threshold_rmse))
	print('\nInput distance information:\n',flush = True)
	print(dist)
	print('\nInitialize P via multi-dimensional scaling\n')
	p_t = initializeP(b,omega,q,n)
	X = np.dot(p_t,p_t.T)
	print('\nStart basin-hopping global optimization, followed by local optimization with %s method!\n' %method, flush = True)

	while True:
		print('Cycle: %d\n' %cycle)

		minimizer_kwargs = {'method': method, 'jac': optimizer.grad, 'options': {'disp':False}}

		p_t = re_initializeP(X,b,omega,q)

		results = basinhopping(optimizer.L_P, x0 = p_t, niter=30, stepsize = n/10, minimizer_kwargs = minimizer_kwargs,callback=optimizer,disp = True,niter_success = 5)
		results.x = optimizer.best_x
		p_t = results.x.reshape((n,q))
		X = np.dot(p_t,p_t.T)
		rmse,maxd = optimizer.print_fun(results.x)
		if maxd < 0.4:
#			results = minimize(optimizer.L_P,x0 = results.x,method = method, jac = optimizer.grad, options = {'disp': True})
			x = bb_optimize(optimizer.L_P,optimizer.grad,results.x,tol=5E-5,max_iter = 500)
			p_t = x.reshape((n,q))
			X = np.dot(p_t,p_t.T)
			rmse,maxd = optimizer.print_fun(x)
		print('Final rmse, maxd: %.5f %.5f\n' %(rmse,maxd))
		if rmse < threshold_rmse and maxd < threshold_maxd:
			break
		
		cycle += 1
		if cycle > maxiter:
			print('Out of the maximum cycle!')
			break

	Ek = r/2 * norm(cal_proj(omega,X,Ncore) - b)**2/n
	Etot = optimizer.L_P(p_t)

#	p_t = np.hstack([p_t,np.array([[0],[0],[0],[0]])])

	print('#######################################################################################',flush = True)
	print('Input distance Matrix (D^2):')
	print(dist)
	print('\nOutput estimated coordinate:\n')
	print('  %12s %12s %12s' %('x','y','z'))
	for i in range(p_t.shape[0]):
		s = ' '
		for j in range(q):
			s += ' %12.6f' %p_t[i,j]
		print(s)
	print('\n Input distance number: %d' %(len(b)))
	print('Output distance number: %d' % (n*n))
	print('Matrix reconstruction ratio: %.2f' % (len(b)/(n**2)))
	print('Maximum input distance: %.4f' %np.sqrt(np.nanmax(dist)))
	print('A(x) - b = : %12.8e' %Ek)
	print('L(P,r,lambda): %12.8e' %Etot)
	print('Maximum error of distance: %12.5f ' %maxd)
	print('         RMSE of distance: %12.5f \n' %rmse)
	endtime = time.time() - starttime
	h = endtime //3600
	m = endtime % 3600 //60
	s = endtime %60
	print('Totol time: %d hours %d minutes %d seconds' %(h,m,s))
	print('End!!!')

	return p_t

def Positionrecovery(dist,ns = None,rcut = None):
	if ns != None and rcut == None:
		dist = dropDistviaNumber(dist,ns = ns)
		print('Values of ns: %d (Neighbor number)' %ns)
	elif ns == None and rcut != None:
		dist = dropDistviaCut(dist,rcut = rcut)
		print('Values of rcut: %.2f (distance cut)' %rcut)
	b, omega, n,r,q,lambda_ = initializeParameters(dist)
	df = pd.DataFrame(omega,columns = ['i','j'])
	df['d'] = b
	count = df['d'].value_counts()
	nn = 0
	for i in count.index:
		if count[i] == 2:
			aa = df[df['d'] == i]
			df.loc[aa.index[0],'n'] = df.loc[aa.index[1],'i']
			df.loc[aa.index[1],'n'] = df.loc[aa.index[0],'i']
		else:
			nn += 1
	print(nn)
#	aa = df[df['n'].isnull()]
#	for i,v in df.groupby('i'):
#		print(i, v.shape)

def getarc(coordinate,abc,atomtype,atomnumber,filename = '1.arc'):
#	lattice = vasp.abc2latt(abc)
#	car = vasp.dir2car(lattice,coordinate)
	arc = pd.DataFrame(coordinate,columns = ['x','y','z'])
	arc['number'] = atomnumber
	arc['type'] = atomtype
	arc = vasp.inCell(arc,abc)
	s = vasp.arcformat(arc,abc)
	ff = open(filename,'w')
	print(s,file=ff)
	ff.close()

with open('300.arc') as f:	lines = f.readlines()
arc = vasp.arcread(lines)
arc.arc = vasp.inCell(arc.arc,arc.abc)

#P = np.array([[0,0],[2,0],[2,2],[2+np.sqrt(3),1]])
P = np.array(arc.arc.loc[:,['x','y','z']])
X = cal_D2(P)

P_t = MatrixReconstruction(X,ns = 40,rcut = None,threshold_rmse = 0.005,threshold_maxd = 0.10,maxiter = 10,abc = arc.abc)
P_t = P_t - np.min(P_t,axis = 0)
getarc(P_t,[50,50,50,90,90,90],arc.arc.loc[:,'type'],arc.arc.loc[:,'number'],filename = '2.arc')
#Positionrecovery(X,ns = None,rcut = 34.96)
