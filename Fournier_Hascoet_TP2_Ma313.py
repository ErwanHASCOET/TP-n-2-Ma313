import numpy as np
import time
import matplotlib.pyplot as plt
import math as ma



def DecompositionGS (A):
	n,m=A.shape

	Q,R=np.zeros((n,m)),np.zeros((n,m))

	R[0,0]=np.linalg.norm(A[:,0])
	Q[:,0]=A[:,0]/R[0,0]

	for i in range(1,n):
		for j in range(i):
			R[j,i]=np.dot(A[:,i],Q[:,j])

		somme =0
		for t in range(i):
			somme += R[t,i]*Q[:,t]
		W=A[:,i]- somme

		R[i,i]=np.linalg.norm(W)

		Q[:,i] = W/R[i,i]

	return Q,R,np.dot(Q,R)


#print(DecompositionGS(A))

def Resosup(A,b):
	n,m=A.shape
	x=np.zeros(n)
	for i in range(n-1, -1,-1):
		S = np.dot(A[i,i+1:],(x[i+1:]))
		x[i]=(1/A[i,i])*(b[i]-S)
	return x

def ResolGS(A,b):
    Q,R,P=DecompositionGS(A)
    Y = np.dot(Q.T,b)
    X=Resosup(R,Y)
    X = np.asarray(X).reshape(n,1)
    return X

#print(ResolGS(A,b))

def Cholesky(A):
    n,m = np.shape(A) 
    L = np.zeros((n,m))

    for k in range(m):
        S = 0
        for j in range(k):
            S = S+L[k,j]**2

        else :
            L[k][k] = (A[k][k]-S)**(1/2)

        for i in range(k+1,n):

            S2 = 0 
            for j in range(k):
                S2 = S2+L[i][j]*L[k][j]

            L[i][k] = (A[i][k]-S2)/L[k][k]


    return L

M= np.random.randint(low=1,high=100,size=(3,3))

while np.linalg.det(M)==0 :
    M= np.random.randint(low=1,high=100,size=(3,3))

A = M.dot(M.T)


#Partie 2#

def ResolutionsystTriinf(Taug):                            
    n,m=Taug.shape
    X=np.zeros(n)
    for i in range(0,n):
        a=0
        for j in range(0,i):
            a=a+Taug[i,j]*X[j]
        X[i]=(Taug[i,n]-a)/Taug[i,i]
    return(X)

def ResolutionsystTrisup(Taug):                           
    n,m=Taug.shape
    X=np.zeros(n)
    for i in range(n-1,-1,-1):
        S=0
        for j in range(i+1,n):
            S=S+Taug[i,j]*X[j]
        X[i]=(Taug[i,n]-S)/Taug[i,i]
    return(X)



def ResolCholesky(A,B):
    L=Cholesky(A)
    Taug= np.hstack((L,B))          ###Hstack --> empile les matrices horizontalement (en colonnes)
    Y=ResolutionsystTriinf(Taug)
    Y=Y[:,np.newaxis]               ###newaxis --> augmente d'une dimension, la dimension de la matrice
    LT=np.transpose(L)              ###transpose --> fait la transposée de L
    Baug= np.hstack((LT,Y))         ###Hstack --> empile les matrices horizontalement (en colonnes)
    X=ResolutionsystTrisup(Baug)
    return X
#print("La solution du système AX=B avec la méthode de cholesky, en sachant que A= \n", A,"\n", "B=\n",B,"\n","est:\n", ResolCholesky(A,B))

def ResolCholeskypy(A,B):
    L=np.linalg.cholesky(A)
    Taug= np.hstack((L,B))          ###Hstack --> empile les matrices horizontalement (en colonnes)
    Y=ResolutionsystTriinf(Taug)
    Y=Y[:,np.newaxis]               ###newaxis --> augmente d'une dimension, la dimension de la matrice
    LT=np.transpose(L)              ###transpose --> fait la transposée de L
    Baug= np.hstack((LT,Y))         ###Hstack --> empile les matrices horizontalement (en colonnes)
    X=ResolutionsystTrisup(Baug)
    return X


def ReductionGauss(Aaug):
    n,m = np.shape(Aaug)
    for k in range (0,n-1):
        for i in range (k+1,n):
            gik = Aaug[i,k]/Aaug[k,k]
            Aaug[i,:] = Aaug[i,:] - gik*Aaug[k,:]
    
          
    return Aaug

#print(ReductionGauss(np.array([[2,5,6,7],[4,11,9,12],[-2,-8,7,3]])))



def ResolutionSysTriSup (Taug):
    n,m = np.shape(Taug)
    x = np.zeros(n)
    for i in range (n-1,-1,-1):
        somme = 0
        for k in range (i, n):
            somme = somme + x[k]*Taug[i,k]
            x[i] = (Taug[i,-1]-somme)/(Taug[i,i])
            
    return x
    
#print(ResolutionSysTriSup(np.array([[2,5,6,7],[0,1,-3,-2],[0,0,4,4]])))

         
def Gauss (A,B):
   Aaug = np.concatenate( (A, B.T), axis = 1)
   Taug = ReductionGauss(Aaug)
   S = ResolutionSysTriSup(Taug)
   return S

def DecompositionLU(A):
    n,m = np.shape(A)
    U=np.zeros((n,m))
    L=np.eye(n)
    for i in range(0,n):
        for k in range(i+1,n):
            piv = A[i][i]
            piv = A[k][i]/piv
            L[k][i]=piv
            for j in range(i,n):
                A[k][j]=A[k][j]-(piv*A[i][j])
    U=A
    return L,U




def ResolutionLU(A,B):
    n,m = np.shape(A)
    X=np.zeros(n)
    L,U=DecompositionLU(A)
    #print("L=\n", L)
    #print("U=\n", U)
    
    Y=ResolutionsystTriinf(np.concatenate((L,B),axis =1))
    Y=np.asarray(Y).reshape(n,1)
    X=ResolutionSysTriSup(np.concatenate((U,Y),axis =1))
    X = np.asarray(X).reshape(n,1)
    
    return (X)

TempsQR = []
IndiceQR = []
TempsCh = []
IndiceCh = []
TempsG = []
IndiceG = []
TempsLU = []
IndiceLU = []
TempsChV = []
IndiceChV = []
ErreurQR = []
ErreurCh = []
ErreurG = []
ErreurLU = []
ErreurChV = []

for n in range (2,500,50):
    try:
        a= np.random.randint(low=-10,high=10,size=(n,n))
        while np.linalg.det(a)==0 :
            a = np.random.randint(low=-10,high=10,size=(n,n))
        A = a.dot(a.T)
        b= np.random.randint(low=-100,high=100,size=(n,1))
        A = np.array(A, dtype = float)
        b = np.array(b, dtype = float)
        #t3 = time.time()
        #ResolCholesky(A,b)
        #t4 = time.time()
        #t5 = t4-t3
        X4 = ResolCholesky(A,b)
        err4 = np.linalg.norm(A.dot(X4)-b)
        ErreurCh.append(err4)
        #TempsCh.append(ma.log(t5))
        IndiceCh.append(n)
    except:
        print('')
      
for n in range(2,500,50):
    try:
        a= np.random.randint(low = -100,high = 100,size = (n,n))
        while np.linalg.det(a)==0 :
            a= np.random.randint(low=-100,high=100,size=(n,n))
        A = a.dot(a.T)
        b= np.random.randint(low=-100,high=100,size=(n,1))
        A=np.array(A, dtype=float)
        b=np.array(b, dtype=float)
        T1 = time.time()
        ResolCholeskypy (A,b)
        T2 = time.time()
        T3 = T2 - T1
        #X3 =ResolCholeskypy (A,b)
        #err3 = np.linalg.norm(A.dot(X3)-b)
        #ErreurChV.append(err3)
        TempsChV.append(ma.log(T3))
        IndiceChV.append(ma.log(n))
    except:
        print('')
  
for n in range (2,500,50):
    try:
        A = np.random.randint(low=-100,high=100,size=(n,n))
        b= np.random.randint(low=-100,high=100,size=(n,1))
        A = np.array(A, dtype = float)
        b = np.array(b, dtype = float)
        #t1 = time.time()
        #ResolGS(A,b)
        #t2 = time.time()
        #t = t2-t1
        #t9 = time.time()
        #ResolutionLU(A,b)
        #t10 = time.time()
        #t11 = t10-t9
        X1 = ResolGS(A,b)
        err1 = np.linalg.norm(A.dot(X1)-b)
        ErreurQR.append(err1)
        #Acop = A.copy()
        #X2 = ResolutionLU(A,b)
        #err2 = np.linalg.norm(Acop.dot(X2)-b)
        #ErreurLU.append(err2)
        #TempsLU.append(ma.log(t11))
        #IndiceLU.append(n)
        #TempsQR.append(ma.log(t))
        IndiceQR.append(n)
    except:
        print('')

      
for n in range (2,500,50):
    try:
        A = np.random.randint(low=-100,high=100,size=(n,n))
        b= np.random.randint(low=-100,high=100,size=(1,n))
        A = np.array(A, dtype = float)
        b = np.array(b, dtype = float)
        #t6 = time.time()
        #Gauss (A,b)
        #t7 = time.time()
        #t8 = t7-t6
        X = Gauss(A,b)
        err = np.linalg.norm(A.dot(X)-b)
        ErreurG.append(err)
        #TempsG.append(ma.log(t8))
        IndiceG.append(n)
    except:
        print('')
        
x1 = IndiceQR
y1 = TempsQR
x2 = IndiceCh
y2 = TempsCh 
x3 = IndiceG
y3 = TempsG
x4 = IndiceLU
y4 = TempsLU
x5 = IndiceChV
y5 = TempsChV
y6 = ErreurQR
y7 = ErreurCh
y8 = ErreurG
y9 = ErreurLU
#y10 = ErreurChV
plt.plot(x1,y6, color = 'green',label = "Méthode QR")
#plt.plot(x2,y7, color = 'red',label ="Cholesky")
#plt.plot(x3,y8, color = 'blue',label="Gauss")
#plt.plot(x4,y9, color = 'yellow',label="Méthode LU")
#plt.plot(x5,y5, color = 'magenta',label="Numpy Cholesky")
plt.title("Erreur en fonction de n")
plt.legend()
plt.xlabel("Indice")
plt.ylabel("Erreur")

plt.show()
