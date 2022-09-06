import sys
import numpy as np
import spkmeansmodule

MAX_ITER = 300
EPSILON = 0

def spk(k, inputFile):
    mtx = spkmeansmodule.partialSpk(k, inputFile)
    k = len(mtx[0])
    return kmeanspp(k, MAX_ITER, EPSILON, mtx)

def printMatrix(mtx:list):
    for i in range(len(mtx)):
        for j in range(len(mtx[i])):
            print(format("%.4f" %mtx[i][j]), end="")
            if(j != len(mtx[i])-1):
                print(",", end="")
        print("")

def kmeanspp(k:int, max_iter:int, eps:int, mtx):
    np.random.seed(0)
    n = len(mtx)
    randI = np.random.choice(n)
    newMu = mtx[randI]
    mu = [newMu]
    choices = [randI]
    D = [0 for i in range(n)]
    P = [0 for i in range(n)]
    i = 1
    while i < k:
        for l in range(n):
            minDist = float("inf") #find min in array algorithm
            for j in range(i):
                curDist = dist(mtx[l], mu[j])
                if curDist < minDist:
                    minDist = curDist
            D[l] = minDist
        sumD = sum(D)
        for l in range(n):
            P[l] = D[l]/sumD # probabilty calculation as instructed
        randI = np.random.choice(a=range(n), p=P)
        newMu = mtx[randI]
        choices.append(randI)
        mu.append(newMu)
        i+=1
    print(str(choices)[1:-1].replace(" ", "")) # print the chosen indexes
    #mu = [list(float(mu[i][j]) for j in range(k)) for i in range(k)]
    return spkmeansmodule.kmeans(k, k, n, max_iter, EPSILON, mtx, mu) # calling the c function

def dist(x:list, u:list) -> float:
    return (sum((x[i] - u[i])**2 for i in range(len(x))))

if __name__ == "__main__":
    k = -1
    mode = ""
    inputFile = ""
    try: # trying to open the files and convert the input
        k = int(sys.argv[1])
        mode = sys.argv[2]
        inputFile = sys.argv[3]
    except Exception as e:
        print("Invalid Input!")
        exit()

    try:
        if mode == "spk":
            printMatrix(spk(k, inputFile))
        elif mode == "wam":
            printMatrix(spkmeansmodule.wam(inputFile))
        elif mode == "ddg":
            printMatrix(spkmeansmodule.ddg(inputFile))
        elif mode == "lnorm":
            printMatrix(spkmeansmodule.lnorm(inputFile))
        elif mode == "jacobi":
            printMatrix(spkmeansmodule.jacobi(inputFile))
        else:
            raise Exception()
    except Exception as e:
        print("An Error Has Occured")
    finally:
        exit()