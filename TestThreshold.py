import numpy as np
import statistics
OutputPathModels=['**********/PrimaEsecuzione/',
                  '***********/SecondaEsecuzione/'] #path of folder containing the files of the output of the model in the text phase

#This evaluate the average and the standard deviation of the sensitivity and FPR for
#the output of the network in the test phase. It's used to change the threshold
#and adapt it to each patient.
def main():
    pat=["01","02", "05", "19","21", "23"]
    nSeizure=[7,3, 5, 2, 4,5]
    secondsInterictalInTest=[7500,31500, 10500, 46500, 21000,10500]
    threshold=[0.6,0.8, 0.4, 0.001, 0.3,0.3]
    #totSens=0
    #totFPR=0
    for j in range(0,len(pat)):
        sensResults=[]
        FPRResults=[]
        for k in range(0,2):
            for i in range(0, nSeizure[j]):            
                interPrediction=np.loadtxt(OutputPathModels[k]+"OutputTest"+"/"+"Int_"+pat[j]+"_"+str(i+1)+".csv",delimiter=',')
                preictPrediction=np.loadtxt(OutputPathModels[k]+"OutputTest"+"/"+"Pre_"+pat[j]+"_"+str(i+1)+".csv",delimiter=',')
                
                acc=0#accumulator
                fp=0
                tp=0
                fn=0
                lastTenResult=list()
                
                for el in interPrediction:
                    if(el[1]>threshold[j]):
                        acc=acc+1
                        lastTenResult.append(1)
                    else:
                        lastTenResult.append(0)
                    if(len(lastTenResult)>10):
                        acc=acc-lastTenResult.pop(0)
                    if(acc>=8):
                      fp=fp+1
                      lastTenResult=list()
                      acc=0
                
                lastTenResult=list()
                for el in preictPrediction:
                    if(el[1]>threshold[j]):
                        acc=acc+1
                        lastTenResult.append(1)
                    else:
                        lastTenResult.append(0)
                    if(len(lastTenResult)>10):
                        acc=acc-lastTenResult.pop(0)
                    if(acc>=8):
                      tp=tp+1 
                    else:
                        if(len(lastTenResult)==10):
                           fn=fn+1 
                           
                sensitivity=tp/(tp+fn)*100
                FPR=fp/(secondsInterictalInTest[j]/(60*60))
                sensResults.append(sensitivity)
                FPRResults.append(FPR)
                
        sdSENS=statistics.stdev(sensResults)
        avSENS=statistics.mean(sensResults)
        
        sdFPR=statistics.stdev(FPRResults)
        avFPR=statistics.mean(FPRResults)
        print(pat[j]+"   AVG_Sens= "+str(avSENS)+" +- "+str(sdSENS)+"   AVG_FPR= "+str(avFPR)+" +- "+str(sdFPR))

if __name__ == '__main__':
    main()