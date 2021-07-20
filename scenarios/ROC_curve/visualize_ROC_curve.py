from numpy.lib.npyio import load
import pandas as pd 
import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import os

# --------------- loading the location where we will save data into --------------- #
load_location=__file__[:__file__.rindex("/",0,-25)]
print(load_location)

save_location=__file__[:__file__.rindex("/")]
print(save_location)


colors=['','blue',"green","red","cyan","magenta","#665228","black"]

def createROCCurve():
    for i in range(1,4):
        fpr= np.load(f"{load_location}/scenario{i}/statistical_measures/fpr_ThirdBest.npy")
        tpr= np.load(f"{load_location}/scenario{i}/statistical_measures/tpr_ThirdBest.npy")
        if i==1:
            plt.plot(fpr, tpr, linestyle='--', color=colors[i],label=f"CNN model")
        if i==2:
            plt.plot(fpr, tpr, linestyle='--', color=colors[i],label=f"pt (CNN)")
        if i==3:
            plt.plot(fpr, tpr, linestyle='--', color=colors[i],label=f"pt+Aug (CNN)")    

    fpr= np.load(f"{load_location}/scenario4/statistical_measures/hct/fpr_hctBestModel.npy")
    tpr= np.load(f"{load_location}/scenario4/statistical_measures/hct/tpr_hctBestModel.npy")
    plt.plot(fpr, tpr, linestyle='--', color=colors[4],label=f"hct leave out")

    fpr= np.load(f"{load_location}/scenario4/statistical_measures/hek/fpr_hekBestModel.npy")
    tpr= np.load(f"{load_location}/scenario4/statistical_measures/hek/tpr_hekBestModel.npy")
    plt.plot(fpr, tpr, linestyle='--', color=colors[5],label=f"hek leave out")

    fpr= np.load(f"{load_location}/scenario4/statistical_measures/hela/fpr_helaBestModel.npy")
    tpr= np.load(f"{load_location}/scenario4/statistical_measures/hela/tpr_helaBestModel.npy")
    plt.plot(fpr, tpr, linestyle='--', color=colors[6],label=f"hela leave out")

    fpr= np.load(f"{load_location}/scenario4/statistical_measures/hl/fpr_hlBestModel.npy")
    tpr= np.load(f"{load_location}/scenario4/statistical_measures/hl/tpr_hlBestModel.npy")
    plt.plot(fpr, tpr, linestyle='--', color=colors[7],label=f"hl leave out")

    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(f"{save_location}/roc with scenario 4",dpi=300)
    plt.close()


def createROCCurveWithOutScenario4():
    for i in range(1,4):
        fpr= np.load(f"{load_location}/scenario{i}/statistical_measures/fpr_ThirdBest.npy")
        tpr= np.load(f"{load_location}/scenario{i}/statistical_measures/tpr_ThirdBest.npy")
        if i==1:
            plt.plot(fpr, tpr, linestyle='--', color=colors[i],label=f"CNN model")
        if i==2:
            plt.plot(fpr, tpr, linestyle='--', color=colors[i],label=f"pt (CNN)")
        if i==3:
            plt.plot(fpr, tpr, linestyle='--', color=colors[i],label=f"pt+Aug (CNN)")    

    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(f"{save_location}/roc without scenario 4",dpi=300)
    plt.close()

def createROCCurveWithEnhancments():
    for i in range(1,4):
        fpr= np.load(f"{load_location}/scenario{i}/statistical_measures/fpr_ThirdBest.npy")
        tpr= np.load(f"{load_location}/scenario{i}/statistical_measures/tpr_ThirdBest.npy")
        if i==1:
            plt.plot(fpr, tpr, linestyle='--', color=colors[i],label=f"CNN model")
        if i==2:
            plt.plot(fpr, tpr, linestyle='--', color=colors[i],label=f"pt (CNN)")
        if i==3:
            plt.plot(fpr, tpr, linestyle='--', color=colors[i],label=f"pt+Aug (CNN)")    

    fpr= np.load(f"enhancments/attention_cls/statistical_measures/fpr_ThirdBest.npy")
    tpr= np.load(f"enhancments/attention_cls/statistical_measures/tpr_ThirdBest.npy")
    plt.plot(fpr, tpr, linestyle='--', color=colors[4],label=f"attention pt+Aug (CNN)")

    fpr= np.load(f"enhancments/dropout_cls/statistical_measures/fpr_ThirdBest.npy")
    tpr= np.load(f"enhancments/dropout_cls/statistical_measures/tpr_ThirdBest.npy")
    plt.plot(fpr, tpr, linestyle='--', color=colors[6],label=f"dropout pt+Aug (CNN)")


    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig(f"{save_location}/roc with Enhancments",dpi=300)
    plt.close()



createROCCurve()
createROCCurveWithOutScenario4()
createROCCurveWithEnhancments()