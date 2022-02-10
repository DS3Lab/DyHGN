import matplotlib.pyplot as plt
import numpy as np
import glob

def plot_from_out(name=''):
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
    
    ### Change with the right names ###
    s = ['gmv acc', 'gmv auc', 'binary auc', 'ap']
    #names = ["just head and relation + Linear"]
    names = ["dysat_gcn origin1", "distmult score + LSTM1","distmult score + LSTM2", "just head and relation + LSTM1", "just head and relation + LSTM2", "head and relation + Linear1", "head and relation + Linear2"]
    names = ["origin2", "distmult score + LSTM2", "just head and relation + LSTM2", "head and relation + Linear2"]

    files = glob.glob('dysat*.out')
    print(sorted(files))
    #####################################

    for file in sorted(files):
        epoch_stg0, gmv_acc_stg0, gmv_auc_stg0, binary_auc_stg0, binary_ap_stg0 = [], [], [], [], []
        epoch_stg1, gmv_acc_stg1, gmv_auc_stg1, binary_auc_stg1, binary_ap_stg1 = [], [], [], [], []
        count = 0
        
        with open(file, encoding="utf8") as f:
            for line in f:
                count+=1
                if line.startswith("Name") and count >500:
                    break
                    """
                    metrics_stg0 = [gmv_acc_stg0, gmv_auc_stg0, binary_auc_stg0, binary_ap_stg0]        
                    metrics_stg1 = [gmv_acc_stg1, gmv_auc_stg1, binary_auc_stg1, binary_ap_stg1]  
                    for i in range(4):
                        ax[0, i].plot(epoch_stg0, metrics_stg0[i], marker='.', linewidth=1, markersize=1)
                        ax[0, i].set_title(s[i])
                    for i in range(4):
                        ax[1, i].plot(epoch_stg1, metrics_stg1[i], marker='.', linewidth=1, markersize=1)
                        ax[1, i].set_title(s[i])
                    
                    epoch_stg0, gmv_acc_stg0, gmv_auc_stg0, binary_auc_stg0, binary_ap_stg0 = [], [], [], [], []
                    epoch_stg1, gmv_acc_stg1, gmv_auc_stg1, binary_auc_stg1, binary_ap_stg1 = [], [], [], [], []
                    """
                if line.startswith("train-stg"):
                    words = line.split()
                    stg = int(words[0][9])
                    if stg==0:
                        
                        try:
                            epoch = int(words[2])
                        except:
                            continue
                        
                        if len(epoch_stg0)>1 and epoch_stg0[-1]==epoch:
                            continue
                        else:
                            epoch_stg0.append(epoch)
                            gmv_acc_stg0.append(float(words[6]))
                            gmv_auc_stg0.append(float(words[9]))
                            binary_auc_stg0.append(float(words[12]))
                            binary_ap_stg0.append(float(words[14][:-1]))
                            
                    else:
                        try:
                            epoch = int(words[2])
                        except:
                            continue
                        
                        if len(epoch_stg1)>1 and epoch_stg1[-1]==epoch:
                            continue
                        else:
                            epoch_stg1.append(epoch)
                            gmv_acc_stg1.append(float(words[6]))
                            gmv_auc_stg1.append(float(words[9]))
                            binary_auc_stg1.append(float(words[12]))
                            binary_ap_stg1.append(float(words[14][:-1]))
                            
        metrics_stg0 = [gmv_acc_stg0, gmv_auc_stg0, binary_auc_stg0, binary_ap_stg0]        
        metrics_stg1 = [gmv_acc_stg1, gmv_auc_stg1, binary_auc_stg1, binary_ap_stg1]  
        for i in range(4):
            ax[0, i].plot(epoch_stg0, metrics_stg0[i], marker='.', linewidth=1, markersize=1)
            ax[0, i].set_title("stage 0 - " + s[i])
        for i in range(4):
            ax[1, i].plot(epoch_stg1, metrics_stg1[i], marker='.', linewidth=1, markersize=1)
            ax[1, i].set_title("stage 1 - " + s[i])
    
    fig.legend(names, loc=3)
    fig.savefig('results' + name + '.png', dpi=200)

def plot_from_txt(name=''):
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
    s = ['gmv acc', 'gmv auc', 'binary auc', 'ap']
    ### Change with the right names ###
    files = glob.glob('results_hgt*.txt')
    names = ["hgt conv layer"]
    ###################################
    for file in sorted(files):
        epoch_stg0, gmv_acc_stg0, gmv_auc_stg0, binary_auc_stg0, binary_ap_stg0 = [], [], [], [], []
        epoch_stg1, gmv_acc_stg1, gmv_auc_stg1, binary_auc_stg1, binary_ap_stg1 = [], [], [], [], []
        count = 0
        
        with open(file, encoding="utf8") as f:
            for line in f:
                if line.startswith("train-stg"):
                    words = line.split()
                    stg = int(words[0][9])
                    if stg==0:
                        
                        try:
                            epoch = int(words[2])
                        except:
                            continue
                        
                        epoch_stg0.append(epoch)
                        gmv_acc_stg0.append(float(words[6]))
                        gmv_auc_stg0.append(float(words[9]))
                        binary_auc_stg0.append(float(words[12]))
                        binary_ap_stg0.append(float(words[14]))
                            
                    else:
                        try:
                            epoch = int(words[2])
                        except:
                            continue
                        
                        epoch_stg1.append(epoch)
                        gmv_acc_stg1.append(float(words[6]))
                        gmv_auc_stg1.append(float(words[9]))
                        binary_auc_stg1.append(float(words[12]))
                        binary_ap_stg1.append(float(words[14]))
                            
        metrics_stg0 = [gmv_acc_stg0, gmv_auc_stg0, binary_auc_stg0, binary_ap_stg0]        
        metrics_stg1 = [gmv_acc_stg1, gmv_auc_stg1, binary_auc_stg1, binary_ap_stg1]  
        for i in range(4):
            ax[0, i].plot(epoch_stg0, metrics_stg0[i], marker='.', linewidth=1, markersize=1)
            ax[0, i].set_title("stage 0 - " + s[i])
        for i in range(4):
            ax[1, i].plot(epoch_stg1, metrics_stg1[i], marker='.', linewidth=1, markersize=1)
            ax[1, i].set_title("stage 1 - " + s[i])
    
    fig.legend(names, loc=3)
    fig.savefig('results' + name + '.png', dpi=200)

def rescue_line(file):
    with open(file, 'r+', encoding="utf8") as f:
        lines = f.readlines()
        last_line = lines[-1]
        lines = lines[:-1]
        words = last_line.split()
        for s in range(0,len(words),15):
            lines.append(' '.join(words[s:s+15])+'\n')
        print(lines)
        f.seek(0)
        for line in lines:
            f.write(line)
        f.truncate()
        f.close()
                    

if __name__ == "__main__":
    plot_from_txt("hgt") #change name here if needed
    