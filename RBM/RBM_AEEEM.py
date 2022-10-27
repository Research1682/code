#-*- coding: utf-8 -*-
import numpy as np
import csv
from sklearn.neural_network import BernoulliRBM
import sys
import gc
sys.path.append("./../Util/")

import preparation
import clustering
import evaluation


PARA = sys.argv

projectName = ['eclipse','equinox','lucene','mylyn','pde']
metricsName = ['bug-metrics','change-metrics','complexity-code-change','single-version-ck-oo','churn','exp-churn','lin-churn','log-churn','weighted-churn','ent','exp-ent','lin-ent','log-ent','weighted-ent']


def data_extend(filename,p_name,times):
    train_data = []
    train_target = []

    filename2 = "./../../data/AEEEM/{0}/churn/{1}/test_times{2}.csv"
    filename3 = "./../../data/AEEEM/{0}/entropy/{1}/test_times{2}.csv"

    f1 = open(filename.format(p_name,metricsName[0],times),'r')
    f2 = open(filename.format(p_name,metricsName[1],times),'r')
    f3 = open(filename.format(p_name,metricsName[2],times),'r')
    f4 = open(filename.format(p_name,metricsName[3],times),'r')
    f5 = open(filename2.format(p_name,metricsName[4],times),'r')
    f6 = open(filename2.format(p_name,metricsName[5],times),'r')
    f7 = open(filename2.format(p_name,metricsName[6],times),'r')
    f8 = open(filename2.format(p_name,metricsName[7],times),'r')
    f9 = open(filename2.format(p_name,metricsName[8],times),'r')
    f10 = open(filename3.format(p_name,metricsName[9],times),'r')
    f11 = open(filename3.format(p_name,metricsName[10],times),'r')
    f12 = open(filename3.format(p_name,metricsName[11],times),'r')
    f13 = open(filename3.format(p_name,metricsName[12],times),'r')
    f14 = open(filename3.format(p_name,metricsName[13],times),'r')
    dataReader1 = csv.reader(f1)
    dataReader2 = csv.reader(f2)
    dataReader3 = csv.reader(f3)
    dataReader4 = csv.reader(f4)
    dataReader5 = csv.reader(f5)
    dataReader6 = csv.reader(f6)
    dataReader7 = csv.reader(f7)
    dataReader8 = csv.reader(f8)
    dataReader9 = csv.reader(f9)
    dataReader10 = csv.reader(f10)
    dataReader11 = csv.reader(f11)
    dataReader12 = csv.reader(f12)
    dataReader13 = csv.reader(f13)
    dataReader14 = csv.reader(f14)
    for row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12,row13,row14 in zip(dataReader1,dataReader2,dataReader3,dataReader4,dataReader5,dataReader6,dataReader7,dataReader8,dataReader9,dataReader10,dataReader11,dataReader12,dataReader13,dataReader14):
        temp = []
        temp.extend(list(map(float,row1[1:6])))
        temp.extend(list(map(float,row2[1:16])))
        temp.extend(list(map(float,row3[1:6])))
        temp.extend(list(map(float,row4[1:18])))
        temp.extend(list(map(float,row5[1:18])))
        temp.extend(list(map(float,row6[1:18])))
        temp.extend(list(map(float,row7[1:18])))
        temp.extend(list(map(float,row8[1:18])))
        temp.extend(list(map(float,row9[1:18])))
        temp.extend(list(map(float,row10[1:18])))
        temp.extend(list(map(float,row11[1:18])))
        temp.extend(list(map(float,row12[1:18])))
        temp.extend(list(map(float,row13[1:18])))
        temp.extend(list(map(float,row14[1:18])))

        train_data.append(temp)
        train_target.append(float(row4[18]))

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    f6.close()
    f7.close()
    f8.close()
    f9.close()
    f10.close()
    f11.close()
    f12.close()
    f13.close()
    f14.close()

    train_data = np.array(train_data)

    temp = []
    for i in train_target:
        if i != 0:
            temp.append(1)
        else:
            temp.append(0)

    train_target = np.array(temp)

    return [train_data, train_target]


def cal_10_fold_CV_PROMISE(projectName,metricsName):
    ALL = {}
    ALL['Error of SC'] = []
    ALL['Error of train_data'] = []

    for file_name in projectName:
        ALL[file_name] = []
        count = 0
        all_count = 0

        SC = []
        PAM = []
        NG = []
        FCM = []
        KM = []

        RBM_feature_data_list = []

        components = []
        intercepts_visible = []
        intercepts_hidden = []
        error_list = []
        for times in range(int(PARA[1]),int(PARA[2])+1):


            print("+++ {0} times {1} fold +++".format(times,file_name))

            filename = "./../../data/AEEEM/{0}/{1}/test_times{2}.csv"

            train_data = []
            train_target = []

            train_data, train_target = data_extend(filename,file_name,times)

            if sum(train_target)==0 or sum(train_target)==len(train_target):
                all_count = all_count + 1
                print("---train data error---")
                continue

            test_data = [1,2,3]

            train_data_std = preparation.standardization(train_data,"0-1scale")


            random_state = 200


            for k in range(10,11):
                #print("# feature is {0}".format(k))

                model = BernoulliRBM(n_components=k,random_state=random_state)



                RBM_feature_data = model.fit_transform(train_data_std)

                components.append(model.components_)
                intercepts_hidden.append(model.intercept_hidden_)
                intercepts_visible.append(model.intercept_visible_)
                #print(model.transform(train_data_std))

                #print(RBM_feature_data) 
                #for temp in RBM_feature_data:
                #    print(temp)

                Origin_feature_data = preparation.standardization(train_data,"z-score")
                RBM_feature_data_list.append(RBM_feature_data)


                feature_data_SC = preparation.standardization(RBM_feature_data,"z-score")
                ans = clustering.cslSCbyR(feature_data_SC,RBM=True)

                if ans != 1:

                    #print('RBM with SC')
                    temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
                    #print(temp)
                    SC.append(temp)
                else :
                    print("Error")
                    error_list.append([times])


                #RBM_feature_data = model.fit_transform(train_data_std)

                ans = clustering.cslPAMbyR(RBM_feature_data)
                #ans = clustering.cslPAMbyR(train_data)

                #print('RBM with PAM')
                temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
                #print(temp)
                PAM.append(temp)


                ans = clustering.cslNGbyR(RBM_feature_data)

                #print('RBM with NG')
                temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
                #print(temp)
                NG.append(temp)


                #ans = cslSCbyR(train_data_std)
                ans = clustering.cslFCMbyR(RBM_feature_data)

                #print('RBM with FCM')
                temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
                #print(temp)
                FCM.append(temp)


                ans = clustering.cslKMbyR(RBM_feature_data)

                #print('RBM with KM')
                temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
                #print(temp)
                KM.append(temp)

                del train_data
                del train_target
                del train_data_std
                del Origin_feature_data
                del RBM_feature_data
                gc.collect()

        ALL[file_name].append(SC)
        ALL[file_name].append(PAM)
        ALL[file_name].append(NG)
        ALL[file_name].append(FCM)
        ALL[file_name].append(KM)
        ALL['Error of SC'].append(count)
        ALL['Error of train_data'].append(all_count)

        print(" --- result of {0} average --- ".format(file_name))
        try :
            print("SC : {0}".format(sum(ALL[file_name][0])/len(ALL[file_name][0])))
        except ZeroDivisionError:
            print("All SCs were not caluculated by ZeroDivisionError")
        print("PAM : {0}".format(sum(ALL[file_name][1])/len(ALL[file_name][1])))
        print("NG : {0}".format(sum(ALL[file_name][2])/len(ALL[file_name][2])))
        print("FCM : {0}".format(sum(ALL[file_name][3])/len(ALL[file_name][3])))
        print("KM : {0}".format(sum(ALL[file_name][4])/len(ALL[file_name][4])))


        f = open('./data/AEEEM/SC_Error_{0}_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)
        for row in error_list:
            csvWriter.writerow([row])


        f = open('./data/AEEEM/{0}_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for metrics in ALL[file_name]:
            csvWriter.writerow(metrics)


        f = open('./data/AEEEM/weight_{0}_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for count,metrics in enumerate(components):
            csvWriter.writerow("".format(count))
            for row in metrics:
                csvWriter.writerow(row)


        f = open('./data/AEEEM/intercepts_visible_{0}_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for count,metrics in enumerate(intercepts_visible):
            csvWriter.writerow("".format(count))
            csvWriter.writerow(metrics)


        f = open('./data/AEEEM/intercepts_hidden_{0}_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for count,metrics in enumerate(intercepts_hidden):
            csvWriter.writerow("".format(count))
            csvWriter.writerow(metrics)


        f = open('./data/AEEEM/{0}_metrics_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for metrics in RBM_feature_data_list:
            for metric in metrics:
                csvWriter.writerow(metric)

            csvWriter.writerow("")

cal_10_fold_CV_PROMISE(projectName,metricsName)


