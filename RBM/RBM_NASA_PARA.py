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


csvName = ['cm1','jm1','kc3','mc1','mc2','mw1','pc1','pc2','pc3','pc4','pc5']
projectName = 'NASA'


def data_extend(filename):
    train_data = []
    train_target = []

    f = open(filename,'r')
    dataReader = csv.reader(f)
    for row in dataReader:
        train_data.append(list(map(float,row[:-1])))
        train_target.append(row[-1])

    f.close()

    train_data = np.array(train_data)

    temp = []
    for i in train_target:
        if i == 'Y' or i=='TRUE':
            temp.append(1)
        elif i=='N' or i=='FALSE':
            temp.append(0)
        else:
            print('error')

    train_target = temp

    train_target = np.array(temp)

    return [train_data, train_target]


def cal_10_fold_CV_PROMISE(csvName,projectName):
    ALL = {}
    ALL['Error of SC'] = []
    ALL['Error of FCM'] = []
    ALL['Error of train_data'] = []

    for file_name in csvName:
        ALL[file_name] = []
        count = 0
        all_count = 0
        count_FCM = 0

        RBM_feature_data_list = []

        SC = []
        PAM = []
        NG = []
        FCM = []
        KM = []

        components = []
        intercepts_visible = []
        intercepts_hidden = []
        error_list = []
        for times in range(int(PARA[1]),int(PARA[2])+1):


            print("+++ {0} times {1} fold +++".format(times,file_name))

            filename = "./../../data/{0}/{1}/test_times{2}.csv".format(projectName, file_name, times)

            train_data = []
            train_target = []

            train_data, train_target = data_extend(filename)

            if sum(train_target)==0 or sum(train_target)==len(train_target):
                all_count = all_count + 1
                print("---train data error---")
                continue


            train_data_std = preparation.standardization(train_data,"0-1scale")


            random_state = 200


            #for k in range(256,257):
            for k in range(10,11):
                #print("# feature is {0}".format(k))

                model = BernoulliRBM(n_components=k,random_state=random_state)



                RBM_feature_data = model.fit_transform(train_data_std)
                RBM_feature_data_list.append(RBM_feature_data)

                components.append(model.components_)
                intercepts_hidden.append(model.intercept_hidden_)
                intercepts_visible.append(model.intercept_visible_)
                #print(model.transform(train_data_std))

                #print(RBM_feature_data) 
                #for temp in RBM_feature_data:
                #    print(temp)


                Origin_feature_data = preparation.standardization(train_data,"z-score")

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
        print("SC : {0}".format(sum(ALL[file_name][0])/len(ALL[file_name][0])))
        print("PAM : {0}".format(sum(ALL[file_name][1])/len(ALL[file_name][1])))
        print("NG : {0}".format(sum(ALL[file_name][2])/len(ALL[file_name][2])))
        print("FCM : {0}".format(sum(ALL[file_name][3])/len(ALL[file_name][3])))
        print("KM : {0}".format(sum(ALL[file_name][4])/len(ALL[file_name][4])))


        f = open('./data/{0}/SC_Error_{1}_{2}_{3}.csv'.format(projectName,file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)
        for row in error_list:
            csvWriter.writerow([row])


        f = open('./data/{1}/{0}_{2}_{3}.csv'.format(file_name,projectName,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for metrics in ALL[file_name]:
            csvWriter.writerow(metrics)


        f = open('./data/NASA/weight_{0}_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for count,metrics in enumerate(components):
            csvWriter.writerow("".format(count))
            for row in metrics:
                csvWriter.writerow(row)


        f = open('./data/NASA/intercepts_visible_{0}_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for count,metrics in enumerate(intercepts_visible):
            csvWriter.writerow("".format(count))
            csvWriter.writerow(metrics)


        f = open('./data/NASA/intercepts_hidden_{0}_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for count,metrics in enumerate(intercepts_hidden):
            csvWriter.writerow("".format(count))
            csvWriter.writerow(metrics)


        f = open('./data/NASA/{0}_metrics_{1}_{2}.csv'.format(file_name,PARA[1],PARA[2]),'w')
        csvWriter = csv.writer(f)

        for metrics in RBM_feature_data_list:
            for metric in metrics:
                csvWriter.writerow(metric)

            csvWriter.writerow("")

cal_10_fold_CV_PROMISE(csvName,projectName)
