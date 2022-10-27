import numpy as np
import csv
import sys
from sklearn.cluster import FeatureAgglomeration
import gc
sys.path.append("./../Util/")

import preparation
import clustering
import evaluation

PARA = sys.argv

csvName = ['ant_v1_7','camel_v1_6','ivy_v1_4','jedit_v4_0','log4j_v1_0','lucene_v2_4','poi_v3_0','tomcat_v6_0','xalan_v2_6','xerces_v1_3']
projectName = "PROMISE"


def data_extend(filename):
    train_data = []
    train_target = []

    f = open(filename,'r')
    dataReader = csv.reader(f)
    for row in dataReader:
        train_data.append(list(map(float,row[3:-1])))
        train_target.append(float(row[-1]))

    f.close()

    train_data = np.array(train_data)

    temp = []
    for i in train_target:
        if i != 0:
            temp.append(1)
        else:
            temp.append(0)

    train_target = np.array(temp)

    return [train_data, train_target]


def cal_10_fold_cv(times1,times2):

    file_name = csvName[int(PARA[3])]

    ALL = []

    SC = []
    PAM = []
    NG = []
    FCM = []
    KM = []

    feature_data_list = []

    components = []
    intercepts_visible = []
    error_list = []
    for times in range(times1,times2):
    #for times in range(1,2):


        print("+++ {0} times {1} fold +++".format(times,file_name))

        filename = "./../../data/{0}/{1}/test_times{2}.csv".format(projectName, file_name, times)

        train_data = []
        train_target = []

        train_data, train_target = data_extend(filename)

        Origin_feature_data = preparation.standardization(train_data,"z-score")


        model = FeatureAgglomeration(n_clusters=10)
        feature_data = model.fit_transform(Origin_feature_data)


        feature_data_list.append(feature_data)


        feature_data_SC = preparation.standardization(feature_data,"z-score")
        ans = clustering.cslSCbyR(feature_data_SC,RBM=True)

        if ans != 1:

            #print('RBM with SC')
            temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
            #print(temp)
            SC.append(temp)
        else :
            print("Error")
            error_list.append([times])




        ans = clustering.cslPAMbyR(feature_data)
        #ans = clustering.cslPAMbyR(train_data)

        #print('RBM with PAM')
        temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
        #print(temp)
        PAM.append(temp)


        ans = clustering.cslNGbyR(feature_data)

        #print('RBM with NG')
        temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
        #print(temp)
        NG.append(temp)


        #ans = cslSCbyR(train_data_std)
        ans = clustering.cslFCMbyR(feature_data)

        #print('RBM with FCM')
        temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
        #print(temp)
        FCM.append(temp)


        ans = clustering.cslKMbyR(feature_data)

        #print('RBM with KM')
        temp = evaluation.calAUC(ans,train_target,Origin_feature_data)
        #print(temp)
        KM.append(temp)

        del train_data
        del train_target
        del feature_data
        del Origin_feature_data
        gc.collect()

    ALL.append(SC)
    ALL.append(PAM)
    ALL.append(NG)
    ALL.append(FCM)
    ALL.append(KM)


    f = open('./data/{0}/SC_Error_{1}_{2}_{3}.csv'.format(projectName,file_name,times1,times2),'w')
    csvWriter = csv.writer(f)
    for row in error_list:
        csvWriter.writerow([row])


    f = open('./data/{0}/{1}_{2}_{3}.csv'.format(projectName,file_name,times1,times2),'w')
    csvWriter = csv.writer(f)

    for metrics in ALL:
        csvWriter.writerow(metrics)



    f = open('./data/{0}/{1}_metrics_{2}_{3}.csv'.format(projectName,file_name,times1,times2),'w')
    csvWriter = csv.writer(f)

    for metrics in feature_data_list:
        for metric in metrics:
            csvWriter.writerow(metric)

        csvWriter.writerow("")


cal_10_fold_cv(int(PARA[1]),int(PARA[2]))
