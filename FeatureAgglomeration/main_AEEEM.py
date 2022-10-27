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


csvName = ['eclipse','equinox','lucene','mylyn','pde']
metricsName = ['bug-metrics','change-metrics','complexity-code-change','single-version-ck-oo','churn','exp-churn','lin-churn','log-churn','weighted-churn','ent','exp-ent','lin-ent','log-ent','weighted-ent']
projectName = 'AEEEM'


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

        filename = "./../../data/AEEEM/{0}/{1}/test_times{2}.csv"

        train_data = []
        train_target = []

        train_data, train_target = data_extend(filename,file_name,times)

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


#cal_10_fold_cv(1,101)
#cal_10_fold_cv(101,201)
#cal_10_fold_cv(201,301)
#cal_10_fold_cv(301,401)
#cal_10_fold_cv(401,501)
cal_10_fold_cv(int(PARA[1]),int(PARA[2]))

