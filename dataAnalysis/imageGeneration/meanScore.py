import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

#cnn
#score = np.load("C:/Users/marti/Desktop/memoire/imagesMemoire/results/regression/cnn/scoreCnnSingleSubject.npy")
score = np.array([[(427.59089352021437, 14.94, 40.33, 0.5225411259431805, 13.4365392561573), (260.3900999189172, 11.16, 52.2, 0.438408643830972, 10.488219144733005), (673.2784083735976, 18.58, 30.84, 0.2500152113412629, 21.470669560214574), (342.46563447781756, 13.31, 44.53, 0.40873674607093796, 13.282363200710249), (273.2193600136867, 11.36, 50.11, 0.5783601803974395, 10.26523599165876), (389.68916037394507, 13.82, 43.09, 0.32983288950396616, 14.679912562242322)], [(503.55907496561633, 15.99, 28.21, 0.5538456238745639, 15.393227737952582), [None, None, None, None, None], (293.8654698039628, 11.82, 56.07, 0.7452126559514305, 8.302205613262196), (393.66693482140744, 14.11, 43.99, 0.6068789055783976, 12.495460894569954), (310.08399614369176, 12.21, 45.77, 0.6717221847860574, 10.045528919849225), (373.7427390125852, 12.65, 58.28, 0.6034639230177764, 10.651212382354878)], [(537.7303023199712, 17.06, 28.9, 0.4588916610102361, 16.432862707171207), (284.1379253827223, 12.53, 45.24, 0.5423186411872932, 10.13946087218645), (353.1801994787947, 13.56, 48.05, 0.5793245705667011, 10.907434307993393), (402.08847783072497, 15.12, 42.5, 0.5314905489142702, 12.75635747073924), (344.6878724851021, 13.61, 48.09, 0.5702007447471549, 10.947630631398832), (334.04857397413554, 13.32, 57.59, 0.4066232025062741, 10.057264244960573)], [(404.2693988346907, 14.62, 38.44, 0.4546285503972319, 14.014821925297792), (532.8544269760798, 15.57, 32.69, 0.2099496461702138, 20.386519230202776), (420.6366611934446, 14.41, 46.06, 0.5142280461287974, 13.653625749954957), (334.0422357744581, 13.02, 44.79, 0.4931084453126205, 11.97921206990083), (231.18393933366985, 10.69, 52.76, 0.5100632676871125, 9.287922943086507), (254.00446357059582, 11.29, 48.12, 0.5532090377069758, 9.796061209137521)], [(440.299810985162, 14.96, 30.0, 0.5192095545756652, 14.9181810590947), (323.1999348635668, 12.64, 50.89, 0.5828302618722663, 10.59566733679116), (249.89691537236925, 11.0, 56.29, 0.7029329816431059, 7.705933041973758), (442.85163509023045, 14.71, 44.93, 0.5359426008761982, 13.619857653984099), (382.04527501379755, 14.08, 42.64, 0.5136191233914257, 12.9951266098637), (370.21111267175706, 13.79, 49.9, 0.527410404383991, 11.914664800329493)], [(545.3679583146724, 17.72, 10.55, 0.3370197621334521, 20.511682573731957), (705.7990088229268, 20.0, 36.97, 0.41179042742915406, 20.17638884161466), (508.09121134378586, 16.16, 52.76, 0.4432251165167488, 14.346377638697007), (347.41918299257844, 13.91, 34.17, 0.6142658098204993, 11.730156321520857), (442.7972512036887, 15.59, 33.46, 0.44458059200604344, 14.490985210822927), (439.1144150973589, 14.7, 50.69, 0.4702596628266103, 13.508807769490446)], [(488.1155892492651, 15.12, 22.98, 0.5331946402135183, 17.04268489684989), (453.4593028123718, 15.37, 41.36, 0.5049987715993799, 13.954897068727792), (346.6240144654741, 12.72, 45.66, 0.691945743962347, 10.096561381284571), (380.7271630430973, 13.69, 42.67, 0.5953584350386415, 11.81889893069638), (384.6232240270558, 13.88, 43.16, 0.5960797193124544, 12.559187187447376), (421.948267652686, 14.95, 38.45, 0.5700524748086319, 13.002018687080731)], [(428.8999002615798, 14.98, 32.36, 0.40010672076106735, 15.784854122315297), (471.1501822373934, 15.63, 49.08, 0.6121623532336881, 13.5532968101981), (331.0787703479718, 12.84, 49.49, 0.6770285503507794, 9.82404758388638), (570.6602634768544, 17.75, 8.34, 0.5375299803060859, 18.51885967539495), (382.49861825545315, 14.31, 39.93, 0.4753797685957139, 12.917562501234045), (599.7684830307196, 17.88, 40.68, 0.3585499418042502, 18.475139672697633)], [(566.6165526093772, 18.06, 28.02, 0.4356991105186657, 18.105510430224825), (324.78334384759023, 12.96, 49.17, 0.6153644226864008, 10.177326405291286), (297.64351783354186, 12.17, 50.58, 0.6611706562239762, 8.587087371382623), (367.4209380008148, 13.93, 46.84, 0.6394212121478401, 11.247808437935227), (484.44898721806317, 16.14, 26.54, 0.5095207587364068, 15.572171823530645), (466.470135385794, 15.37, 58.34, 0.4562158077327015, 13.111915691497032)], [(529.3555372023438, 17.03, 37.57, 0.4989970431699952, 15.347348281913279), (290.7598383317097, 12.55, 51.02, 0.6100125355589435, 9.473978337388303), (256.93715250273505, 11.35, 61.51, 0.7023222116311444, 7.271574414563512), (310.46233156797007, 13.24, 50.87, 0.6725603715590117, 8.835960724490274), (237.7420166302035, 10.74, 47.5, 0.5312120102731368, 9.938849752420564), (408.91731067308405, 14.85, 47.35, 0.4563726702414878, 13.162946801284393)], [(486.85489511863415, 16.1, 21.67, 0.2841045262704595, 18.665093256068904), (247.39619377886237, 11.33, 40.56, 0.31011698141281546, 11.373671903169397), (296.8713418953569, 12.3, 46.48, 0.42053871467354753, 11.386743941402365), (338.15332569081994, 13.87, 34.97, 0.47534010354755324, 13.028583965163792), [None, None, None, None, None], (439.48707656891514, 15.35, 39.38, 0.35733129951188297, 14.969516041418721)], [(497.3018706051393, 15.71, 36.25, 0.408043176833885, 17.633755591873054), (266.64149929452464, 11.45, 49.84, 0.7291178269740726, 8.003651137585683), (379.5142900003658, 12.82, 52.82, 0.704435772226138, 10.979241132011124), (362.8625200021065, 13.2, 38.13, 0.6077985302350463, 11.636818594925444), (472.0139765587062, 15.25, 31.3, 0.49192080124724125, 16.165255025436327), (791.2560413018483, 20.8, 26.9, 0.24207766533582006, 23.519669353955383)], [(386.73319471442863, 13.91, 32.04, 0.5518531912287276, 13.472549390347716), (241.4521761425201, 10.81, 54.11, 0.7138453996703319, 7.970429678239601), (348.26089364114375, 11.92, 54.86, 0.7002900215985173, 10.126407799824115), (290.07978814795194, 12.13, 50.18, 0.6285929978017892, 9.509467753735123), (190.54268314423774, 9.54, 52.97, 0.5806972254686542, 8.269457593163693), (358.8623322643644, 13.13, 57.19, 0.5977326974222938, 10.439024017505815)], [(332.0229458682369, 14.24, 20.23, 0.26533274630506515, 17.30952498410981), (256.9568376008532, 11.75, 48.43, 0.4213019860990667, 10.46133967842411), (507.58109489303735, 16.14, 38.54, 0.32206333344779026, 18.11441916217195), (335.11486055914725, 14.18, 33.77, 0.2865125146279522, 14.704076379959552), (497.48922357352825, 16.09, 35.15, 0.33778585149132057, 17.28771411977696), (299.66632117453344, 12.48, 49.42, 0.45335522368899867, 11.660204510641627)], [(419.54982054621667, 14.71, 37.61, 0.49055628417837255, 14.29378300157682), (324.7782595635396, 12.57, 48.37, 0.4830520328378426, 11.70492929661401), (345.38594731556014, 13.28, 49.89, 0.6538045712458233, 10.557497923930532), (270.25099741108374, 11.5, 50.17, 0.5686364760116922, 9.696899842330506), (283.69521249213, 11.74, 45.79, 0.5732365642616858, 10.663874211405233), (345.18953520103366, 12.88, 47.82, 0.46886336140388374, 12.569075750394655)]])
score = list(score)
for i in range(len(score)):
    c = []
    for j in range(len(score[i])):
        if None not in score[i][j]:
            c.append(score[i][j])
    score[i] = c

m = [[] for _ in range(5)]
for i in range(5):
    for s in range(len(score)):
        for k in range(len(score[s])):
            m[i].append(score[s][k][i])

m = np.mean(m, (1,))

print("cnn enveloppe, single subject", m)


#TDF single subject
score = np.load("C:/Users/marti/Desktop/memoire/imagesMemoire/results/regression/estimation/RegressionScoreStandardTDFSingleSuject.npy")
score = score.transpose((2, 0, 1, 3))

models = ["LR", "KNN", 'RF', 'MLP']
for i, s in enumerate(score):
    print(s.shape)
    m = np.round(np.mean(s, (0, 1)), 2)
    print(models[i], "TDF, single subject", m)


#TDF inter subject (session)
score = np.load("C:/Users/marti/Desktop/memoire/imagesMemoire/results/regression/estimation/RegressionScoreStandardTDFInterSubjectsSessions.npy")
score = score.transpose((1, 0, 2))

for i in range(len(score)):
    print(score[i].shape)
    m = np.round(np.mean(score[i], (0,)), 2)
    print(models[i], "TDF, inter subject 1", m)


#TDF inter subject (subject)
score = np.load("C:/Users/marti/Desktop/memoire/imagesMemoire/results/regression/estimation/RegressionScoreStandardTDFInterSubjectsSubjects.npy")
score = score.transpose((1, 0, 2))

for i in range(len(score)):
    print(score[i].shape)
    m = np.round(np.mean(score[i], (0,)), 2)
    print(models[i], "TDF, inter subject 2", m)


#CSP single subject
score = np.load("C:/Users/marti/Desktop/memoire/imagesMemoire/results/regression/estimation/RegressionScoreStandardCSPSingleSuject.npy")
score = score.transpose((2, 0, 1, 3))

models = ["LR", "KNN", 'RF', 'MLP']
for i, s in enumerate(score):
    print(s.shape)
    m = np.round(np.mean(s, (0, 1)), 2)
    print(models[i], "CSP, single subject", m)


#CSP inter subject (session)
score = np.load("C:/Users/marti/Desktop/memoire/imagesMemoire/results/regression/estimation/RegressionScoreStandardCSPInterSubjectsSessions.npy")
score = score.transpose((1, 0, 2))

for i in range(len(score)):
    print(score[i].shape)
    m = np.round(np.mean(score[i], (0,)), 2)
    print(models[i], "CSP, inter subject 1", m)


#CSP inter subject (subject)
score = np.load("C:/Users/marti/Desktop/memoire/imagesMemoire/results/regression/estimation/RegressionScoreStandardCSPInterSubjectsSubjects.npy")
score = score.transpose((1, 0, 2))

for i in range(len(score)):
    print(score[i].shape)
    m = np.round(np.mean(score[i], (0,)), 2)
    print(models[i], "CSP, inter subject 2", m)
