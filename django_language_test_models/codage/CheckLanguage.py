import csv
import math
from collections import Counter

import joblib
import numpy as np
import random as rd
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.ensemble._forest import ExtraTreesClassifier
from sklearn.ensemble._gb import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
from sklearn.ensemble._voting import VotingClassifier
from sklearn.ensemble._weight_boosting import AdaBoostClassifier
from sklearn.linear_model._logistic import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection._search import GridSearchCV
from sklearn.model_selection._validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors._classification import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.preprocessing._data import StandardScaler
from sklearn.svm._classes import SVC
from sklearn.tree._classes import DecisionTreeClassifier

from django_language_test_models.codage.Test import IsTheCodeUniquelyDeciperable


def isPrefixe(string_code, string_prefix):
    return string_code.startswith(string_prefix)

def languageHasPrefix(langage,string_prefix):
    for i in langage:
        if isPrefixe(i,string_prefix) and i != string_prefix:
            return True
    return False

def countPrefixe(langage):
    nombrePrefixe=0
    for i in langage :
        if languageHasPrefix(langage,i):
            nombrePrefixe+=1
    return nombrePrefixe
def removePrefix(string_code, string_prefix):
    resultat = string_code.removeprefix(string_prefix)
    if resultat == "":
        resultat = "epsi";
    return resultat


def getResiduelFromTwoList(L, L1):
    Lfinal = []
    for residuel in L:
        Lfinal += getResiduelOfAlangage(residuel, L1)
    return Lfinal


def getResiduelOfAlangage(residuel="", langage=[]):
    resultat = []
    for lan in langage:
        if isPrefixe(lan, residuel):
            resultat.append(removePrefix(lan, residuel))
    return resultat


#  remove epsilon
def removeEpsilon(langage=[]):
    return remove_items(langage, "epsi")


def remove_items(test_list, item):
    res = [i for i in test_list if i != item]
    return res


# get l1
def getL1(langage):
    result = getResiduelFromTwoList(langage, langage)
    return removeEpsilon(result)


# get ln+1
def getLnPlus1(langageN, langage):
    L_Ln1 = getResiduelFromTwoList(langage, langageN)
    ln1_L = getResiduelFromTwoList(langageN, langage)
    return L_Ln1 + ln1_L


def checkIfCode(langage):
    historique = []
    #     set l0
    historique.append(langage)
    #     set l1
    ln = getL1(langage)
    #     get ln +
    lnplus = getLnPlus1(ln, langage)
    while lnplus.__contains__("epsi") != True:
        lnplus = getLnPlus1(lnplus, langage)

        # if (historique.__contains__(lnplus)):
        if (checkTabPrincipale(historique, lnplus)):
            return True
        historique.append(lnplus)
    return False


# generation fichiers
def decimal_to_binary(number):
    if number == 0:
        return '0'
    binary = ''
    while number > 0:
        binary = str(number % 2) + binary
        number //= 2

    return binary


def generateBinary(langage):
    number = rd.randint(0, 127)
    binary = decimal_to_binary(number)
    if langage.__contains__(binary):
        binary = generateBinary(langage)
    return binary


def generatelangage():
    size = rd.randint(1, 10)
    langage = []
    for i in range(0, size):
        bin = generateBinary(langage)
        langage.append(bin)
    return langage


def generatelangageArray(langagePrincipale=[]):
    lan = generatelangage()
    if checkTabPrincipale(langagePrincipale, lan):
        lan = generatelangageArray(langagePrincipale)
    return lan


def checkIfContains(tab1=[], tab2=[]):
    count = 0
    size = len(tab2)
    for i in tab2:
        if tab1.__contains__(i):
            count += 1
    if count == size:
        return True
    return False


def checkTabPrincipale(tabPrincipale=[], langage=[]):
    for i in tabPrincipale:
        if checkIfContains(i, langage):
            return True
    return False


def generateABunchOflangage(number):
    langages = []
    for i in range(0, number):
        # langages.append(generatelangageArray(langages))
        langages.append(generatelangage())
    return langages


def generateAnEqualListOflangages(number):
    code = []
    notCode = []
    allLan = code + notCode
    max = number // 2

    i = 0
    while len(allLan) != number:
        i += 1
        langage = generatelangageArray(allLan)
        print("(",langage,"," ,checkIfCode(langage),"),")
        if checkIfCode(langage) and len(code) < (max):
            code.append(langage)
            allLan.append(langage)
        elif checkIfCode(langage) == False and len(notCode) < (max):
            notCode.append(langage)
            allLan.append(langage)
    return code + notCode


# get the features YAYYYY
def getAverageLength(langage):
    return sum(len(seq) for seq in langage) / len(langage) if langage else 0

def get_0_proportion(langage):
    total_0 = sum(seq.count('0') for seq in langage)
    total_bits = sum(len(seq) for seq in langage)
    return total_0 / total_bits if total_bits else 0

def get_1_proportion(langage):
    total_1 = sum(seq.count('1') for seq in langage)
    total_bits = sum(len(seq) for seq in langage)
    return total_1 / total_bits if total_bits else 0

def get_ecartType_nombre_sequence(langage):
    moyenne = getAverageLength(langage)
    sommeCarreEcart = sum((len(x) - moyenne) ** 2 for x in langage)
    variance = sommeCarreEcart / len(langage)
    ecartType = np.sqrt(variance)
    return ecartType

def get_number_seq_start_1(langage):
    return sum(seq.startswith('1') for seq in langage)

def get_number_seq_start_0(langage):
    return sum(seq.startswith('0') for seq in langage)

def get_medianeLength(langage):
    lengths = [len(seq) for seq in langage]
    return pd.Series(lengths).median() if lengths else 0

def get_mode(langage):
    lengths = [len(seq) for seq in langage]
    return pd.Series(lengths).mode()[0] if lengths else 0

def getEcartInterQuartile(langage):
    lengths = [len(seq) for seq in langage]
    return pd.Series(lengths).quantile(0.75) - pd.Series(lengths).quantile(0.25) if lengths else 0

def get_squew(langage):
    lengths = [len(seq) for seq in langage]
    return pd.Series(lengths).skew() if lengths else 0

def minLength(langage):
    lengths = [len(seq) for seq in langage]
    return min(lengths) if lengths else 0

def maxLength(langage):
    lengths = [len(seq) for seq in langage]
    return max(lengths) if lengths else 0

def calculate_entropy(langage):
    from math import log2
    lengths = [len(seq) for seq in langage]
    total = sum(lengths)
    entropy = -sum((length/total) * log2(length/total) for length in lengths) if total else 0
    return entropy

def count_total_bit_transitions(langage):
    total_transitions = 0
    for seq in langage:
        total_transitions += sum(seq[i] != seq[i+1] for i in range(len(seq)-1))
    return total_transitions

# générer l'array pour le ML
def create_data_row(langage=[]):
    row = []
    # Longueur moyenne des séquences
    row.append(getAverageLength(langage))
    # Longueur des elements du langage séquences
    row.append(len(langage))
    # Proportion de '0' dans les séquences
    row.append(get_0_proportion(langage))
    # Proportion de '1' dans les séquences
    row.append(get_1_proportion(langage))
    # Écart type des longueurs des séquences
    row.append(get_ecartType_nombre_sequence(langage))
    # Nombre de séquences commençant par '1'
    row.append(get_number_seq_start_1(langage))
    # Nombre de séquences commençant par '0'
    row.append(get_number_seq_start_0(langage))
    # Médiane des longueurs des séquences
    row.append(get_medianeLength(langage))
    # Mode des longueurs des séquences
    row.append(get_mode(langage))
    # Écart interquartile des longueurs des séquences
    row.append(getEcartInterQuartile(langage))
    # Skewness des longueurs des séquences
    row.append(get_squew(langage))
    # longueur min
    row.append(minLength(langage))
    # longueur max
    row.append(maxLength(langage))
    # entropy
    row.append(calculate_entropy(langage))
    # bit trnsition
    row.append(count_total_bit_transitions(langage))
    # dangling suffixe
    row.append(0 if IsTheCodeUniquelyDeciperable(set(langage)) else 1)
    # voir si c'est un code ou non
    if checkIfCode(langage):
        row.append(1)
    else:
        row.append(0)
    return row

def create_language_properties (langage=[]):
    row = []
    # Longueur moyenne des séquences
    row.append(getAverageLength(langage))
    # Longueur des elements du langage séquences
    row.append(len(langage))
    # Proportion de '0' dans les séquences
    row.append(get_0_proportion(langage))
    # Proportion de '1' dans les séquences
    row.append(get_1_proportion(langage))
    # Écart type des longueurs des séquences
    row.append(get_ecartType_nombre_sequence(langage))
    # Nombre de séquences commençant par '1'
    row.append(get_number_seq_start_1(langage))
    # Nombre de séquences commençant par '0'
    row.append(get_number_seq_start_0(langage))
    # Médiane des longueurs des séquences
    row.append(get_medianeLength(langage))
    # Mode des longueurs des séquences
    row.append(get_mode(langage))
    # Écart interquartile des longueurs des séquences
    row.append(getEcartInterQuartile(langage))
    # Skewness des longueurs des séquences
    # row.append(get_squew(langage))
    # longueur min
    row.append(minLength(langage))
    # longueur max
    row.append(maxLength(langage))
    # entropy
    row.append(calculate_entropy(langage))
    # bit trnsition
    row.append(count_total_bit_transitions(langage))
    # dangling suffixe
    row.append(0 if IsTheCodeUniquelyDeciperable(set(langage)) else 1)
    return row

# generer les listes de données
def create_data_from_list(langages=[]):
    data = []
    for langage in langages:
        row = create_data_row(langage)
        data.append(row)
    return data

def getRandomFalseLanguage():
    langage= generatelangage()
    if checkIfCode(langage):
        return getRandomFalseLanguage()
    return langage

def save_langages_to_csv(langages, filename):
    header = [
        "AverageLength", "longueur", "Proportion0", "Proportion1", "ecartType",
        "NumSeqStart1", "NumSeqStart0", "MedianLength", "ModeLength",
        "IQRLength", "SkewnessLength", "minLength", "max length", "entropy", "bit_transition","dangling_prefixes", "IsCode"
    ]
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for langage in langages:
            row = create_data_row(langage)
            writer.writerow(row)


# create a model

def createAdaBoostClassifierModel():
    df = pd.read_csv('/Users/priscafehiarisoadama/django_language_Prediction_models/data/data3.csv')
    test_data=pd.read_csv('/Users/priscafehiarisoadama/django_language_Prediction_models/data/data3.csv')
    # df=pd.concat([df,test_data])
    # df.SkewnessLength = df.SkewnessLength.fillna(0)

    df.ecartType=df.ecartType.fillna(df.ecartType.mean())
    df.SkewnessLength=df.SkewnessLength.fillna(df.SkewnessLength.mean())

    # df=df.drop(['longueur'],axis=1)
    # df=df.drop(['ModeLength'],axis=1)
    # df=df.drop(['MedianLength'],axis=1)
    # df=df.drop(['AverageLength'],axis=1)
    # df=df.drop(['IQRLength'],axis=1)
    # df=df.drop(['NumSeqStart0'],axis=1)
    # df=df.drop(['NumSeqStart1'],axis=1)
    # df=df.drop(['Proportion1'],axis=1)
    # df=df.drop(['Proportion0'],axis=1)
    df=df.drop(['SkewnessLength'],axis=1)

    # df=df.drop(['entropy'],axis=1)
    # df=df.drop(['bit_transition'],axis=1)
    training_data = df
    np.random.seed(42)
    X=training_data.drop("IsCode",axis=1)
    Y=training_data["IsCode"]
    training_columns = X.columns
    print(training_columns)
    print("unique ",df["IsCode"])

    # split the data
    x_train, x_test, y_train , y_test=train_test_split(X,Y,test_size= 0.3)
    # Normalize the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    ada = AdaBoostClassifier(algorithm='SAMME', n_estimators=100, random_state=0)
    ada = ada.fit(x_train_scaled, y_train)
    scores4 = ada.score(x_test_scaled, y_test)
    print(scores4)

    # Save the model and scaler
    joblib.dump(ada, '/Users/priscafehiarisoadama/django_language_Prediction_models/data/model/best_ada_model.pkl')
    joblib.dump(scaler, '/Users/priscafehiarisoadama/django_language_Prediction_models/data/model/scaler.pkl')
    joblib.dump(training_columns, '/Users/priscafehiarisoadama/django_language_Prediction_models/data/model/training_columns.pkl')


def predict_language( language):

    modelPath="/Users/priscafehiarisoadama/django_language_Prediction_models/data/model/best_ada_model.pkl"
    scalerPath="/Users/priscafehiarisoadama/django_language_Prediction_models/data/model/scaler.pkl"
    columns="/Users/priscafehiarisoadama/django_language_Prediction_models/data/model/training_columns.pkl"

    model = joblib.load(modelPath)
    scaler = joblib.load(scalerPath)
    columns = joblib.load(columns)
    properties = create_language_properties(language)
    properties_df = pd.DataFrame(data=[properties], columns=columns)  # Create DataFrame with correct columns
    properties_scaled = scaler.transform(properties_df)
    resp=model.predict(properties_scaled)[0]
    if resp==1:
        return True
    return False

# langage = generatelangage()
langage =['1100', '110011', '1010', '1011010', '1111010', '110', '100001', '1010100', '1010101']
# lan = generateAnEqualListOflangages(5000)
## save langages to csv
# save_langages_to_csv(lan,"/Users/priscafehiarisoadama/django_language_Prediction_models/data/data3.csv")


# createAdaBoostClassifierModel()
examples = [
    ['111101', '1111100', '110', '11101'],
    ['111000', '0011', '1001011', '1000111', '11', '010101', '000100'],
    ['10', '1010'],
    ['01100', '101', '1011', '1010', '0100010', '111000', '0', '110001', '00111', '001'],
    ['100001'],
    ['01', '111110', '0100', '01'],
    ['00000', '000', '10101', '100', '01', '111001'],
    ['00101', '0000111', '1001', '100011', '0001', '101', '11', '01101'],
    ['10110', '001110', '1', '0', '01001', '10', '1101110'],
    ['100', '01110'],
    ['1010', '0', '010110', '0011', '0010', '10111', '1']
]
#
for i in examples:
    print(i)
    print("predicted : ",predict_language(i))
    print("sardinas : ",checkIfCode(i))

# print(lan)
# for i in lan:
#     print(i)
#     print("predicted : ",predict_language(i))
#     print("sardinas : ",checkIfCode(i))
#     print("prefixes",countPrefixe(i))
# for ii in examples:
#     print(get_number_seq_start_0(ii))

# l=['100011', '110011', '1101100', '111', '100100', '1000011', '1111110', '1000110', '11101', '111101']
# print("sardinas : ",checkIfCode(l))













