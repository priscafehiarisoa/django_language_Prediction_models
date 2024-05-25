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


def isPrefixe(string_code, string_prefix):
    return string_code.startswith(string_prefix)


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
        print(i)
        langage = generatelangageArray(allLan)
        print(langage, checkIfCode(langage))
        if checkIfCode(langage) and len(code) < (max):
            code.append(langage)
            allLan.append(langage)
        elif checkIfCode(langage) == False and len(notCode) < (max):
            notCode.append(langage)
            allLan.append(langage)
    return code + notCode


# get the features YAYYYY
def getAverageLength(langage=[]):
    count = len(langage)
    sumCount = 0
    for i in langage:
        sumCount += len(i)
    return sumCount / count


def get_0_proportion(langage=[]):
    digit_0_count = sum(s.count('0') for s in langage)
    digit_1_count = sum(s.count('1') for s in langage)
    total_digits = (digit_1_count if digit_1_count else 0) + (digit_0_count if digit_0_count else 0)
    return digit_0_count / total_digits


def get_1_proportion(langage=[]):
    digit_0_count = sum(s.count('0') for s in langage)
    digit_1_count = sum(s.count('1') for s in langage)
    total_digits = (digit_1_count if digit_1_count else 0) + (digit_0_count if digit_0_count else 0)
    return digit_1_count / total_digits


def get_ecartType_nombre_sequence(langage=[]):
    moyenne = getAverageLength(langage)
    sommeCarreEcart = sum((len(x) - moyenne) ** 2 for x in langage)
    variance = sommeCarreEcart / len(langage)
    ecartType = np.sqrt(variance)
    return ecartType


def get_number_seq_start_1(langage=[]):
    count = 0
    for i in langage:
        if (str(i).startswith("1")):
            count += 1
    return count


def get_number_seq_start_0(langage=[]):
    count = 0
    for i in langage:
        if (str(i).startswith("0")):
            count += 1
    return count


def get_medianeLength(langage=[]):
    lengths = []
    for i in langage:
        lengths.append(len(i))
    return np.median(lengths)


# La longueur la plus fréquente parmi toutes les séquences.
def get_mode(langage=[]):
    lengths = []
    for i in langage:
        lengths.append(len(i))
    return np.argmax(np.bincount(lengths))


def getEcartInterQuartile(langage=[]):
    # Écart interquartile
    lengths = []
    for i in langage:
        lengths.append(len(i))
    quartiles = np.percentile(lengths, [25, 75])
    return quartiles[1] - quartiles[0]


def maxLength(langage=[]):
    lengths = []
    for i in langage:
        lengths.append(len(i))
    return max(lengths)


def minLength(langage=[]):
    lengths = []
    for i in langage:
        lengths.append(len(i))
    return min(lengths)


# Une mesure de l'asymétrie de la distribution des longueurs des séquences par rapport à la moyenne.
def get_squew(langage=[]):
    lengths = []
    for i in langage:
        lengths.append(len(i))
    return np.mean((lengths - np.mean(lengths)) ** 3) / np.std(lengths) ** 3


def calculate_entropy(language):
    # Combine all sequences into a single string
    combined_sequence = ''.join(language)

    # Calculate the frequency of each character
    char_count = Counter(combined_sequence)
    total_chars = len(combined_sequence)

    # Calculate the entropy
    entropy = 0
    for char, count in char_count.items():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)

    return entropy


def count_total_bit_transitions(language):
    total_transitions = 0
    for sequence in language:
        # Iterate through the sequence and count the transitions
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i - 1]:
                total_transitions += 1
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
    # row.append(len(langage))
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
        "IQRLength", "SkewnessLength", "minLength", "max length", "entropy", "bit_transition", "IsCode"
    ]
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for langage in langages:
            row = create_data_row(langage)
            writer.writerow(row)


# create a model

def createAdaBoostClassifierModel():
    df = pd.read_csv('/Users/priscafehiarisoadama/IdeaProjects/django_language_test_models/data/data.csv')
    test_data=pd.read_csv('/Users/priscafehiarisoadama/IdeaProjects/django_language_test_models/data/data2.csv')
    df=df+test_data
    # df.SkewnessLength = df.SkewnessLength.fillna(0)
    df=df.drop(['longueur'],axis=1)
    df=df.drop(['SkewnessLength'],axis=1)
    training_data = df
    np.random.seed(42)
    X=training_data.drop("IsCode",axis=1)
    Y=training_data["IsCode"]
    training_columns = X.columns
    print(training_columns)
    print("unique ",df["IsCode"])

    # split the data
    x_train, x_test, y_train , y_test=train_test_split(X,Y,test_size= 0.2)
    # Normalize the data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    ada = AdaBoostClassifier(algorithm='SAMME', n_estimators=100, random_state=0)
    ada = ada.fit(x_train_scaled, y_train)
    scores4 = ada.score(x_test_scaled, y_test)
    print(scores4)

    # Save the model and scaler
    joblib.dump(ada, '../../data/model/best_ada_model.pkl')
    joblib.dump(scaler, '../../data/model/scaler.pkl')
    joblib.dump(training_columns, '../../data/model/training_columns.pkl')


def predict_language( language):

    modelPath="/Users/priscafehiarisoadama/IdeaProjects/django_language_test_models/data/model/best_ada_model.pkl"
    scalerPath="/Users/priscafehiarisoadama/IdeaProjects/django_language_test_models/data/model/scaler.pkl"
    columns="/Users/priscafehiarisoadama/IdeaProjects/django_language_test_models/data/model/training_columns.pkl"

    model = joblib.load(modelPath)
    scaler = joblib.load(scalerPath)
    columns = joblib.load(columns)
    properties = create_language_properties(language)
    properties_df = pd.DataFrame(data=[properties], columns=columns)  # Create DataFrame with correct columns
    properties_scaled = scaler.transform(properties_df)
    resp=model.predict(properties_scaled)[0]
    if resp==2:
        return True
    return False

# langage = generatelangage()
langage =['1100', '110011', '1010', '1011010', '1111010', '110', '100001', '1010100', '1010101']
# lan = generateAnEqualListOflangages(5000)
## save langages to csv
# save_langages_to_csv(lan,"/Users/priscafehiarisoadama/IdeaProjects/django_language_test_models/data/data2.csv")


# createAdaBoostClassifierModel()
print(predict_language(langage))
print(langage)
print(checkIfCode(langage))







