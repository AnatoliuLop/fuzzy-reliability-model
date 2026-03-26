#include "Config.h"
#include "FuzzyReliabilityModel.h"
#include "OutputCSV.h"
#include "ReliabilityUtils.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>

using namespace std;

struct Hyperparams {
    double alpha, beta, gamma, lambda;
};

// Метрики трёх режимов
struct Metrics {
    double acc_baseline, avgR_baseline;
    double acc_retrained, avgR_retrained;
    double acc_batched, avgR_batched;
};

// === 1) Grid Search: 5-fold 
Hyperparams gridSearch(
    const vector<vector<int>>& features,
    const vector<int>& labels,
    const vector<int>& domainSizes,
    int numClasses,
    const vector<pair<int, int>>& dependentPairs)
{
    vector<double> alphas = { 0.1,0.3,0.5 };
    vector<double> betas = { 0.5,0.7,0.9 };
    vector<double> gammas = { 0.5,0.7,0.9 };
    vector<double> lambdas = { 1.0,2.0,5.0 };

    int N = features.size();
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    mt19937_64 rng(42);
    shuffle(idx.begin(), idx.end(), rng);

    int K = 5;
    int foldSize = N / K;
    Hyperparams bestHP{ 0,0,0,0 };
    double bestScore = -1e9;

    for (double a : alphas)
        for (double b : betas)
            for (double g : gammas)
                for (double l : lambdas) {
                    double sumAcc = 0;
                    for (int fold = 0; fold < K; ++fold) {
                        vector<vector<int>> tF, vF;
                        vector<int> tL, vL;
                        int start = fold * foldSize, end = (fold + 1) * foldSize;
                        for (int i = 0; i < N; ++i) {
                            int id = idx[i];
                            if (i >= start && i < end) {
                                vF.push_back(features[id]);
                                vL.push_back(labels[id]);
                            }
                            else {
                                tF.push_back(features[id]);
                                tL.push_back(labels[id]);
                            }
                        }
                        FuzzyReliabilityModel m(
                            domainSizes, dependentPairs, numClasses,
                            a, b, g, l
                        );
                        m.TrainOnline(tF, tL);
                        int correct = 0;
                        for (int i = 0; i < (int)vF.size(); ++i)
                            if (m.Classify(vF[i]) == vL[i]) correct++;
                        sumAcc += double(correct) / vF.size();
                    }
                    double avg = sumAcc / K;
                    if (avg > bestScore) {
                        bestScore = avg;
                        bestHP = { a,b,g,l };
                    }
                }

    cout << "Grid Search best ACC=" << bestScore
        << " for (a,b,g,l)=("
        << bestHP.alpha << "," << bestHP.beta << ","
        << bestHP.gamma << "," << bestHP.lambda << ")\n";
    return bestHP;
}

void loadData(
    int choice,
    const string& filename,
    int numAttributes,
    const vector<unordered_map<string, int>>& attributeMaps,
    const unordered_map<string, int>& classMap,
    vector<vector<int>>& features,
    vector<int>& labels
) {
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Cannot open " << filename << "\n";
        exit(1);
    }
    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        istringstream iss(line);
        vector<string> tok;
        string t;
        while (getline(iss, t, ',')) tok.push_back(t);
        if ((int)tok.size() != numAttributes + 1) continue;

        vector<int> inst(numAttributes);
        string labelStr;
        if (choice == 0) {
            labelStr = tok[0];
            for (int i = 0; i < numAttributes; ++i)
                inst[i] = attributeMaps[i].at(tok[i + 1]);
        }
        else {
            for (int i = 0; i < numAttributes; ++i)
                inst[i] = attributeMaps[i].at(tok[i]);
            labelStr = tok[numAttributes];
        }
        auto it = classMap.find(labelStr);
        if (it == classMap.end()) continue;
        features.push_back(inst);
        labels.push_back(it->second);
    }
}

void shuffleAllData(
    vector<vector<int>>& features,
    vector<int>& labels
) {
    int N = features.size();
    vector<pair<vector<int>, int>> zipped;
    zipped.reserve(N);
    for (int i = 0; i < N; ++i)
        zipped.emplace_back(features[i], labels[i]);
    mt19937_64 rng(42);
    shuffle(zipped.begin(), zipped.end(), rng);
    for (int i = 0; i < N; ++i) {
        features[i] = move(zipped[i].first);
        labels[i] = zipped[i].second;
    }
}

FuzzyReliabilityModel trainFinalModel(
    const vector<vector<int>>& features,
    const vector<int>& labels,
    const vector<int>& domainSizes,
    const vector<pair<int, int>>& dependentPairs,
    double alpha, double beta,
    double gamma, double lambda
) {
    int numClasses = *max_element(labels.begin(), labels.end()) + 1;
    FuzzyReliabilityModel model(
        domainSizes, dependentPairs, numClasses,
        alpha, beta, gamma, lambda
    );
    model.TrainOnline(features, labels);
    return model;
}

Metrics evaluateAll(
    const FuzzyReliabilityModel& mdl,
    const vector<vector<int>>& features,
    const vector<int>& labels,
    double alpha, double beta,
    double gamma, double lambda
) {
    int N = features.size();
    int bat = int(0.1 * N);
    int ok0 = 0, ok1 = 0, ok2 = 0;
    double s0 = 0, s1 = 0, s2 = 0;

    for (int i = 0; i < N; ++i) {
        // baseline
        if (mdl.Classify(features[i]) == labels[i]) ok0++;
        double r0 = mdl.EvaluateReliability(features[i], labels[i]);
        s0 += r0;
        // retrained 
        FuzzyReliabilityModel m1 = mdl;
        if (i > 0) {
            m1.TrainOnline(
                vector<vector<int>>(features.begin(), features.begin() + i),
                vector<int>(labels.begin(), labels.begin() + i)
            );
        }
        if (m1.Classify(features[i]) == labels[i]) ok1++;
        s1 += m1.EvaluateReliability(features[i], labels[i]);
        // batched-end
        if (i < N - bat) {
            if (mdl.Classify(features[i]) == labels[i]) ok2++;
            s2 += r0;
        }
        else {
            FuzzyReliabilityModel m2 = mdl;
            m2.TrainOnline(
                vector<vector<int>>(features.begin() + (N - bat), features.end()),
                vector<int>(labels.begin() + (N - bat), labels.end())
            );
            if (m2.Classify(features[i]) == labels[i]) ok2++;
            s2 += m2.EvaluateReliability(features[i], labels[i]);
        }
    }

    return Metrics{
        100.0 * ok0 / N, s0 / N,
        100.0 * ok1 / N, s1 / N,
        100.0 * ok2 / N, s2 / N
    };
}

void printMetrics(const Metrics& m, const string& title) {
    cout << fixed << setprecision(1)
        << "\n" << title << "\n"
        << "Mode         Acc(%)  Avg R\n"
        << "baseline     " << m.acc_baseline << "     " << m.avgR_baseline << "\n"
        << "retrained    " << m.acc_retrained << "     " << m.avgR_retrained << "\n"
        << "batched-end  " << m.acc_batched << "     " << m.avgR_batched << "\n";
}

// === main ===
int main() {
    int scenario;
    cout << "Scenario (0=Original, 1=Enhanced): ";
    cin >> scenario;

    int choice;
    cout << "Select dataset (0=balance, 1=car): ";
    cin >> choice;
    string filename;
    int numAttributes, numClasses;
    vector<unordered_map<string, int>> attributeMaps;
    vector<vector<string>> attributeVals;
    vector<string> attributeNames, classVals;
    unordered_map<string, int> classMap;
    vector<int> domainSizes;
    vector<pair<int, int>> dependentPairs;
    getConfig(
        choice, filename,
        numAttributes, numClasses,
        attributeMaps, attributeVals, attributeNames,
        classMap, classVals, domainSizes, dependentPairs
    );
    auto configDeps = dependentPairs;

    vector<vector<int>> features;
    vector<int> labels;
    loadData(
        choice, filename,
        numAttributes,
        attributeMaps,
        classMap,
        features,
        labels
    );
    if (features.empty()) {
        cerr << "Empty dataset\n";
        return 1;
    }

    double alpha = 0.3, beta = 0.7, gamma = 0.9, lambda = 2.0;
    if (scenario == 1) {
        auto dyn = FuzzyReliabilityModel::computeDependentPairs(features, 0.005);
        dependentPairs = dyn.empty() ? configDeps : dyn;
        cout << "Using pairs: ";
        for (auto& p : dependentPairs) cout << "(" << p.first << "," << p.second << ") ";
        cout << "\n";
        Hyperparams hp = gridSearch(features, labels, domainSizes, numClasses, dependentPairs);

        alpha = hp.alpha;
        beta = hp.beta;
        gamma = hp.gamma;
        lambda = hp.lambda;
    }
    else {
        dependentPairs = configDeps;
        cout << "Using config pairs only\n";
    }

    int doShuffle;
    cout << "Shuffle data before train/test? (0=no,1=yes): ";
    cin >> doShuffle;
    if (doShuffle) shuffleAllData(features, labels);

    auto model = trainFinalModel(
        features, labels,
        domainSizes, dependentPairs,
        alpha, beta, gamma, lambda
    );

    Metrics met = evaluateAll(
        model, features, labels,
        alpha, beta, gamma, lambda
    );
    printMetrics(met, scenario == 0 ? "Original model" : "Enhanced model");

    OutputCSV csv;
    csv.writeRelationScheme("table1_RelationScheme.csv",
        attributeNames, attributeVals, classVals);
    csv.writeAttributeCoding("table2_AttributeCoding.csv",
        attributeNames, attributeVals);

    csv.writeHighestInputWeights("table3_topSingleWeights.csv",
        model.topSingleWeights(10, attributeNames, attributeVals, classVals));
    csv.writeHighestPairWeights("table4_topPairWeights.csv",
        model.topPairWeights(10, attributeNames, attributeVals, classVals));

    int N = features.size(), batchedSize = int(0.1 * N);
    vector<EvalRecord> baseEval, retrainEval, batchedEval;
    for (int i = 0; i < N; ++i) {
        vector<string> featStr;
        for (int j = 0; j < numAttributes; ++j)
            featStr.push_back(attributeVals[j][features[i][j]]);
        int actual = labels[i];

        // Base
        int pb = model.Classify(features[i]);
        double rb = model.EvaluateReliability(features[i], actual);
        baseEval.push_back({ i + 1,featStr,classVals[actual],classVals[pb],rb });

        // Retrained
        FuzzyReliabilityModel m1 = model;
        if (i > 0) m1.TrainOnline(
            vector<vector<int>>(features.begin(), features.begin() + i),
            vector<int>(labels.begin(), labels.begin() + i)
        );
        int pr = m1.Classify(features[i]);
        double rr = m1.EvaluateReliability(features[i], actual);
        retrainEval.push_back({ i + 1,featStr,classVals[actual],classVals[pr],rr });

        // Batched-end
        int px; double rx;
        if (i < N - batchedSize) { px = pb; rx = rb; }
        else {
            FuzzyReliabilityModel m2 = model;
            m2.TrainOnline(
                vector<vector<int>>(features.begin() + (N - batchedSize), features.end()),
                vector<int>(labels.begin() + (N - batchedSize), labels.end())
            );
            px = m2.Classify(features[i]);
            rx = m2.EvaluateReliability(features[i], actual);
        }
        batchedEval.push_back({ i + 1,featStr,classVals[actual],classVals[px],rx });
    }
    csv.writeEvaluationResults("table7_base.csv", baseEval);
    csv.writeEvaluationResults("table7_retrained.csv", retrainEval);
    csv.writeEvaluationResults("table7_batched.csv", batchedEval);

    // low-confidence 
    const double threshold = 10.0; // %
    auto writeLow = [&](auto& ev, const string& fn, const string& tag) {
        vector<EvalRecord> low;
        for (auto& r : ev)
            if ((2.0 - r.reliability) * 100.0 < threshold)
                low.push_back(r);
        if (!low.empty()) {
            csv.writeEvaluationResults(fn, low);
            cout << tag << " low-confidence -> " << fn << "\n";
        }
        };
    writeLow(baseEval, "table5_low_base.csv", "Base");
    writeLow(retrainEval, "table5_low_retrained.csv", "Retrained");
    writeLow(batchedEval, "table5_low_batched.csv", "Batched");

    csv.writeCombinedEvaluationResults(
        "table7_all.csv", baseEval, retrainEval, batchedEval
    );

    vector<string> actualLabels, predictedLabels;
    vector<double> relB, relR, relT;
    actualLabels.reserve(N);
    predictedLabels.reserve(N);
    relB.reserve(N);
    relR.reserve(N);
    relT.reserve(N);
    for (int i = 0; i < N; ++i) {
        actualLabels.push_back(baseEval[i].actualClass);
        predictedLabels.push_back(baseEval[i].predictedClass);
        relB.push_back(baseEval[i].reliability);
        relR.push_back(retrainEval[i].reliability);
        relT.push_back(batchedEval[i].reliability);
    }
    csv.writeReliabilityComparison(
        "table6_reliability_comparison.csv",
        actualLabels, predictedLabels,
        relB, relR, relT
    );
    cout << "All CSV tables written\n";

    return 0;
}
