#include "FuzzyReliabilityModel.h"
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <map>
#include <unordered_map>


FuzzyReliabilityModel::FuzzyReliabilityModel(const std::vector<int>& domSizes,
    const std::vector<std::pair<int, int>>& depPairs,
    int nClasses,
    double a, double b, double g, double l)
    : numInputs(int(domSizes.size())),
    numClasses(nClasses),
    alpha(a), beta(b), gamma(g), lambda(l),
    dependentPairs(depPairs),
    totalExamples(0)
{
    targetCounts.assign(numClasses, 0);

   
    conditionalCounts.resize(numInputs);
    for (int i = 0; i < numInputs; ++i) {
        conditionalCounts[i].assign(domSizes[i],
            std::vector<int>(numClasses, 0));
    }

    pairCounts.resize(numInputs);
    for (int i1 = 0; i1 < numInputs; ++i1) {
        pairCounts[i1].resize(domSizes[i1]);
        for (int x1 = 0; x1 < domSizes[i1]; ++x1) {
            pairCounts[i1][x1].resize(numInputs);
            for (int i2 = 0; i2 < numInputs; ++i2) {
                pairCounts[i1][x1][i2].assign(
                    domSizes[i2],
                    std::vector<int>(numClasses, 0)
                );
            }
        }
    }

    priorProbs.assign(numClasses, 1.0 / numClasses);

    condProbs.resize(numInputs);
    for (int i = 0; i < numInputs; ++i) {
        condProbs[i].assign(domSizes[i],
            std::vector<double>(numClasses, 1.0 / numClasses));
    }

    pairProbs.resize(numInputs);
    for (int i1 = 0; i1 < numInputs; ++i1) {
        pairProbs[i1].resize(domSizes[i1]);
        for (int x1 = 0; x1 < domSizes[i1]; ++x1) {
            pairProbs[i1][x1].resize(numInputs);
            for (int i2 = 0; i2 < numInputs; ++i2) {
                pairProbs[i1][x1][i2].assign(
                    domSizes[i2],
                    std::vector<double>(numClasses, 1.0 / numClasses)
                );
            }
        }
    }

    bias.assign(numClasses, 0.0);

    singleWeights.resize(numInputs);
    for (int i = 0; i < numInputs; ++i) {
        singleWeights[i].assign(domSizes[i],
            std::vector<double>(numClasses, 0.0));
    }

    pairWeights.resize(numInputs);
    for (int i1 = 0; i1 < numInputs; ++i1) {
        pairWeights[i1].resize(domSizes[i1]);
        for (int x1 = 0; x1 < domSizes[i1]; ++x1) {
            pairWeights[i1][x1].resize(numInputs);
            for (int i2 = 0; i2 < numInputs; ++i2) {
                pairWeights[i1][x1][i2].assign(
                    domSizes[i2],
                    std::vector<double>(numClasses, 0.0)
                );
            }
        }
    }
}

std::vector<int> FuzzyReliabilityModel::getDomainSizes() const {
    std::vector<int> sizes;
    for (const auto& attr : conditionalCounts) {
        sizes.push_back(int(attr.size()));
    }
    return sizes;
}
std::vector<std::pair<int, int>> FuzzyReliabilityModel::getDependentPairs() const { return dependentPairs; }
int FuzzyReliabilityModel::getNumClasses() const { return numClasses; }
double FuzzyReliabilityModel::getAlpha() const { return alpha; }
double FuzzyReliabilityModel::getBeta() const { return beta; }
double FuzzyReliabilityModel::getGamma() const { return gamma; }
double FuzzyReliabilityModel::getLambda() const { return lambda; }



void FuzzyReliabilityModel::TrainOnline(const std::vector<std::vector<int>>& F,
    const std::vector<int>& L)
{
    assert(F.size() == L.size());


    totalExamples = 0;
    std::fill(targetCounts.begin(), targetCounts.end(), 0);
    for (int i = 0; i < numInputs; ++i)
        for (auto& vec : conditionalCounts[i])
            std::fill(vec.begin(), vec.end(), 0);
    for (int i1 = 0; i1 < numInputs; ++i1)
        for (int x1 = 0; x1 < conditionalCounts[i1].size(); ++x1)
            for (int i2 = 0; i2 < numInputs; ++i2)
                for (auto& vec : pairCounts[i1][x1][i2])
                    std::fill(vec.begin(), vec.end(), 0);

    priorProbs.assign(numClasses, 1.0 / numClasses);

    for (size_t k = 0; k < F.size(); ++k) {
        UpdateOnline(F[k], L[k]);
    }

    
   
}


void FuzzyReliabilityModel::UpdateOnline(const std::vector<int>& inst, int label)
{
    
    ++totalExamples;
    ++targetCounts[label];
    for (int i = 0; i < numInputs; ++i) {
        int x = inst[i];
        ++conditionalCounts[i][x][label];
    }
    for (auto& pr : dependentPairs) {
        int i1 = pr.first, i2 = pr.second;
        int x1 = inst[i1], x2 = inst[i2];
        ++pairCounts[i1][x1][i2][x2][label];
        ++pairCounts[i2][x2][i1][x1][label];
    }

    // 2) interpPriors  — P2k(V=j) = (1−α)P2k−1 + α·P1k    (7.3.2a)
    interpPriors();

    // 3) interpConditionals — P2k(V|i,x)           (7.3.2b)
    interpConditionals();

    // 4) interpPairConditionals — парные  (7.3.2c)
    interpPairConditionals();

    // 5) bias[j] = log P2k(V=j)                     (w_j in 7.3.2d)
    computeBias();

    // 6) singleWeights[i][x][j] = log P2k(V|i,x)−logP2k(V)
    computeSingleWeights();

    // 7) pairWeights = excess mutual information     (7.3.2d)
    computePairWeights();
}


void FuzzyReliabilityModel::interpPriors()
{
    double k = double(totalExamples);
    for (int j = 0; j < numClasses; ++j) {
        double p1 = double(targetCounts[j]) / k;               // P1k(V=j)
        priorProbs[j] = (1.0 - alpha) * priorProbs[j] + alpha * p1; // P2k(V=j)
    }
}

// 3) 7.3.2b
void FuzzyReliabilityModel::interpConditionals()
{
    for (int i = 0; i < numInputs; ++i) {
        for (int x = 0; x < conditionalCounts[i].size(); ++x) {
            int sum = 0;
            for (int j = 0; j < numClasses; ++j)
                sum += conditionalCounts[i][x][j];
            for (int j = 0; j < numClasses; ++j) {
                double p1 = sum > 0
                    ? double(conditionalCounts[i][x][j]) / sum
                    : 1.0 / numClasses;
                // P2k(V|i,x) = (1−β)·P2k(V) + β·P1k(V|i,x)
                condProbs[i][x][j]
                    = (1.0 - beta) * priorProbs[j]
                    + beta * p1;
            }
        }
    }
}

void FuzzyReliabilityModel::interpPairConditionals()
{
    for (auto& pr : dependentPairs) {
        int i1 = pr.first, i2 = pr.second;
        for (int x1 = 0; x1 < conditionalCounts[i1].size(); ++x1)
            for (int x2 = 0; x2 < conditionalCounts[i2].size(); ++x2) {
                int sum = 0;
                for (int j = 0; j < numClasses; ++j)
                    sum += pairCounts[i1][x1][i2][x2][j];
                for (int j = 0; j < numClasses; ++j) {
                    double p1 = sum > 0
                        ? double(pairCounts[i1][x1][i2][x2][j]) / sum
                        : 1.0 / numClasses;
                    double pu1 = condProbs[i1][x1][j];
                    double pu2 = condProbs[i2][x2][j];
                    double prod = priorProbs[j] > 0
                        ? pu1 * pu2 / priorProbs[j]
                        : 0.0;
                    // P2k(pair) = (1−γ)·prod + γ·p1
                    pairProbs[i1][x1][i2][x2][j]
                        = (1.0 - gamma) * prod + gamma * p1;
                    pairProbs[i2][x2][i1][x1][j]
                        = pairProbs[i1][x1][i2][x2][j];
                }
            }
    }
}

// 5) bias[j] = log P2k(V=j)
void FuzzyReliabilityModel::computeBias()
{
    for (int j = 0; j < numClasses; ++j)
        bias[j] = safeLog(priorProbs[j]);
}

// 6) w_{j;i x} = log P2k(V|i,x) − log P2k(V)
void FuzzyReliabilityModel::computeSingleWeights()
{
    for (int i = 0; i < numInputs; ++i)
        for (int x = 0; x < condProbs[i].size(); ++x)
            for (int j = 0; j < numClasses; ++j) {
                singleWeights[i][x][j]
                    = safeLog(condProbs[i][x][j])
                    - safeLog(priorProbs[j]);
            }
}

void FuzzyReliabilityModel::computePairWeights()
{
    for (auto& pr : dependentPairs) {
        int i1 = pr.first, i2 = pr.second;
        for (int x1 = 0; x1 < condProbs[i1].size(); ++x1)
            for (int x2 = 0; x2 < condProbs[i2].size(); ++x2)
                for (int j = 0; j < numClasses; ++j) {
                    double p12 = pairProbs[i1][x1][i2][x2][j];
                    double p1 = condProbs[i1][x1][j];
                    double p2 = condProbs[i2][x2][j];
                    pairWeights[i1][x1][i2][x2][j]
                        = safeLog(p12)
                        - (safeLog(p1) + safeLog(p2));
                    pairWeights[i2][x2][i1][x1][j]
                        = pairWeights[i1][x1][i2][x2][j];
                }
    }
}

double FuzzyReliabilityModel::safeLog(double v) const {
    static const double EPS = 1e-12;
    return std::log(std::max(v, EPS));
}

int FuzzyReliabilityModel::Classify(const std::vector<int>& inst) const
{
    assert(int(inst.size()) == numInputs);
    std::vector<double> a(numClasses, 0.0);
    for (int j = 0; j < numClasses; ++j) {
        double s = bias[j];
        for (int i = 0; i < numInputs; ++i)
            s += singleWeights[i][inst[i]][j];
        for (auto& pr : dependentPairs) {
            int i1 = pr.first, i2 = pr.second;
            s += pairWeights[i1][inst[i1]]
                [i2][inst[i2]][j];
        }
        a[j] = s;
    }
    int best = 0;
    for (int j = 1; j < numClasses; ++j)
        if (a[j] > a[best]) best = j;
    return best;
}

double FuzzyReliabilityModel::EvaluateReliability(const std::vector<int>& inst,
    int actualClass) const
{
    std::vector<double> a(numClasses, 0.0);
    for (int j = 0; j < numClasses; ++j) {
        double s = bias[j];
        for (int i = 0; i < numInputs; ++i)
            s += singleWeights[i][inst[i]][j];
        for (auto& pr : dependentPairs) {
            int i1 = pr.first, i2 = pr.second;
            s += pairWeights[i1][inst[i1]]
                [i2][inst[i2]][j];
        }
        a[j] = s;
    }
    double maxA = *std::max_element(a.begin(), a.end());
    double d = maxA - a[actualClass];
    return 2.0 / (1.0 + std::exp(-lambda * d));
}

std::vector<ConnectionWeight>
FuzzyReliabilityModel::topSingleWeights(int k,
    const std::vector<std::string>& attributeNames,
    const std::vector<std::vector<std::string>>& attributeVals,
    const std::vector<std::string>& classVals) const
{
    struct Item { double w; int i, x, j; };
    std::vector<Item> all;
    for (int i = 0; i < numInputs; ++i)
        for (int x = 0; x < singleWeights[i].size(); ++x)
            for (int j = 0; j < numClasses; ++j)
                all.push_back({ singleWeights[i][x][j], i,x,j });
    std::sort(all.begin(), all.end(),
        [](auto& a, auto& b) {return a.w > b.w; });
    std::vector<ConnectionWeight> out;
    for (int idx = 0; idx < std::min(k, (int)all.size()); ++idx) {
        auto& it = all[idx];
        ConnectionWeight cw;
        cw.inputName = attributeNames[it.i];
        cw.inputValue = attributeVals[it.i][it.x];
        cw.targetName = classVals[it.j];
        cw.targetValue = classVals[it.j];
        cw.weight = it.w;
        cw.interpretation = "w_j;i=" + attributeNames[it.i]
            + "=" + attributeVals[it.i][it.x]
            + " → class=" + classVals[it.j];
        out.push_back(cw);
    }
    return out;
}

std::vector<PairConnectionWeight>
FuzzyReliabilityModel::topPairWeights(int k,
    const std::vector<std::string>& attributeNames,
    const std::vector<std::vector<std::string>>& attributeVals,
    const std::vector<std::string>& classVals) const
{
    struct Item { double w; int i1, x1, i2, x2, j; };
    std::vector<Item> all;
    for (auto& pr : dependentPairs) {
        int i1 = pr.first, i2 = pr.second;
        for (int x1 = 0; x1 < pairWeights[i1].size(); ++x1)
            for (int x2 = 0; x2 < pairWeights[i1][x1][i2].size(); ++x2)
                for (int j = 0; j < numClasses; ++j)
                    all.push_back({ pairWeights[i1][x1][i2][x2][j],
                                    i1,x1,i2,x2,j });
    }
    std::sort(all.begin(), all.end(),
        [](auto& a, auto& b) {return a.w > b.w; });
    std::vector<PairConnectionWeight> out;
    for (int idx = 0; idx < std::min(k, (int)all.size()); ++idx) {
        auto& it = all[idx];
        PairConnectionWeight pw;
        pw.input1 = attributeNames[it.i1];
        pw.value1 = attributeVals[it.i1][it.x1];
        pw.input2 = attributeNames[it.i2];
        pw.value2 = attributeVals[it.i2][it.x2];
        pw.targetName = classVals[it.j];
        pw.targetValue = classVals[it.j];
        pw.weight = it.w;
        pw.interpretation = "w_j;i1=" + attributeNames[it.i1]
            + "=" + attributeVals[it.i1][it.x1]
            + "; i2=" + attributeNames[it.i2]
            + "=" + attributeVals[it.i2][it.x2];
        out.push_back(pw);
    }
    return out;
}

std::vector<BatchInfo>
FuzzyReliabilityModel::lowReliabilityBatches(const std::vector<std::vector<int>>& F,
    const std::vector<int>& L,
    double threshold) const
{
    std::vector<BatchInfo> out;
    for (int t = 0; t < F.size(); ++t) {
        double rel = EvaluateReliability(F[t], L[t]);
        if (rel < threshold) {
            BatchInfo bi;
            bi.ID = t;
            bi.NW = F[t][0];
            bi.Exp = F[t][1];
            bi.Type = F[t][2];
            bi.Size = F[t][3];
            bi.Status = F[t][4];
            bi.Priority = F[t][5];
            bi.Operation = F[t][6];
            bi.actualTime = F[t][7];
            bi.reliability = rel;
            bi.predictedTime = Classify(F[t]);
            out.push_back(bi);
        }
    }

    return out;
}


std::vector<std::pair<int, int>>
FuzzyReliabilityModel::computeDependentPairs(
    const std::vector<std::vector<int>>& features,
    double eps)
{
    int N = features.size();
    int m = features[0].size();
    std::vector<std::pair<int, int>> deps;

    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            // Подсчёт частот
            std::unordered_map<int, int> cntX, cntY;
            std::unordered_map<long long, int> cntXY;
            for (auto& inst : features) {
                int xi = inst[i], yj = inst[j];
                cntX[xi]++; cntY[yj]++;
                long long key = (static_cast<long long>(xi) << 32) | (unsigned long long)yj;
                cntXY[key]++;
            }
            double mi = 0.0;
            for (auto& kv : cntXY) {
                int xi = int(kv.first >> 32);
                int yj = int(kv.first & 0xFFFFFFFF);
                double pxy = kv.second / double(N);
                double px = cntX[xi] / double(N);
                double py = cntY[yj] / double(N);
                mi += pxy * std::log(pxy / (px * py));
            }
            if (mi > eps) deps.emplace_back(i, j);
        }
    }
    return deps;
}