#ifndef FUZZY_RELIABILITY_MODEL_H
#define FUZZY_RELIABILITY_MODEL_H

#include <vector>
#include <utility>
#include <string>
#include "OutputCSV.h"   

class FuzzyReliabilityModel {
public:
    FuzzyReliabilityModel(const std::vector<int>& domSizes,
        const std::vector<std::pair<int, int>>& dependentPairs,
        int numClasses,
        double alpha,    // (7.3.2a)
        double beta,     // (7.3.2b)
        double gamma,    // (7.3.2c)
        double lambda);  

    
    void TrainOnline(const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels);
  
    void UpdateOnline(const std::vector<int>& instance, int label);

    int Classify(const std::vector<int>& instance) const;

    double EvaluateReliability(const std::vector<int>& instance,
        int actualClass) const;


    double getSingleWeight(int i, int x, int j) const {
        return singleWeights[i][x][j];
    }
    double getPairWeight(int i1, int x1, int i2, int x2, int j) const {
        return pairWeights[i1][x1][i2][x2][j];
    }
    std::vector<int> getDomainSizes() const;
    std::vector<std::pair<int, int>> getDependentPairs() const;
    int getNumClasses() const;
    double getAlpha() const;
    double getBeta() const;
    double getGamma() const;
    double getLambda() const;

    std::vector<ConnectionWeight>
        topSingleWeights(int k,
            const std::vector<std::string>& attributeNames,
            const std::vector<std::vector<std::string>>& attributeVals,
            const std::vector<std::string>& classVals) const;

    std::vector<PairConnectionWeight>
        topPairWeights(int k,
            const std::vector<std::string>& attributeNames,
            const std::vector<std::vector<std::string>>& attributeVals,
            const std::vector<std::string>& classVals) const;

    std::vector<BatchInfo>
        lowReliabilityBatches(const std::vector<std::vector<int>>& features,
            const std::vector<int>& labels,
            double threshold) const;

   

    static std::vector<std::pair<int, int>>
        computeDependentPairs(const std::vector<std::vector<int>>& features,
            double miThreshold);

private:
    int numInputs;    
    int numClasses;   
    double alpha, beta, gamma, lambda;
    std::vector<std::pair<int, int>> dependentPairs;

    
    int totalExamples;       
    std::vector<int> targetCounts;  
    
    std::vector<std::vector<std::vector<int>>> conditionalCounts;
    
    std::vector<
        std::vector<
        std::vector<
        std::vector<
        std::vector<int>
        >
        >
        >
    > pairCounts;

    
    std::vector<double> priorProbs;   // P2k(V=j)
    std::vector<std::vector<std::vector<double>>> condProbs;  // P2k(V=j|i,x)
    std::vector<
        std::vector<
        std::vector<
        std::vector<
        std::vector<double>
        >
        >
        >
    > pairProbs;  // P2k(V=j|i1,x1,i2,x2)

    std::vector<double> bias;  // w_j = log P2k(V=j)
    // singleWeights[i][x][j] = log(P2k(V=j|i,x)/P2k(V=j))
    std::vector<std::vector<std::vector<double>>> singleWeights;
    // pairWeights[i1][x1][i2][x2][j] = excess mutual info
    std::vector<
        std::vector<
        std::vector<
        std::vector<
        std::vector<double>
        >
        >
        >
    > pairWeights;


    double safeLog(double v) const;  

   
    void interpPriors();
    // P2k(V=j|i,x) — 7.3.2b
    void interpConditionals();
    // P2k(V=j|i1,x1,i2,x2) — 7.3.2c
    void interpPairConditionals();

    // bias[j]=log P2k(V=j)
    void computeBias();
    
    void computeSingleWeights();
   
    void computePairWeights();
};

#endif // FUZZY_RELIABILITY_MODEL_H