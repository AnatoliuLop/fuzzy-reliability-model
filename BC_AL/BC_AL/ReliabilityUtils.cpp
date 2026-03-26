#include "ReliabilityUtils.h"
#include <cmath>
#include <algorithm>
#include <iostream>

double ReliabilityUtils::computeReliability(
    const FuzzyReliabilityModel& model,
    const std::vector<std::vector<int>>& features,
    const std::vector<int>& labels,
    int index,
    ReliabilityMode mode)
{
    switch (mode) {
    case ReliabilityMode::Base:
        return baselineReliability(model, features, labels, index);
    case ReliabilityMode::Retrained:
        return retrainedReliability(model, features, labels, index);
    case ReliabilityMode::BatchedEnd:
        return batchedEndReliability(model, features, labels, index, int(0.1 * features.size()));
    default:
        return baselineReliability(model, features, labels, index);
    }
}
double ReliabilityUtils::baselineReliability(
    const FuzzyReliabilityModel& model,
    const std::vector<std::vector<int>>& features,
    const std::vector<int>& labels,
    int index)
{
    return model.EvaluateReliability(features[index], labels[index]);
}

double ReliabilityUtils::retrainedReliability(
    const FuzzyReliabilityModel& baseModel,
    const std::vector<std::vector<int>>& features,
    const std::vector<int>& labels,
    int index)
{
    if (index == 0) return 1.0;

    FuzzyReliabilityModel retrainedModel(
        baseModel.getDomainSizes(),
        baseModel.getDependentPairs(),
        baseModel.getNumClasses(),
        baseModel.getAlpha(),
        baseModel.getBeta(),
        baseModel.getGamma(),
        baseModel.getLambda()
    );
    std::vector<std::vector<int>> partialF(features.begin(), features.begin() + index);
    std::vector<int> partialL(labels.begin(), labels.begin() + index);
    retrainedModel.TrainOnline(partialF, partialL);

    

    return retrainedModel.EvaluateReliability(features[index], labels[index]);
}

double ReliabilityUtils::batchedEndReliability(
    const FuzzyReliabilityModel& baseModel,
    const std::vector<std::vector<int>>& features,
    const std::vector<int>& labels,
    int index,
    int batchSize)
{
    int total = int(features.size());
    if (index < total - batchSize)
        return baseModel.EvaluateReliability(features[index], labels[index]);

    int start = std::max(0, total - batchSize);
    std::vector<std::vector<int>> lastF(features.begin() + start, features.end());
    std::vector<int> lastL(labels.begin() + start, labels.end());

    FuzzyReliabilityModel retrainedModel(
        baseModel.getDomainSizes(),
        baseModel.getDependentPairs(),
        baseModel.getNumClasses(),
        baseModel.getAlpha(),
        baseModel.getBeta(),
        baseModel.getGamma(),
        baseModel.getLambda()
    );
    retrainedModel.TrainOnline(lastF, lastL);

    return retrainedModel.EvaluateReliability(features[index], labels[index]);
}

std::vector<double> ReliabilityUtils::getReliabilityScores(
    const FuzzyReliabilityModel& model,
    const std::vector<std::vector<int>>& features,
    const std::vector<int>& labels,
    ReliabilityMode mode)
{
    std::vector<double> scores;
    for (size_t i = 0; i < features.size(); ++i) {
        double r = computeReliability(model, features, labels, int(i), mode);
        scores.push_back(r);
    }
    return scores;
}
