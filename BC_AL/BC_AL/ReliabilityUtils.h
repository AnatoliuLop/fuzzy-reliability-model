#pragma once
#include "FuzzyReliabilityModel.h"
#include <vector>

enum class ReliabilityMode {
    Base,
    Retrained,
    BatchedEnd
};

class ReliabilityUtils {
public:
    static double computeReliability(
        const FuzzyReliabilityModel& model,
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels,
        int index,
        ReliabilityMode mode
    );
    static std::vector<double> getReliabilityScores(
        const FuzzyReliabilityModel& model,
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels,
        ReliabilityMode mode
    );
private:
    static double baselineReliability(
        const FuzzyReliabilityModel& model,
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels,
        int index
    );

    static double retrainedReliability(
        const FuzzyReliabilityModel& baseModel,
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels,
        int index
    );

    static double batchedEndReliability(
        const FuzzyReliabilityModel& baseModel,
        const std::vector<std::vector<int>>& features,
        const std::vector<int>& labels,
        int index,
        int batchSize
    );

};
