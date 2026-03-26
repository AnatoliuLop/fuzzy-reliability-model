#ifndef OUTPUT_CSV_H
#define OUTPUT_CSV_H

#include <vector>
#include <string>
#include <tuple>

struct ConnectionWeight {
    std::string inputName;
    std::string inputValue;
    std::string targetName;
    std::string targetValue;
    double weight;
    std::string interpretation;
};

struct PairConnectionWeight {
    std::string input1, value1;
    std::string input2, value2;
    std::string targetName;
    std::string targetValue;
    double weight;
    std::string interpretation;
};

struct BatchInfo {
    int ID;
    int NW;
    int Exp, Type, Size, Status, Priority, Operation;
    int actualTime;
    double reliability;
    int predictedTime;
};



struct EvalRecord {
    int ID;
    std::vector<std::string> features;
    std::string actualClass;
    std::string predictedClass;
    double reliability;
};

class OutputCSV {
public:
    // Table 1–6
    void writeRelationScheme(
        const std::string& fn,
        const std::vector<std::string>& attributeNames,
        const std::vector<std::vector<std::string>>& attributeVals,
        const std::vector<std::string>& classVals);

    void writeAttributeCoding(
        const std::string& fn,
        const std::vector<std::string>& attributeNames,
        const std::vector<std::vector<std::string>>& attributeVals);
    static void writeHighestInputWeights(const std::string& fn, const std::vector<ConnectionWeight>& data);
    static void writeHighestPairWeights(const std::string& fn, const std::vector<PairConnectionWeight>& data);
    static void writeLowReliabilityBatches(const std::string& fn, const std::vector<BatchInfo>& data);

    //  Table 7
    static void writeEvaluationResults(const std::string& fn, const std::vector<EvalRecord>& data);
  
    void writeReliabilityComparison(
        const std::string& filename,
        const std::vector<std::string>& actualLabels,
        const std::vector<std::string>& predictedLabels,
        const std::vector<double>& relBase,
        const std::vector<double>& relRetrained,
        const std::vector<double>& relBatched
    );
    void writeCombinedEvaluationResults(
        const std::string& filename,
        const std::vector<EvalRecord>& base,
        const std::vector<EvalRecord>& retrained,
        const std::vector<EvalRecord>& batched);
};

#endif // OUTPUT_CSV_H
