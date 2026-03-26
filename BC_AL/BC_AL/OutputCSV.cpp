
#include "OutputCSV.h"
#include <fstream>
#include <iomanip>

// Table 1
void OutputCSV::writeRelationScheme(
    const std::string& fn,
    const std::vector<std::string>& attributeNames,
    const std::vector<std::vector<std::string>>& attributeVals,
    const std::vector<std::string>& classVals)
{
    std::ofstream f(fn);
    f << "Attribute name;Attribute description;Domain;Classification\n";
    for (size_t i = 0; i < attributeNames.size(); ++i) {
        f << "A" << i << ";"
            << attributeNames[i] << ";";
        for (size_t j = 0; j < attributeVals[i].size(); ++j) {
            f << attributeVals[i][j];
            if (j + 1 < attributeVals[i].size()) f << ",";
        }
        f << ";Input\n";
    }
    f << "A" << attributeNames.size() << ";"
        << "class"
        << ";";
    for (size_t j = 0; j < classVals.size(); ++j) {
        f << classVals[j];
        if (j + 1 < classVals.size()) f << ",";
    }
    f << ";Target\n";
}


// Table 2
void OutputCSV::writeAttributeCoding(
    const std::string& fn,
    const std::vector<std::string>& attributeNames,
    const std::vector<std::vector<std::string>>& attributeVals)
{
    std::ofstream f(fn);
    f << "Attribute;";
    for (int i = 0; i < 10; ++i) f << i << ";";
    f << "\n";

    for (size_t i = 0; i < attributeNames.size(); ++i) {
        f << attributeNames[i] << ";";
        for (size_t j = 0; j < attributeVals[i].size(); ++j) {
            f << attributeVals[i][j] << ";";
        }
        f << "\n";
    }
}



// Table 3
void OutputCSV::writeHighestInputWeights(
    const std::string& fn,
    const std::vector<ConnectionWeight>& data)
{
    std::ofstream f(fn);
    f << "Input attribute;Value;Target attribute;Connection weight;;Interpretation\n";
    for (auto& r : data) {
        f << r.inputName << ";"
            << r.inputValue << ";"
            << r.targetValue << ";"
            << std::fixed << std::setprecision(3) << r.weight << ";"
            << r.interpretation << "\n";
    }
}

// Table 4
void OutputCSV::writeHighestPairWeights(
    const std::string& fn,
    const std::vector<PairConnectionWeight>& data)
{
    std::ofstream f(fn);
    f << "Input attribute;Value;Input attribute;Value;Target attribute;Value;Connection weight;Interpretation\n";
    for (auto& r : data) {
        f << r.input1 << ";" << r.value1 << ";"
            << r.input2 << ";" << r.value2 << ";"
            << r.targetValue << ";"
            << std::fixed << std::setprecision(3) << r.weight << ";"
            << r.interpretation << "\n";
    }
}

// Table 5
void OutputCSV::writeLowReliabilityBatches(
    const std::string& fn,
    const std::vector<BatchInfo>& data)
{
    std::ofstream f(fn);
    f << "ID;NW;Exp;Type;Size;Status;Priority;Oper;Actual time;Rel.;Pred. time\n";
    f << std::fixed << std::setprecision(1);
    for (auto& r : data) {
        double confidence = (2.0 - r.reliability) * 100.0;  
        f << r.ID << ";" << r.NW << ";"
          << r.Exp << ";" << r.Type << ";" << r.Size << ";"
          << r.Status << ";" << r.Priority << ";"
          << r.Operation << ";" << r.actualTime << ";"
          << confidence << "%;"  
          << r.predictedTime << "\n";
    }
}

void OutputCSV::writeEvaluationResults(
    const std::string& fn,
    const std::vector<EvalRecord>& data)
{
    std::ofstream f(fn);
    f << "ID;";
    for (size_t i = 0; i < data[0].features.size(); ++i)
        f << "Attr" << i + 1 << ";";
    f << "Actual;Predicted;Reliability(%)\n";

    for (const auto& r : data) {
        f << r.ID << ";";
        for (const auto& val : r.features)
            f << val << ";";
        f << r.actualClass << ";"
            << r.predictedClass << ";"
            << std::fixed << std::setprecision(1)
            << (2.0 - r.reliability) * 100.0 << "%\n";
    }
}

void OutputCSV::writeReliabilityComparison(
    const std::string& filename,
    const std::vector<std::string>& actualLabels,
    const std::vector<std::string>& predictedLabels,
    const std::vector<double>& relBase,
    const std::vector<double>& relRetrained,
    const std::vector<double>& relBatched)
{
    std::ofstream fout(filename);
    fout << "Index;Actual;Predicted;BaseReliability;RetrainedReliability;BatchedEndReliability\n";

    for (size_t i = 0; i < actualLabels.size(); ++i) {
        fout << (i + 1) << ";"
            << actualLabels[i] << ";"
            << predictedLabels[i] << ";"
            << relBase[i] << ";"
            << relRetrained[i] << ";"
            << relBatched[i] << "\n";
    }
    fout.close();
}
void OutputCSV::writeCombinedEvaluationResults(
    const std::string& filename,
    const std::vector<EvalRecord>& base,
    const std::vector<EvalRecord>& retrained,
    const std::vector<EvalRecord>& batched)
{
    std::ofstream f(filename);
    f << "ID;";
    for (size_t i = 0; i < base[0].features.size(); ++i)
        f << "Attr" << i + 1 << ";";
    f << "Actual;";
    f << "PredictedBase;PredictedRetrained;PredictedBatched;";
    f << "ReliabilityBase(%);ReliabilityRetrained(%);ReliabilityBatched(%)\n";

    for (size_t i = 0; i < base.size(); ++i) {
        f << base[i].ID << ";";
        for (const auto& val : base[i].features)
            f << val << ";";

        f << base[i].actualClass << ";"
            << base[i].predictedClass << ";"     // Base
            << retrained[i].predictedClass << ";"  // Retrained
            << batched[i].predictedClass << ";";   // Batched

        f << std::fixed << std::setprecision(1)
            << (2.0 - base[i].reliability) * 100.0 << "%;"
            << (2.0 - retrained[i].reliability) * 100.0 << "%;"
            << (2.0 - batched[i].reliability) * 100.0 << "%\n";
    }
}


