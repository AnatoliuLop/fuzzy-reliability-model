#pragma once
#include <string>
#include <vector>
#include <unordered_map>

void getConfig(int choice,
    std::string& filename,
    int& numAttributes,
    int& numClasses,
    std::vector<std::unordered_map<std::string, int>>& attributeMaps,
    std::vector<std::vector<std::string>>& attributeVals,
    std::vector<std::string>& attributeNames,
    std::unordered_map<std::string, int>& classMap,
    std::vector<std::string>& classVals,
    std::vector<int>& domainSizes,
    std::vector<std::pair<int, int>>& dependentPairs);
