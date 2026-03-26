
#include "Config.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

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
    std::vector<std::pair<int, int>>& dependentPairs)
{
    
    static const std::vector<std::string> car_b = { "vhigh","high","med","low" };
    static const std::vector<std::string> car_d = { "2","3","4","5more" };
    static const std::vector<std::string> car_p = { "2","4","more" };
    static const std::vector<std::string> car_l = { "small","med","big" };
    static const std::vector<std::string> car_s = { "low","med","high" };
    static const std::vector<std::string> car_c = { "unacc","acc","good","vgood" };
    static const std::vector<std::string> carNames = { "buying","maint","doors","persons","lug_boot","safety" };

    
    static const std::vector<std::string> bs_vals = { "1","2","3","4","5" };
    static const std::vector<std::string> bs_c = { "L","B","R" };
    static const std::vector<std::string> bsNames = { "left_weight","left_distance","right_weight","right_distance" };


    if (choice == 1) {
        // Car
        filename = "data/car+evaluation_Stock/car.data";
        numAttributes = 6;
        numClasses = 4;

        attributeVals = { car_b, car_b, car_d, car_p, car_l, car_s };
        attributeNames = carNames;
        classVals = car_c;

        attributeMaps = {
          { {"vhigh",0}, {"high",1}, {"med",2}, {"low",3} },
          { {"vhigh",0}, {"high",1}, {"med",2}, {"low",3} },
          { {"2",0}, {"3",1}, {"4",2}, {"5more",3} },
          { {"2",0}, {"4",1}, {"more",2} },
          { {"small",0}, {"med",1}, {"big",2} },
          { {"low",0}, {"med",1}, {"high",2} }
        };
        classMap = { {"unacc",0}, {"acc",1}, {"good",2}, {"vgood",3} };
        domainSizes = { 4,4,4,3,3,3 };
        dependentPairs = { {0,1}, {2,3} };
    }
    else if (choice == 0) {
        // Balance Scale
        filename = "data/balance+scale/balance.data";   
        numAttributes = 4;
        numClasses = 3;

        attributeVals = { bs_vals, bs_vals, bs_vals, bs_vals };
        attributeNames = bsNames;
        classVals = bs_c;

        attributeMaps = {
          {{"1",0},{"2",1},{"3",2},{"4",3},{"5",4}},
          {{"1",0},{"2",1},{"3",2},{"4",3},{"5",4}},
          {{"1",0},{"2",1},{"3",2},{"4",3},{"5",4}},
          {{"1",0},{"2",1},{"3",2},{"4",3},{"5",4}}
        };
        classMap = { {"L",0},{"B",1},{"R",2} };
        domainSizes = { 5,5,5,5 };
        dependentPairs.clear();  // no fixed pairs, use dynamic MI or none

    }
    
}
