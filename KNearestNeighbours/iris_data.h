#pragma once

#ifndef IRIS_DATA_H
#define IRIS_DATA_H

#include <vector>
#include <string>

struct IrisData {
    double sepal_length;
    double sepal_width;
    double petal_length;
    double petal_width;
    std::string species;
};

using IrisDataSet = std::vector<IrisData>;

IrisDataSet load_iris_data_temp(const std::string& filename);

#endif