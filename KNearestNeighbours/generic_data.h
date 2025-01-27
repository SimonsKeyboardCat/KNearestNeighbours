#pragma once

#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>

template <typename T>
struct DataSetPoint {
    std::vector<T> features;
    std::string label;
    int group_id;
};

template <typename T>
using GenericDataSet = std::vector<DataSetPoint<T>>;

template <typename T>
DataSetPoint<T> load_iris_data(const std::string& filename);

#endif