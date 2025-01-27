#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "Timer.h"
#include "generic_data.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <chrono>
#include <string>

enum DistanceMetric {
	Manhattan,
    Euclidean,
    Minkowski,
    Chebyshev,
    TriangleInequality
};

int get_label_iris(const std::string& species) {
    if (species == "Iris-setosa") return 0;
    else if (species == "Iris-versicolor") return 1;
    else if (species == "Iris-virginica") return 2;
    else {
        std::cerr << "Unknown species: " << species << std::endl;
        return -1;
    }
}

std::string get_glass_label(int id) {
    switch (id) {
    case 1:
        return "building_windows_float_processed";
    case 2:
        return "building_windows_non_float_processed";
    case 3:
        return "vehicle_windows_float_processed";
    case 4:
        return "vehicle_windows_non_float_processed";
    case 5:
        return "containers";
    case 6:
        return "tableware";
    case 7:
        return "headlamps";
    default:
        std::cerr << "Unknown label: " << id << std::endl;
        return "Unknown";
    }
}

GenericDataSet<double> load_glass_data(const std::string& filename) {
    GenericDataSet<double> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (getline(file, line)) {
        std::istringstream ss(line);
        DataSetPoint<double> point;

        std::string temp;
        for (int i = 0; i < 11; ++i) {
            std::getline(ss, temp, ',');
            if (i == 0) {
                continue;
            }
            else if (i == 10)
            {
                point.group_id = atoi(temp.c_str());
                point.label = get_glass_label(point.group_id);
            }
            else {
                point.features.push_back(std::stod(temp));
            }
            point.features.push_back(std::stod(temp));
        }

        data.push_back(point);
    }

    file.close();
    return data;
}

GenericDataSet<double> load_iris_data(const std::string& filename) {
    GenericDataSet<double> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (getline(file, line)) {
        std::istringstream ss(line);
        DataSetPoint<double> point;

        std::string temp;
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, temp, ',');
            point.features.push_back(std::stod(temp));
        }

        std::getline(ss, point.label);
		point.group_id = get_label_iris(point.label);

        data.push_back(point);
    }

    file.close();
    return data;
}

GenericDataSet<double> load_wine_data(const std::string& filename) {
    GenericDataSet<double> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (getline(file, line)) {
        std::istringstream ss(line);
        DataSetPoint<double> point;

        std::string temp;
        for (int i = 0; i < 13; ++i) {
            std::getline(ss, temp, ',');
            if (i == 0) {
				point.label = temp;
                point.group_id = atoi(point.label.c_str());
            }
            else {
				point.features.push_back(std::stod(temp));
            }
            
        }

        data.push_back(point);
    }

    file.close();
    return data;
}

template <typename T>
bool save_out_file(const GenericDataSet<T>& dataset, const std::vector<int>& predicted_labels, const std::string& output_file_name) {
    std::ofstream output_file("OUT_" + output_file_name + ".csv");

    if (!output_file.is_open()) {
        std::cerr << "Error creating output file" << std::endl;
        return false;
    }

    output_file << "id,";
    for (size_t i = 0; i < dataset[0].features.size(); ++i) {
        output_file << "feature_" << i + 1;
        if (i < dataset[0].features.size() - 1) {
            output_file << ",";
        }
    }
    output_file << ",RId(real_group),CId(predicted_group)\n";

    for (size_t i = 0; i < dataset.size(); ++i) {
        output_file << i << ",";
        for (const auto& feature : dataset[i].features) {
            output_file << feature << ",";
        }
        output_file << dataset[i].group_id << "," << predicted_labels[i] << std::endl;
    }

    output_file.close();
    return true;
}

template <typename T>
double minkowski_distance(const DataSetPoint<T>& p1, const DataSetPoint<T>& p2, double p) {
    if (p1.features.size() != p2.features.size()) {
        std::cerr << "Error: Feature dimensions do not match." << std::endl;
        return -1;
    }

    double sum_powers = 0.0;
    for (size_t i = 0; i < p1.features.size(); ++i) {
        sum_powers += std::pow(std::abs(p1.features[i] - p2.features[i]), p);
    }

    return std::pow(sum_powers, 1.0 / p);
}

template <typename T>
double chebyshev_distance(const DataSetPoint<T>& p1, const DataSetPoint<T>& p2) {
    if (p1.features.size() != p2.features.size()) {
        std::cerr << "Error: Feature dimensions do not match." << std::endl;
        return -1;
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < p1.features.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(p1.features[i] - p2.features[i]));
    }

    return max_diff;
}

template <typename T>
double triangle_inequality_distance(const DataSetPoint<T>& query_point, const DataSetPoint<T>& target_point, const DataSetPoint<T>& reference_point) {
    double dist_query_ref = minkowski_distance(query_point, reference_point, 2);
    double dist_target_ref = minkowski_distance(target_point, reference_point, 2);

    return std::abs(dist_query_ref - dist_target_ref);
}

template <typename T>
double calculate_distance(const DataSetPoint<T>& query_point, const DataSetPoint<T>& target_point, DistanceMetric metric, const int minkowski_p = 2, const DataSetPoint<T>& reference_point = {}) {
    switch (metric) {
    case DistanceMetric::Manhattan:
        return minkowski_distance(query_point, target_point, 1.0);
    case DistanceMetric::Euclidean:
        return minkowski_distance(query_point, target_point, 2.0);
    case DistanceMetric::Minkowski:
        return minkowski_distance(query_point, target_point, minkowski_p); // (p=2) Eucledian distance (p=1) Manhattan distance
    case DistanceMetric::Chebyshev:
        return chebyshev_distance(query_point, target_point);
    case DistanceMetric::TriangleInequality:
        if (!reference_point.group_id) {
            std::cerr << "Reference point not specified for Triangle Inequality" << std::endl;
            return -1;
        }
        return triangle_inequality_distance(query_point, target_point, reference_point);
    default:
        std::cerr << "Unknown distance metric" << std::endl;
        return -1;
    }
}

template <typename T>
int knn(const GenericDataSet<T>& dataset, const DataSetPoint<T>& query_point, int k, DistanceMetric metric = DistanceMetric::Euclidean, const int minkowski_p = 2, const DataSetPoint<T>& reference_point = {}) {
    std::vector<std::pair<double, int>> distances;

    for (size_t i = 0; i < dataset.size(); ++i) {
        double distance = calculate_distance(query_point, dataset[i], metric, minkowski_p, reference_point);
        distances.push_back({ distance, static_cast<int>(i) });
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> labels;
    for (int i = 0; i < k; ++i) {
        int index = distances[i].second;
        labels.push_back((dataset[index].group_id));
    }

    std::map<int, int> label_counts;
    for (int label : labels) {
        label_counts[label]++;
    }

    int max_count = 0;
    int predicted_class = -1;
    for (const auto& pair : label_counts) {
        if (pair.second > max_count) {
            max_count = pair.second;
            predicted_class = pair.first;
        }
    }

    return predicted_class;
}

double calculate_rand_index(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    int n = true_labels.size();
    int a = 0;
    int b = 0;

    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            bool same_true = true_labels[i] == true_labels[j];
            bool same_pred = predicted_labels[i] == predicted_labels[j];

            if (same_true && same_pred) {
                a++;
            }
            else if (!same_true && !same_pred) {
                b++;
            }
        }
    }

    return (a + b) / static_cast<double>(n * (n - 1) / 2);
}

double calculate_adjusted_rand_index(const std::vector<int>& true_labels, const std::vector<int>& predicted_labels) {
    int n = true_labels.size();
    int a = 0; // Number of pairs with matching labels in both true and predicted
    int b = 0; // Number of pairs with non-matching labels in both true and predicted
    int c = 0; // Number of pairs with matching true labels but different predicted labels
    int d = 0; // Number of pairs with different true labels but matching predicted labels

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            bool same_true = true_labels[i] == true_labels[j];
            bool same_pred = predicted_labels[i] == predicted_labels[j];

            if (same_true && same_pred) {
                a++;
            }
            else if (!same_true && !same_pred) {
                b++;
            }
            else if (same_true && !same_pred) {
                c++;
            }
            else if (!same_true && same_pred) {
                d++;
            }
        }
    }

    double expected_a = (static_cast<double>(a + c) * (a + d)) / static_cast<double>(n * (n - 1) / 2);
    double expected_b = (static_cast<double>(b + c) * (b + d)) / static_cast<double>(n * (n - 1) / 2);

    double max_rand_index = (expected_a + expected_b) / static_cast<double>(n * (n - 1) / 2);

    return (a + b - expected_a - expected_b) / (max_rand_index - expected_a - expected_b);
}

double calculate_variance(const std::vector<int>& labels) {
    double mean = std::accumulate(labels.begin(), labels.end(), 0.0) / labels.size();
    double sum_squared_diff = 0.0;
    for (int label : labels) {
        sum_squared_diff += std::pow(label - mean, 2);
    }
    return sum_squared_diff / labels.size();
}

template <typename T>
int density(const GenericDataSet<T>& dataset, const DataSetPoint<T>& query_point, double radius, DistanceMetric metric = DistanceMetric::Euclidean, const int minkowski_p = 2, const DataSetPoint<T>& reference_point = {}) {
    int count = 0;
    for (const auto& target_point : dataset) {
        if (calculate_distance(query_point, target_point, metric, minkowski_p, reference_point) <= radius) {
            count++;
        }
    }
    return count;
}

template <typename T>
std::vector<std::vector<int>> nbc_grouping(const GenericDataSet<T>& dataset, double radius, double min_density, DistanceMetric metric = DistanceMetric::Euclidean, const int minkowski_p = 2, const DataSetPoint<T>& reference_point = {}) {
    int n = dataset.size();
    vector<vector<int>> clusters;
    vector<bool> visited(n, false);

    for (int i = 0; i < n; ++i) {
        if (!visited[i] && density(dataset, dataset[i], radius, metric, minkowski_p, reference_point) >= min_density) {
            vector<int> cluster;
            cluster.push_back(i);
            visited[i] = true;

            for (int j = 0; j < n; ++j) {
                if (!visited[j] && calculate_distance(dataset[i], dataset[j], metric, minkowski_p, reference_point) <= radius) {
                    cluster.push_back(j);
                    visited[j] = true;
                }
            }

            clusters.push_back(cluster);
        }
    }

    return clusters;
}

template <typename T>
double furthest_dataset_point(const GenericDataSet<T>& dataset, const DataSetPoint<T>& query_point, DistanceMetric metric = DistanceMetric::Euclidean, const int minkowski_p = 2, const DataSetPoint<T>& reference_point = {}) {
	double max_distance = 0.0;
	for (const auto& target_point : dataset) {
		double distance = calculate_distance(query_point, target_point, metric, minkowski_p, reference_point);
		if (distance > max_distance) {
			max_distance = distance;
		}
	}
	return max_distance;
}

template <typename T>
double furthest_neighbour(const GenericDataSet<T>& dataset, const DataSetPoint<T>& query_point, int k, DistanceMetric metric = DistanceMetric::Euclidean, const int minkowski_p = 2, const DataSetPoint<T>& reference_point = {}) {
	std::vector<std::pair<double, int>> distances;
	for (size_t i = 0; i < dataset.size(); ++i) {
		double distance = calculate_distance(query_point, dataset[i], metric, minkowski_p, reference_point);
		distances.push_back({ distance, static_cast<int>(i) });
	}

	std::sort(distances.begin(), distances.end());
	return distances[k - 1].first;
}

int main()
{
	// User input parameters
	int dataset_id = 0; // 0: Iris, 1: Wine, 2: Glass
	int distance_metric_id = 2; // 0: Manhattan, 1: Euclidean, 2: Minkowski, 3: Chebyshev, 4: Triangle Inequality

    int k;
    double radius;
    double minimum_density;
    DistanceMetric metric;
    int dimensions;

	// Time measurements
    Timer timer_file_input;
	Timer timer_knn;
	Timer timer_grouping;
	Timer timer_rand_index;
	Timer timer_file_output; 

	// Variables for storing results
    int groups_real;
    vector<int> real_cluster_sizes;

    GenericDataSet<double> dataset_input;
    string input_file_name;
    string dataset_name;
	string metric_name;

    const std::string input_file_name_iris = "iris.csv";
    const std::string input_file_name_wine = "wine.csv";
    const std::string input_file_name_glass = "glass.csv";

    GenericDataSet<double> dataset_input_wine = load_wine_data(input_file_name_wine);
    GenericDataSet<double> dataset_input_glass = load_glass_data(input_file_name_glass);


	// Dataset selection with default parameters
    switch (dataset_id) {
        case 0:
            dataset_name = "iris";
            input_file_name = input_file_name_iris;
            timer_file_input.startTimer();
			dataset_input = load_iris_data(input_file_name);
            timer_file_input.stopTimer();
			

            k = 3;
            dimensions = 4;
            radius = 0.5;
            minimum_density = 3;
            groups_real = 3;
			real_cluster_sizes = { 50, 50, 50 };
            break;
		case 1:
            dataset_name = "wine";
			input_file_name = input_file_name_wine;
            timer_file_input.startTimer();
            dataset_input = load_wine_data(input_file_name);
            timer_file_input.stopTimer();
			

            k = 3;
            dimensions = 13;
            radius = 5;
            minimum_density = 3;
            groups_real = 3;
			real_cluster_sizes = { 59, 71, 48 };
			break;
		case 2:
            dataset_name = "glass";
			input_file_name = input_file_name_glass;
            timer_file_input.startTimer();
            dataset_input = load_glass_data(input_file_name);
            timer_file_input.stopTimer();
			

            k = 3;
            dimensions = 11;
            radius = 0.5;
            minimum_density = 3;
            groups_real = 6;
			real_cluster_sizes = { 70, 76, 17, 13, 9, 29 }; //Note: This dataset has 7 groups, but group with id 4 has no data so it is excluded
			break;
		default:
			std::cerr << "Unknown dataset ID" << std::endl;
			return 1;
    }

	switch (distance_metric_id) {
	    case 0:
		    metric = DistanceMetric::Manhattan;
			metric_name = "Manhattan";
		    break;
	    case 1:
		    metric = DistanceMetric::Euclidean;
			metric_name = "Euclidean";
		    break;
	    case 2:
		    metric = DistanceMetric::Minkowski;
			metric_name = "Minkowski"; // TODO: Add p value
		    break;
	    case 3:
		    metric = DistanceMetric::Chebyshev;
			metric_name = "Chebyshev";
		    break;
	    case 4:
		    metric = DistanceMetric::TriangleInequality;
			metric_name = "Triangle Inequality"; // TODO: Add reference point
		    break;
	    default:
		    std::cerr << "Unknown distance metric ID" << std::endl;
		    return 1;
	}

    std::string output_file_name = "_SNN_" + dataset_name + "_D" + std::to_string(dimensions) + "_R" + std::to_string(dataset_input.size()) + "_k" + std::to_string(k) + "_" + metric_name;

    std::vector<int> true_labels;
    std::vector<int> predicted_labels;
	int groups_predicted;
    

	// kNN classification
	timer_knn.startTimer();
    for (auto query_point : dataset_input)
    {
        predicted_labels.push_back(knn(dataset_input, query_point, k, metric));
        true_labels.push_back(query_point.group_id);
    }
	timer_knn.stopTimer();
    

	// Grouping the results
	timer_grouping.startTimer();
    vector<vector<int>> clusters = nbc_grouping(dataset_input, radius, minimum_density);
	groups_predicted = clusters.size();

    for (const auto& cluster : clusters) {
        cout << "Cluster: ";
        for (int i : cluster) {
            cout << i << " ";
        }
        cout << endl;
    }

    int explored_cluster_min_size = INT_MAX;
    int cluster_min_index = -1;
    for (size_t i = 0; i < clusters.size(); ++i) {
        if (clusters[i].size() < explored_cluster_min_size) {
            explored_cluster_min_size = clusters[i].size();
            cluster_min_index = i;
        }
    }

    int explored_cluster_max_size = 0;
    int cluster_max_index = -1;
    for (size_t i = 0; i < clusters.size(); ++i) {
        if (clusters[i].size() > explored_cluster_max_size) {
            explored_cluster_max_size = clusters[i].size();
            cluster_max_index = i;
        }
    }

    vector<int> cluster_sizes;
    for (const auto& cluster : clusters) {
        cluster_sizes.push_back(cluster.size());
    }
    double explored_cluster_average_size = accumulate(cluster_sizes.begin(), cluster_sizes.end(), 0.0) / clusters.size();

    int real_cluster_min_size = INT_MAX;
    int real_cluster_min_index = -1;
    for (size_t i = 0; i < real_cluster_sizes.size(); ++i) {
        if (real_cluster_sizes[i] < real_cluster_min_size) {
            real_cluster_min_size = real_cluster_sizes[i];
            real_cluster_min_index = i;
        }
    }

    int real_cluster_max_size = 0;
    int real_cluster_max_index = -1;
    for (size_t i = 0; i < real_cluster_sizes.size(); ++i) {
        if (real_cluster_sizes[i] > real_cluster_max_size) {
            real_cluster_max_size = real_cluster_sizes[i];
            real_cluster_max_index = i;
        }
    }

    vector<int> cluster_sizes_real;
    for (const auto& cluster_size : real_cluster_sizes) {
        cluster_sizes_real.push_back(cluster_size);
    }
    double real_cluster_average_size = accumulate(cluster_sizes_real.begin(), cluster_sizes_real.end(), 0.0) / real_cluster_sizes.size();

	double clusters_variance = calculate_variance(cluster_sizes);
	double real_clusters_variance = calculate_variance(real_cluster_sizes);

    double true_labels_variance = calculate_variance(true_labels);
    double predicted_labels_variance = calculate_variance(predicted_labels);

	timer_grouping.stopTimer();


	// Output file creation
    timer_file_output.startTimer();
	save_out_file(dataset_input, predicted_labels, output_file_name);
	timer_file_output.stopTimer();


	// Rand Index Calculation
	timer_rand_index.startTimer();
    double rand_index = calculate_rand_index(true_labels, predicted_labels);
    double adjusted_rand_index = calculate_adjusted_rand_index(true_labels, predicted_labels);
	timer_rand_index.stopTimer();


	// Stat file creation
    std::ofstream stat_file("STAT_" + output_file_name + ".txt");
    if (!stat_file.is_open()) {
        std::cerr << "Error creating stat file" << std::endl;
        return 1;
    }

    stat_file << "Input File: " << input_file_name << std::endl;
    stat_file << "Number of Dimensions: " << dimensions << std::endl;
    stat_file << "Dataset Size: " << dataset_input.size() << std::endl;
    stat_file << "Value of k: " << k << std::endl;
	stat_file << "Distance Metric: " << metric_name << std::endl;

    stat_file << "Period of high_resolution_clock in this version of C++ implementation: " << timer_file_input.getPeriod() << " seconds" << std::endl;
    stat_file << "File read time: " << timer_file_input.getElapsedTime() << " seconds" << std::endl;
    stat_file << "kNN computation time: " << timer_knn.getElapsedTime() << " seconds" << std::endl;
	stat_file << "Grouping time: " << timer_grouping.getElapsedTime() << " seconds" << std::endl;
    stat_file << "RAND index computing time: " << timer_rand_index.getElapsedTime() << " seconds" << std::endl;
    stat_file << "Ouptup file save time: " << timer_file_output.getElapsedTime() << " seconds" << std::endl;
	double total_time = timer_file_input.getElapsedTime() + timer_knn.getElapsedTime() + timer_grouping.getElapsedTime() + timer_rand_index.getElapsedTime() + timer_file_output.getElapsedTime();
    stat_file << "Total time: " << total_time << " seconds" << std::endl;

    stat_file << "Number of real groups: " << groups_real << std::endl;
    stat_file << "Number of predicted groups: " << groups_predicted << std::endl;

	stat_file << "Minimum explored cluster size: " << explored_cluster_min_size << " (Cluster index: " << cluster_min_index << ")" << std::endl;
	stat_file << "Maximum explored cluster size: " << explored_cluster_max_size << " (Cluster index: " << cluster_max_index << ")" << std::endl;
	stat_file << "Average explored cluster size: " << explored_cluster_average_size << std::endl;
	stat_file << "Variance of explored clusters: " << clusters_variance << std::endl;

	stat_file << "Minimum real cluster size: " << real_cluster_min_size << " (Cluster index: " << real_cluster_min_index << ")" << std::endl;
	stat_file << "Maximum real cluster size: " << real_cluster_max_size << " (Cluster index: " << real_cluster_max_index << ")" << std::endl;
	stat_file << "Average real cluster size: " << real_cluster_average_size << std::endl;
	stat_file << "Variance of real clusters: " << real_clusters_variance << std::endl;

	stat_file << "Variance of true labels: " << true_labels_variance << std::endl;
	stat_file << "Variance of predicted labels: " << predicted_labels_variance << std::endl;

	stat_file << "Point pairs: " << true_labels.size() << std::endl;
	stat_file << "RAND index: " << rand_index << std::endl;
    stat_file << "Adjusted RAND index: " << adjusted_rand_index << std::endl;
    stat_file.close();


    // kNN output file creation
    std::ofstream knn_file("kNN_" + output_file_name + ".csv");
    if (!knn_file.is_open()) {
        std::cerr << "Error creating knn file" << std::endl;
        return 1;
    }

    knn_file << "id,Eps,maxEps,\n";
    int id_knn_file = 0;
    for (auto query_point : dataset_input)
    {
		double eps = furthest_neighbour(dataset_input, query_point, k);
		double maxEps = furthest_dataset_point(dataset_input, query_point);
        knn_file << id_knn_file << ","
            << eps << "," 
            << maxEps << std::endl;

        id_knn_file++;
    }
	knn_file.close();


    return 0;
}
