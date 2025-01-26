#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "iris_data.h"
#include "Timer.h"
#include <fstream>
#include <sstream>
#include <map>
#include <chrono>

typedef std::vector<IrisData> DataPoint;
typedef std::vector<DataPoint> DataSet;
typedef std::vector<int> Labels;

enum DistanceMetric {
    Euclidean,
    Minkowski,
    Chebyshev,
    TriangleInequality
};

IrisDataSet load_iris_data(const std::string& filename) {
    IrisDataSet data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    std::string line;
    while (getline(file, line)) {
        std::istringstream ss(line);
        IrisData iris;
        std::string temp;

        std::getline(ss, temp, ',');
        iris.sepal_length = std::stod(temp);
        std::getline(ss, temp, ',');
        iris.sepal_width = std::stod(temp);
        std::getline(ss, temp, ',');
        iris.petal_length = std::stod(temp);
        std::getline(ss, temp, ',');
        iris.petal_width = std::stod(temp);
        std::getline(ss, iris.species);

        data.push_back(iris);
    }

    file.close();
    return data;
}

int get_label(const std::string& species) {
    if (species == "Iris-setosa") return 0;
    else if (species == "Iris-versicolor") return 1;
    else if (species == "Iris-virginica") return 2;
    else {
        std::cerr << "Unknown species: " << species << std::endl;
        return -1;
    }
}

std::string get_species(int label) {
    switch (label) {
    case 0:
        return "Iris-setosa";
    case 1:
        return "Iris-versicolor";
    case 2:
        return "Iris-virginica";
    default:
        std::cerr << "Unknown label: " << label << std::endl;
        return "Unknown";
    }
}

double minkowski_distance(const IrisData& query_point, const IrisData& target_point, double p) {
    double sum_powers = 0.0;

    sum_powers += std::pow(std::abs(query_point.petal_length - target_point.petal_length), p);
    sum_powers += std::pow(std::abs(query_point.petal_width - target_point.petal_width), p);
    sum_powers += std::pow(std::abs(query_point.sepal_length - target_point.sepal_length), p);
    sum_powers += std::pow(std::abs(query_point.sepal_width - target_point.sepal_width), p);

    return std::pow(sum_powers, 1.0 / p);
}

double chebyshev_distance(const IrisData& query_point, const IrisData& target_point) {
    double max_diff = 0.0;

    max_diff = std::max(max_diff, std::abs(query_point.petal_length - target_point.petal_length));
    max_diff = std::max(max_diff, std::abs(query_point.petal_width - target_point.petal_width));
    max_diff = std::max(max_diff, std::abs(query_point.sepal_length - target_point.sepal_length));
    max_diff = std::max(max_diff, std::abs(query_point.sepal_width - target_point.sepal_width));

    return max_diff;
}

double triangle_inequality_distance(const IrisData& query_point, const IrisData& target_point, const IrisData& reference_point) {
    double dist_query_ref = minkowski_distance(query_point, reference_point, 2);
    double dist_target_ref = minkowski_distance(target_point, reference_point, 2);

    return std::abs(dist_query_ref - dist_target_ref);
}

int knn(IrisDataSet& iris_dataset, const IrisData& query_point, int k, DistanceMetric metric = DistanceMetric::Euclidean, const IrisData& reference_point = {}) {
    std::vector<std::pair<double, int>> distances;

    for (size_t i = 0; i < iris_dataset.size(); ++i) {
        IrisData target_point = {
            iris_dataset[i].petal_length,
            iris_dataset[i].petal_width,
            iris_dataset[i].sepal_length,
            iris_dataset[i].sepal_width,
            iris_dataset[i].species
        };

        double distance;
        switch (metric) {
            case DistanceMetric::Euclidean:
                distance = minkowski_distance(query_point, target_point, 2.0);
                break;
            case DistanceMetric::Minkowski:
                distance = minkowski_distance(query_point, target_point, 2.0); // (p=2) Eucledian distance (p=1) Manhattan distance
                break;
            case DistanceMetric::Chebyshev:
                distance = chebyshev_distance(query_point, target_point);
                break;
            case DistanceMetric::TriangleInequality:
                if (reference_point.species.empty()) {
                    std::cerr << "Reference point not specified for Triangle Inequality" << std::endl;
                    return -1;
                }
                distance = triangle_inequality_distance(query_point, target_point, reference_point);
                break;
            default:
                std::cerr << "Unknown distance metric" << std::endl;
                return -1;
        }

        distances.push_back({ distance, static_cast<int>(i) });
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> labels;
    for (int i = 0; i < k; ++i) {
        int index = distances[i].second;
        labels.push_back(get_label(iris_dataset[index].species));
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

double density(IrisDataSet& iris_dataset, const IrisData& query_point, double radius) {
    int count = 0;
    for (const auto& target_point : iris_dataset) {
        if (minkowski_distance(query_point, target_point, 2) <= radius) {
            count++;
        }
    }
    return count;
}

vector<vector<int>> nbc_grouping(IrisDataSet& iris_dataset, double radius, double min_density) {
    int n = iris_dataset.size();
    std::vector<std::pair<double, int>> distances;
    vector<vector<int>> clusters;
    vector<bool> visited(n, false);

    for (int i = 0; i < n; ++i) {
        if (!visited[i] && density(iris_dataset, iris_dataset[i], radius) >= min_density) {
            vector<int> cluster;
            cluster.push_back(i);
            visited[i] = true;

            for (int j = 0; j < n; ++j) {
                if (!visited[j] && minkowski_distance(iris_dataset[i], iris_dataset[j], 2) <= radius) {
                    cluster.push_back(j);
                    visited[j] = true;
                }
            }

            clusters.push_back(cluster);
        }
    }

    return clusters;
}

double furthest_dataset_point(IrisDataSet& iris_dataset, const IrisData& query_point) {
	double max_distance = 0.0;
	for (const auto& target_point : iris_dataset) {
		double distance = minkowski_distance(query_point, target_point, 2);
		if (distance > max_distance) {
			max_distance = distance;
		}
	}
	return max_distance;
}

double furthest_neighbour(IrisDataSet& iris_dataset, const IrisData& query_point, int k) {
	std::vector<std::pair<double, int>> distances;
	for (size_t i = 0; i < iris_dataset.size(); ++i) {
		IrisData target_point = {
			iris_dataset[i].petal_length,
			iris_dataset[i].petal_width,
			iris_dataset[i].sepal_length,
			iris_dataset[i].sepal_width,
			iris_dataset[i].species
		};
		double distance = minkowski_distance(query_point, target_point, 2.0);
		distances.push_back({ distance, static_cast<int>(i) });
	}

	std::sort(distances.begin(), distances.end());
	return distances[k - 1].first;
}

int main()
{
    Timer timer_file_input;
	Timer timer_knn;
	Timer timer_grouping;
	Timer timer_rand_index;
	Timer timer_file_output;


    int k = 3;
    int dimensions = 4;
    double radius = 0.5;
    double minimum_density = 3;
    int groups_real = 3;

    std::string input_file_name = "iris.csv";
    IrisDataSet iris_dataset = load_iris_data(input_file_name);
    std::string output_file_name = "_SNN_iris_D" + std::to_string(dimensions) + "_R" + std::to_string(iris_dataset.size()) + "_k" + std::to_string(k);


	//IrisData query_point = { 5.5,2.4,3.7,1.1, "" };
    DistanceMetric metric = DistanceMetric::Minkowski;
    IrisData reference_point = iris_dataset[0];
    

    std::vector<int> true_labels;
    std::vector<int> predicted_labels;
	std::vector<IrisData> query_points;

	int groups_predicted;


	// Reading test data
    timer_file_input.startTimer();
    std::ifstream test_data_file("test_data_iris.csv");
    if (!test_data_file.is_open()) {
        std::cerr << "Error opening test data file" << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(test_data_file, line)) {
        std::istringstream ss(line);
        std::string temp;

        IrisData query_point;
        std::getline(ss, temp, ',');
        query_point.petal_length = std::stod(temp);
        std::getline(ss, temp, ',');
        query_point.petal_width = std::stod(temp);
        std::getline(ss, temp, ',');
        query_point.sepal_length = std::stod(temp);
        std::getline(ss, temp, ',');
        query_point.sepal_width = std::stod(temp);
        std::getline(ss, query_point.species);

		query_points.push_back(query_point);
    }
    timer_file_input.stopTimer();


	// kNN classification
	timer_knn.startTimer();
    for (auto query_point : query_points)
    {
        predicted_labels.push_back(knn(iris_dataset, query_point, k, metric));
        true_labels.push_back(get_label(query_point.species));
    }
	timer_knn.stopTimer();
    

	// Grouping the results
	timer_grouping.startTimer();
    vector<vector<int>> clusters = nbc_grouping(iris_dataset, radius, minimum_density);
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

    vector<int> real_cluster_sizes = { 0, 0, 0 };
    for (size_t i = 0; i < iris_dataset.size(); ++i) {
        if (get_label(iris_dataset[i].species) == 0) {
			real_cluster_sizes[0] += 1;
        }
        else if (get_label(iris_dataset[i].species) == 1) {
            real_cluster_sizes[1] += 1;
		}
        else if (get_label(iris_dataset[i].species) == 2) {
            real_cluster_sizes[2] += 1;
        }
    }

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
    std::ofstream output_file("OUT_" + output_file_name + ".csv");
    if (!output_file.is_open()) {
        std::cerr << "Error creating output file" << std::endl;
        return 1;
    }

    output_file << "id,petal_length,petal_width,sepal_length,sepal_width,real_species,predicted_species\n";
    int id_output_file = 0;
    for (auto query_point : query_points)
    {
        std::string predicted_species = get_species(predicted_labels[id_output_file]);
        output_file << id_output_file << ","
            << query_point.petal_length << ","
            << query_point.petal_width << ","
            << query_point.sepal_length << ","
            << query_point.sepal_width << ","
            << query_point.species << ","
            << predicted_species << std::endl;

        id_output_file++;
    }
	timer_file_output.stopTimer();


	// Rand Index Calculation
	timer_rand_index.startTimer();
    double rand_index = calculate_rand_index(true_labels, predicted_labels);
    double adjusted_rand_index = calculate_adjusted_rand_index(true_labels, predicted_labels);
	timer_rand_index.stopTimer();


	// Stat file creation
    std::ofstream stat_file("STAT_" + output_file_name + ".txt");
    if (!stat_file.is_open()) {
        std::cerr << "Error creating metadata file" << std::endl;
        return 1;
    }

    stat_file << "Input File: " << input_file_name << std::endl;
    stat_file << "Number of Dimensions: " << dimensions << std::endl;
    stat_file << "Dataset Size: " << iris_dataset.size() << std::endl;
    stat_file << "Value of k: " << k << std::endl;
    stat_file << "Distance Metric: ";
    switch (metric) {
    case DistanceMetric::Euclidean:
        stat_file << "Euclidean" << std::endl;
        break;
    case DistanceMetric::Minkowski:
        stat_file << "Minkowski" << std::endl;
        break;
    case DistanceMetric::Chebyshev:
        stat_file << "Chebyshev" << std::endl;
        break;
    case DistanceMetric::TriangleInequality:
        stat_file << "Triangle Inequality" << std::endl;
        break;
    default:
        stat_file << "Unknown" << std::endl;
    }

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


    // kNN output file creation
    std::ofstream knn_file("kNN_" + output_file_name + ".csv");
    if (!knn_file.is_open()) {
        std::cerr << "Error creating knn file" << std::endl;
        return 1;
    }

    knn_file << "id,Eps,maxEps,\n";
    int id_knn_file = 0;
    for (auto query_point : query_points)
    {
		double eps = furthest_neighbour(iris_dataset, query_point, k);
		double maxEps = furthest_dataset_point(iris_dataset, query_point);
        knn_file << id_knn_file << ","
            << eps << "," 
            << maxEps << ","
            << std::endl;

        id_knn_file++;
    }
    timer_file_output.stopTimer();

    test_data_file.close();
    output_file.close();
	stat_file.close();
	knn_file.close();

    return 0;
}
