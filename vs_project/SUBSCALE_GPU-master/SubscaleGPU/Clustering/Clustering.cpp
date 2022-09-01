#include "Clustering.h"
#include "mlpack/core.hpp"
#include "mlpack/methods/dbscan/dbscan.hpp"
#include <vector>

using namespace mlpack;
using namespace std;

Clustering::Clustering(int minPoints, double epsilon)
{
	this->minPoints = minPoints;
	this->epsilon = epsilon;
}

// find clusters in the given cluster candidates
vector<Cluster> Clustering::calculateClusters(vector<DataPoint> points, LocalSubspaceTable* clusterCandidates)
{
	dbscan::DBSCAN<> dbs(epsilon, minPoints);
	arma::mat data;

	vector<Cluster> clusters;

	// iterate over all entries in the table
	for (int i = 0; i < clusterCandidates->getTableSize(); i++) 
	{
		//
		// convert entry to a matrix
		vector<unsigned int> ids = clusterCandidates->getIdsVec(i);
		vector<unsigned int> dimensions = clusterCandidates->getDimensionsVec(i);

		data.set_size(dimensions.size(), ids.size());

		for (int j = 0; j < ids.size(); j++)
		{
			for (int k = 0; k < dimensions.size(); k++)
			{
				data(k, j) = points[ids[j]].values[dimensions[k]];
			}
		}

		//
		// clustering
		arma::mat centroids;
		arma::Row<size_t> assignements;

		int numClusters = dbs.Cluster(data, assignements, centroids);

		// check number of found clusters
		if (numClusters > 0)
		{
			//
			// convert cluster to a Cluster struct
			vector<vector<unsigned int>> pointsInClusters;
			pointsInClusters.resize(numClusters);

			for (int j = 0; j < ids.size(); j++)
			{
				if (assignements[j] != -1)
				{
					pointsInClusters[assignements[j]].push_back(ids[j]);
				}
			}

			for (std::vector<unsigned int> pointsInCluster : pointsInClusters)
			{
				Cluster cluster = { dimensions, pointsInCluster };
				clusters.push_back(cluster);
			}
		}

	}

	return clusters;
}
