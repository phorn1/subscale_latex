#include "CsvDataHandler/CsvDataHandler.h"
#include "SubscaleTypes.h"
#include "Subscale/Subscale.h"
#include "Subscale/SubscaleSeq.h"
#include "Clustering/Clustering.h"
#include "TimeMeasurement/TimeMeasurement.h"
#include "SubscaleConfig/SubscaleConfig.h"
#include <string>
#include <chrono>
#include <stdio.h>



int main(int argc, char* argv[])
{
    /*try
    {*/
        // Read command line arguments
        char* configPath;

        if (argc == 2)
        {
            // only 1 argument allowed which is the path to the config file
            configPath = argv[1];
        }
        else
        {
            throw std::runtime_error("Wrong number of arguments. Path to config file needed!");
        }

        // Start timer
        TimeMeasurement timer;
        timer.start();

        // Read config
        SubscaleConfig* config = new SubscaleConfig();
        config->readJson(configPath);

        // Handler for IO operations
        CsvDataHandler* csvHandler = new CsvDataHandler();

        
        // Read data points from a csv file
        vector<DataPoint> points;
        points = csvHandler->read(config->dataPath.c_str(), ',');

        int numberOfDimensions = points[0].values.size();
        // Shrink allocated memory of vector
        points.shrink_to_fit();
        printf("Number of Points: %llu   Number of Dimensions: %d\n", points.size(), numberOfDimensions);
        timer.createTimestamp("Read data");

        //
        // Calculate cluster candidates with the SUBSCALE algorithm
        ISubscale* subscale;

        if (config->runSequential)
        {
            // sequential
            subscale = new SubscaleSeq(config);
        }
        else
        {
            // parallel
            subscale = new Subscale(config);
        }

        LocalSubspaceTable* clusterCandidates = subscale->calculateClusterCandidates(points);
        timer.createTimestamp("Calculate cluster candidates with SUBSCALE");

        std::string resultFilePath;

        if (config->saveCandidates)
        {
            // Write cluster candidates to an output file
            resultFilePath = config->resultPath + "candidates.csv";
            csvHandler->writeVecTable(resultFilePath.c_str(), clusterCandidates, clusterCandidates->getTableSize());
            timer.createTimestamp("Write candidates");
        }

        if (config->useDBSCAN)
        {
            //
            // Search cluster candidates for real clusters with the DBSCAN algorithm
            Clustering* finalClustering = new Clustering(config->minPoints, config->epsilon);
            std::vector<Cluster> clusters = finalClustering->calculateClusters(points, clusterCandidates);
            timer.createTimestamp("Calculate final clusters with DBSCAN");

            if (config->saveClusters)
            {
                // Write clusters to an output file
                resultFilePath = config->resultPath + "clusters.csv";
                csvHandler->writeClusters(resultFilePath.c_str(), clusters);
                timer.createTimestamp("Write clusters");
            }
        }

        // Write timestamp differences to an output file
        resultFilePath = config->resultPath + "time_Complete.txt";
        timer.writeTimestampDeltas(resultFilePath.c_str());
    //}
    //catch (const std::exception& e)
    //{
    //    // Error handling
    //    printf("Catch: %s\n", e.what());
    //    return -1;
    //}
    
  
    return 0;
}
