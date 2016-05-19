libnd4j SIGFPE Fatal Error
==========================

Encountered a fatal error during Spark training of a convolutional net on JVM 1.8. Appears related to libnd4j. 

To run the test, type `gradle run -Plibnd4jOS=linux` (note there's a platform-specific parameter). Main class is `ai.bernie.researchtests.TrainNetSpark`.

#Dataset

Data lives in the root folder `cnn_dataset`. You will need to add your own dataset.

#Prerequisites

This repo requires the latest version of Gradle.
