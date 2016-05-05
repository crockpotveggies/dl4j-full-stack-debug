package ai.bernie.researchtests

import java.util
import java.util.concurrent.TimeUnit

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.api.java.JavaSparkContext
import org.canova.api.split.FileSplit
import org.canova.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration
import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, MaxTimeIterationTerminationCondition}
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.earlystopping.{SparkDataSetLossCalculator, SparkEarlyStoppingTrainer}
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.ui.weights.HistogramIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.{Logger, LoggerFactory}

//import scala.collection.JavaConversions._
import scala.collection.mutable


/**
  * BernieFaces is a test to demonstrate a classifier of ethnicity using the
  * Deeplearning4j framework. This test trains a convolutional neural network on
  * a dataset of faces labelled for ethnicity. Dataset is not guaranteed to be
  * labelled 100% accurately as it as not been examined by multiple labelers.
  *
  * 9383 individual photos
  * Each image is a PNG 100 x 100 RGB
  *
  */
// TODO add dropout to each layer in the network
// typically use .1 to .5
// this will cut neurons out to prevent subsequent layer from overfitting from previous layer
object TrainNetStandalone {

  lazy val log: Logger = LoggerFactory.getLogger(TrainNetSpark.getClass)

  def main(args: Array[String]) = {

    println(s"\n\n\nRuntime max memory ${Runtime.getRuntime().maxMemory()}\n\n\n")

    // neural network parameters
    val imageWidth = 100
    val imageHeight = 100
    val nChannels = 1
    val outputNum = 8 // number of labels
    val numSamples = 9383 // LFWLoader.NUM_IMAGES

    val batchSize = 10
    val iterations = 5
    val splitTrainNum = (batchSize*.8).toInt
    val seed = 123
    val listenerFreq = iterations/5
    val testInputBuilder = mutable.ArrayBuilder.make[INDArray]
    val testLabelsBuilder = mutable.ArrayBuilder.make[INDArray]

    // load datasets

    // training dataset (parallelized with Spark)
    log.info("Load training data...")
    val trainLabels = new java.util.ArrayList[String]()
    val trainRecordReader = new ImageRecordReader(imageWidth, imageHeight, nChannels, true, trainLabels)
    trainRecordReader.initialize(new FileSplit(new java.io.File("./cnn_dataset")))
    //val dataSetIterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, imageWidth*imageHeight*nChannels, labels.size())
    val trainingSetIterator: DataSetIterator = new RecordReaderDataSetIterator(trainRecordReader, batchSize, -1, trainLabels.size())

    log.info(s"\n\nNumber of labels: ${trainLabels}\n\n")

    // perform training

    // configure the early stopping trainer
    //TODO val listener: LoggingEarlyStoppingListener = new LoggingEarlyStoppingListener()

    val esConf = new EarlyStoppingConfiguration.Builder()
      .epochTerminationConditions(new MaxEpochsTerminationCondition(30))
      .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
      .scoreCalculator(new DataSetLossCalculator(trainingSetIterator, true))
      .evaluateEveryNEpochs(1)
      .modelSaver(new InMemoryModelSaver())
      .build()

    val nn = new MultiLayerNetwork(CNN.multiLayerConf(seed, iterations, nChannels, outputNum, imageWidth, imageHeight))
    nn.setListeners(new HistogramIterationListener(1))

    val trainer: EarlyStoppingTrainer = new EarlyStoppingTrainer(esConf, nn, trainingSetIterator)

    val result = trainer.fit
    val bestModel: MultiLayerNetwork = result.getBestModel

    //TODO serialize bestModel and put it somewhere
  }

}
