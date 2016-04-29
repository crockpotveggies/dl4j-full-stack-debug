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
import org.deeplearning4j.earlystopping.termination.{MaxEpochsTerminationCondition, MaxTimeIterationTerminationCondition}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.spark.earlystopping.{SparkDataSetLossCalculator, SparkEarlyStoppingTrainer}
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.ui.weights.HistogramIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable


/**
  * TrainNetSpark trains a convolutional neural network on Apache Spark using
  * Deeplearning4j. An early stopping method is used to prevent overfitting.
  *
  * 9383 individual photos in original test
  * Each image is a PNG 100 x 100 RGB
  *
  */
object TrainNetSpark {

  lazy val log: Logger = LoggerFactory.getLogger(TrainNetSpark.getClass)

  def main(args: Array[String]) = {

    println(s"\n\n\nRuntime max memory ${Runtime.getRuntime().maxMemory()}\n\n\n")

    // create spark context
    val sparkConf = new SparkConf()
    sparkConf.setMaster("local[2]")
    sparkConf.setAppName("SIGFPEDOOMSDAY")
    //sparkConf.set("spark.executor.memory","4g")
    //sparkConf.set("spark.driver.memory","4g")
    sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(true))
    val sc = new JavaSparkContext(sparkConf)

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

    // parallelize the dataset
    val list = new util.ArrayList[DataSet](numSamples)
    while(trainingSetIterator.hasNext) list.add(trainingSetIterator.next)
    val trainRDD = sc.parallelize(list)

    // perform training

    // configure the early stopping trainer
    //TODO val listener: LoggingEarlyStoppingListener = new LoggingEarlyStoppingListener()

    val esConf = new EarlyStoppingConfiguration.Builder()
      .epochTerminationConditions(new MaxEpochsTerminationCondition(30))
      .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(20, TimeUnit.MINUTES))
      .scoreCalculator(new SparkDataSetLossCalculator(trainRDD, true, sc))
      .evaluateEveryNEpochs(1)
      .modelSaver(new InMemoryModelSaver())
      .build()

    val nn = new MultiLayerNetwork(CNN.multiLayerConf(seed, iterations, nChannels, outputNum, imageWidth, imageHeight))
    nn.setListeners(new HistogramIterationListener(1))

    val trainer: SparkEarlyStoppingTrainer = new SparkEarlyStoppingTrainer(sc, esConf, nn, trainRDD, 50, 150, 5)

    val result = trainer.fit
    val bestModel: MultiLayerNetwork = result.getBestModel

    //TODO serialize bestModel and put it somewhere
  }

}
