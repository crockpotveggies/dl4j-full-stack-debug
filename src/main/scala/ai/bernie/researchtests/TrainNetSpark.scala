package ai.bernie.researchtests

import java.util
import java.util.Collections
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
import org.deeplearning4j.optimize.api.IterationListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.spark.earlystopping.{SparkDataSetLossCalculator, SparkEarlyStoppingTrainer}
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
//import org.deeplearning4j.ui.weights.HistogramIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.slf4j.{Logger, LoggerFactory}

//import scala.collection.JavaConversions._
import scala.collection.mutable


object TrainNetSpark {

  lazy val log: Logger = LoggerFactory.getLogger(TrainNetSpark.getClass)

  def main(args: Array[String]) = {

    println(s"\n\n\nRuntime max memory ${Runtime.getRuntime().maxMemory()}\n\n\n")

    println("Testing MKL lib functionality...")
    import org.nd4j.linalg.api.ndarray.INDArray;
    import org.nd4j.linalg.api.ops.impl.transforms.SoftMax;
    import org.nd4j.linalg.api.shape.Shape;
    import org.nd4j.linalg.factory.Nd4j;
    import org.nd4j.linalg.factory.Nd4jBackend;
    import org.nd4j.linalg.indexing.NDArrayIndex;
    val input = Nd4j.create(Array[Double]( -0.75, 0.58, 0.42, 1.03, -0.61, 0.19, -0.37, -0.40, -1.42, -0.04)).transpose();
    System.out.println("Input transpose " + Shape.shapeToString(input.shapeInfo()));
    val output = Nd4j.create(10,1);
    System.out.println("Element wise stride of output " + output.elementWiseStride());
    Nd4j.getExecutioner().exec(new SoftMax(input, output));

    // create spark context
    val sparkConf = new SparkConf()
//    sparkConf.setMaster("local[2]")
    sparkConf.setAppName("DEBUGAPP")
//    sparkConf.set("spark.yarn.am.memory", "8G")
//    sparkConf.set("spark.executor.memory","16G")
//    sparkConf.set("spark.driver.memory","8G")
//    sparkConf.set("spark.driver.cores", "4")
    sparkConf.set(SparkDl4jMultiLayer.AVERAGE_EACH_ITERATION, String.valueOf(false))
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

    // training dataset
    println("Load training data...")

    val trainRecordReader = new ImageRecordReader(imageWidth, imageHeight, nChannels, true)
    trainRecordReader.initialize(new FileSplit(new java.io.File("./cnn_dataset")))

    //val dataSetIterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, imageWidth*imageHeight*nChannels, labels.size())
    val trainingSetIterator: DataSetIterator = new RecordReaderDataSetIterator(trainRecordReader, batchSize, -1, 8)

    // parallelize the dataset
    val list = new util.ArrayList[DataSet](numSamples)
    while(trainingSetIterator.hasNext) list.add(trainingSetIterator.next)
    val trainRDD = sc.parallelize(list)

    // perform training

    // configure the early stopping trainer
    val earlyStoppingListener: LoggingEarlyStoppingListener = new LoggingEarlyStoppingListener()

    val esConf = new EarlyStoppingConfiguration.Builder()
      .epochTerminationConditions(new MaxEpochsTerminationCondition(30))
      .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(12, TimeUnit.HOURS))
      .scoreCalculator(new SparkDataSetLossCalculator(trainRDD, true, sc))
      .evaluateEveryNEpochs(1)
      .modelSaver(new InMemoryModelSaver())
      .build()

    val nn = new MultiLayerNetwork(CNN.multiLayerConf(seed, iterations, nChannels, outputNum, imageWidth, imageHeight))
//    nn.setListeners(new HistogramIterationListener(1))
    nn.setListeners(Collections.singletonList(new ScoreIterationListener(listenerFreq).asInstanceOf[IterationListener]))

    val trainer: SparkEarlyStoppingTrainer = new SparkEarlyStoppingTrainer(sc, esConf, nn, trainRDD, 100, numSamples, 20, earlyStoppingListener)

    val result = trainer.fit
    val bestModel: MultiLayerNetwork = result.getBestModel
    println("Score of best model is: "+bestModel.score().toString())

    //TODO serialize bestModel and put it somewhere
  }

}
