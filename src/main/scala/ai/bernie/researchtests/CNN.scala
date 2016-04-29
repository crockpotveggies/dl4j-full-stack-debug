package ai.bernie.researchtests

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup
import org.deeplearning4j.nn.conf.layers.{OutputLayer, DenseLayer, SubsamplingLayer, ConvolutionLayer}
import org.deeplearning4j.nn.conf.{Updater, GradientNormalization, NeuralNetConfiguration, MultiLayerConfiguration}
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * Convenience object for generating a neural net on the fly for different scenarios.
  */
object CNN {

  def multiLayerConf(seed: Int, iterations: Int, nChannels: Int, outputNum: Int, imageWidth: Int, imageHeight: Int): MultiLayerConfiguration = {
    val builder: MultiLayerConfiguration.Builder = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation("relu")
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.00005)
      .momentum(0.9)
      .regularization(false)
      .updater(Updater.ADAGRAD)
      .useDropConnect(true)
      .list(9)
      .layer(0, new ConvolutionLayer.Builder(3, 3)
        .padding(1,1)
        .name("cnn1")
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(20)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .build())
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
        .name("pool1")
        .build())
      .layer(2, new ConvolutionLayer.Builder(3, 3)
        .name("cnn2")
        .padding(1,1)
        .stride(1,1)
        .nOut(40)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
        .name("pool2")
        .build())
      .layer(4, new ConvolutionLayer.Builder(3, 3)
        .name("cnn3")
        .stride(1,1)
        .nOut(60)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .build())
      .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array[Int](2, 2))
        .name("pool3")
        .build())
      .layer(6, new ConvolutionLayer.Builder(2, 2)
        .name("cnn4")
        .stride(1,1)
        .nOut(80)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .build())
      .layer(7, new DenseLayer.Builder()
        .weightInit(WeightInit.XAVIER)
        .name("ffn1")
        .nOut(160)
        .dropOut(0.5)
        .weightInit(WeightInit.XAVIER)
        .activation("relu")
        .build())
      .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(outputNum)
        .weightInit(WeightInit.XAVIER)
        .activation("softmax")
        .build())
      .backprop(true).pretrain(false)

    new ConvolutionLayerSetup(builder, imageWidth, imageHeight, nChannels)

    builder.build()
  }

}
