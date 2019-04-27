/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.recommendation

import java.io.File
import java.util.Random

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.{ArrayBuffer, WrappedArray}
import scala.language.existentials

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import org.apache.commons.io.FileUtils
import org.apache.commons.io.filefilter.TrueFileFilter
import org.scalatest.BeforeAndAfterEach

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.recommendation.ALS._
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTest, MLTestingUtils}
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.scheduler.{SparkListener, SparkListenerStageCompleted}
import org.apache.spark.sql.{DataFrame, Encoder, Row, SparkSession}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.StreamingQueryException
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils

// scalastyle:off
class ALSExplainSuite extends MLTest with DefaultReadWriteTest with Logging {

  override def beforeAll(): Unit = {
    super.beforeAll()
    sc.setCheckpointDir(tempDir.getAbsolutePath)
  }

  override def afterAll(): Unit = {
    super.afterAll()
  }

  /**
   * Generates random user/item factors, with i.i.d. values drawn from U(a, b).
   * @param size number of users/items
   * @param rank number of features
   * @param random random number generator
   * @param a min value of the support (default: -1)
   * @param b max value of the support (default: 1)
   * @return a sequence of (ID, factors) pairs
   */
  private def genFactors(
      size: Int,
      rank: Int,
      random: Random,
      a: Float = -1.0f,
      b: Float = 1.0f): Seq[(Int, Array[Float])] = {
    require(size > 0 && size < Int.MaxValue / 3)
    require(b > a)
    val ids = mutable.Set.empty[Int]
    while (ids.size < size) {
      ids += random.nextInt()
    }
    val width = b - a
    ids.toSeq.sorted.map(id => (id, Array.fill(rank)(a + random.nextFloat() * width)))
  }

  /**
   * Generates an implicit feedback dataset for testing ALS.
   *
   * @param sc SparkContext
   * @param numUsers number of users
   * @param numItems number of items
   * @param rank rank
   * @param noiseStd the standard deviation of additive Gaussian noise on training data
   * @param seed random seed
   * @return (training, test)
   */
  def genImplicitTestData(
      sc: SparkContext,
      numUsers: Int,
      numItems: Int,
      rank: Int,
      noiseStd: Double = 0.0,
      seed: Long = 11L): (RDD[Rating[Int]], RDD[Rating[Int]]) = {
    // The assumption of the implicit feedback model is that unobserved ratings are more likely to
    // be negatives.
    val positiveFraction = 1.0
    val negativeFraction = 1.0 - positiveFraction
    val trainingFraction = 0.6
    val testFraction = 0.3
    val totalFraction = trainingFraction + testFraction
    val random = new Random(seed)
    val userFactors = genFactors(numUsers, rank, random)
    val itemFactors = genFactors(numItems, rank, random)
    val training = ArrayBuffer.empty[Rating[Int]]
    val test = ArrayBuffer.empty[Rating[Int]]
    for ((userId, userFactor) <- userFactors; (itemId, itemFactor) <- itemFactors) {
      val rating = blas.sdot(rank, userFactor, 1, itemFactor, 1)
      val threshold = if (rating > 0) positiveFraction else negativeFraction
      val observed = random.nextDouble() < threshold
      if (observed) {
        val x = random.nextDouble()
        if (x < totalFraction) {
          if (x < trainingFraction) {
            val noise = noiseStd * random.nextGaussian()
            training += Rating(userId, itemId, rating + noise.toFloat)
          } else {
            test += Rating(userId, itemId, rating)
          }
        }
      }
    }
    logInfo(s"Generated an implicit feedback dataset with ${training.size} ratings for training " +
      s"and ${test.size} for test.")
    (sc.parallelize(training, 2), sc.parallelize(test, 2))
  }
  
  def testALSExplain(numUsers: Int = 20,
                     numItems: Int = 40,
                     sampleRank: Int = 1,
                     modelRank: Int = 5,
                     regParam: Float = 0.01f,
                     alpha: Float = 10.0f,
                     maxIterations: Int = 20,
                     tol: Float = 0.1f): Unit = {
     val spark = this.spark
     import spark.implicits._
     
     // Create sample data
     val (training, test) = genImplicitTestData(spark.sparkContext, numUsers, numItems, sampleRank, 0.0f, 11L)
     // TODO Adjust explain calcuation to handle negative ratings
     val ratings_df = training.
       union(test).
       toDF().
       filter(col("rating") >= 0.0) // TODO: Adjust Explain Calc to handle negative ratings

     // Fit ALS model
     val als = new ALS()
       .setRank(modelRank)
       .setRegParam(regParam)
       .setImplicitPrefs(true)
       .setAlpha(alpha)
       .setSeed(0)
       .setColdStartStrategy("drop")
       .setMaxIter(20)
     val model = als.fit(ratings_df)

     // Create predictions
      val predictions = model.transform(ratings_df.drop(col("rating")))

     // Create and run ALSExplain with params
     val alsExplain = new ALSExplain()
     val topExplanation = 10
     val explanation = alsExplain.explain(model.itemFactors, ratings_df, "user", "item", "rating", topExplanation, regParam, alpha)
     
     // Transform output and compare score with predictions
     val explainTable = (
         explanation
         .select(col("user"), explode(col("productExplanations")))
         .select(col("user"), col("col.explainedPid"), col("col.score"), col("col.productInfluence"))
         .toDF("user", "explainedPid", "score", "productInfluence")
         .select(col("user"), col("explainedPid"), col("score"), explode(col("productInfluence")))
         .select(col("user"), col("explainedPid"), col("score"), col("col.productId"), col("col.influenceScore"), col("col.percentage"))
         .toDF("user", "explainedPid", "score", "productId", "influenceScore", "percentage")
     )

     // Extract explain derived scores
     val explainRecsDistinct = explainTable.
                               select(col("user"), col("explainedPid"), col("score")).
                               toDF("user", "item", "score").
                               distinct()

     // Join and compare explain score with predictions (should be within tol)
     val numPredWithinTol = predictions.
                            join(explainRecsDistinct, Seq("user", "item")).
                            withColumn("net", col("prediction") - col("score")).
                            filter(col("net") < tol && col("net") > -tol).
                            count()

     // Assert all estimates were within tolerance
     println("explainRecsDistinct:" + explainRecsDistinct.count())
     println("numPredWithinTol:" + numPredWithinTol)
     assert(explainRecsDistinct.count() === numPredWithinTol)
  }

  test("Test ALSExplain on simple dataset ") {
      import testImplicits._
      import org.apache.spark.sql.functions._
      testALSExplain()
  }

  private def checkRecommendations(
      topK: DataFrame,
      expected: Map[Int, Seq[(Int, Float)]],
      dstColName: String): Unit = {
    val spark = this.spark
    import spark.implicits._

    assert(topK.columns.contains("recommendations"))
    topK.as[(Int, Seq[(Int, Float)])].collect().foreach { case (id: Int, recs: Seq[(Int, Float)]) =>
      assert(recs === expected(id))
    }
    topK.collect().foreach { row =>
      val recs = row.getAs[WrappedArray[Row]]("recommendations")
      assert(recs(0).fieldIndex(dstColName) == 0)
      assert(recs(0).fieldIndex("rating") == 1)
    }
  }
}

object ALSExplainSuite extends Logging {

  /**
   * Mapping from all Params to valid settings which differ from the defaults.
   * This is useful for tests which need to exercise all Params, such as save/load.
   * This excludes input columns to simplify some tests.
   */
  val allModelParamSettings: Map[String, Any] = Map(
    "predictionCol" -> "myPredictionCol"
  )

  /**
   * Mapping from all Params to valid settings which differ from the defaults.
   * This is useful for tests which need to exercise all Params, such as save/load.
   * This excludes input columns to simplify some tests.
   */
  val allEstimatorParamSettings: Map[String, Any] = allModelParamSettings ++ Map(
    "maxIter" -> 1,
    "rank" -> 1,
    "regParam" -> 0.01,
    "numUserBlocks" -> 2,
    "numItemBlocks" -> 2,
    "implicitPrefs" -> true,
    "alpha" -> 0.9,
    "nonnegative" -> true,
    "checkpointInterval" -> 20,
    "intermediateStorageLevel" -> "MEMORY_ONLY",
    "finalStorageLevel" -> "MEMORY_AND_DISK_SER"
  )

  // Helper functions to generate test data we share between ALS test suites

  /**
   * Generates random user/item factors, with i.i.d. values drawn from U(a, b).
   * @param size number of users/items
   * @param rank number of features
   * @param random random number generator
   * @param a min value of the support (default: -1)
   * @param b max value of the support (default: 1)
   * @return a sequence of (ID, factors) pairs
   */
  private def genFactors(
      size: Int,
      rank: Int,
      random: Random,
      a: Float = -1.0f,
      b: Float = 1.0f): Seq[(Int, Array[Float])] = {
    require(size > 0 && size < Int.MaxValue / 3)
    require(b > a)
    val ids = mutable.Set.empty[Int]
    while (ids.size < size) {
      ids += random.nextInt()
    }
    val width = b - a
    ids.toSeq.sorted.map(id => (id, Array.fill(rank)(a + random.nextFloat() * width)))
  }

  /**
   * Generates an implicit feedback dataset for testing ALS.
   *
   * @param sc SparkContext
   * @param numUsers number of users
   * @param numItems number of items
   * @param rank rank
   * @param noiseStd the standard deviation of additive Gaussian noise on training data
   * @param seed random seed
   * @return (training, test)
   */
  def genImplicitTestData(
      sc: SparkContext,
      numUsers: Int,
      numItems: Int,
      rank: Int,
      noiseStd: Double = 0.0,
      seed: Long = 11L): (RDD[Rating[Int]], RDD[Rating[Int]]) = {
    // The assumption of the implicit feedback model is that unobserved ratings are more likely to
    // be negatives.
    val positiveFraction = 0.8
    val negativeFraction = 1.0 - positiveFraction
    val trainingFraction = 0.6
    val testFraction = 0.3
    val totalFraction = trainingFraction + testFraction
    val random = new Random(seed)
    val userFactors = genFactors(numUsers, rank, random)
    val itemFactors = genFactors(numItems, rank, random)
    val training = ArrayBuffer.empty[Rating[Int]]
    val test = ArrayBuffer.empty[Rating[Int]]
    for ((userId, userFactor) <- userFactors; (itemId, itemFactor) <- itemFactors) {
      val rating = blas.sdot(rank, userFactor, 1, itemFactor, 1)
      val threshold = if (rating > 0) positiveFraction else negativeFraction
      val observed = random.nextDouble() < threshold
      if (observed) {
        val x = random.nextDouble()
        if (x < totalFraction) {
          if (x < trainingFraction) {
            val noise = noiseStd * random.nextGaussian()
            training += Rating(userId, itemId, rating + noise.toFloat)
          } else {
            test += Rating(userId, itemId, rating)
          }
        }
      }
    }
    logInfo(s"Generated an implicit feedback dataset with ${training.size} ratings for training " +
      s"and ${test.size} for test.")
    (sc.parallelize(training, 2), sc.parallelize(test, 2))
  }
}
// scalastyle:on