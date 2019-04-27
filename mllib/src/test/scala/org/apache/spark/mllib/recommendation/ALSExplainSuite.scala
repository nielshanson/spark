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

package org.apache.spark.mllib.recommendation

import scala.collection.JavaConverters._
import scala.math.abs
import scala.util.Random

import breeze.linalg.{DenseMatrix => BDM}

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.storage.StorageLevel

object ALSExplainSuite {

  def generateRatingsAsJava(
      users: Int,
      products: Int,
      features: Int,
      samplingRate: Double,
      implicitPrefs: Boolean,
      negativeWeights: Boolean): (java.util.List[Rating], Array[Double], Array[Double]) = {
    val (sampledRatings, trueRatings, truePrefs) =
      generateRatings(users, products, features, samplingRate, implicitPrefs, negativeWeights)
    (sampledRatings.asJava, trueRatings.toArray, if (truePrefs == null) null else truePrefs.toArray)
  }

  def generateRatings(
      users: Int,
      products: Int,
      features: Int,
      samplingRate: Double,
      implicitPrefs: Boolean = false,
      negativeWeights: Boolean = false,
      negativeFactors: Boolean = true): (Seq[Rating], BDM[Double], BDM[Double]) = {
    val rand = new Random(42)

    // Create a random matrix with uniform values from -1 to 1
    def randomMatrix(m: Int, n: Int) = {
      if (negativeFactors) {
        new BDM(m, n, Array.fill(m * n)(rand.nextDouble() * 2 - 1))
      } else {
        new BDM(m, n, Array.fill(m * n)(rand.nextDouble()))
      }
    }

    val userMatrix = randomMatrix(users, features)
    val productMatrix = randomMatrix(features, products)
    val (trueRatings, truePrefs) =
      if (implicitPrefs) {
        // Generate raw values from [0,9], or if negativeWeights, from [-2,7]
        val raw = new BDM(users, products,
          Array.fill(users * products)(
            (if (negativeWeights) -2 else 0) + rand.nextInt(10).toDouble))
        val prefs =
          new BDM(users, products, raw.data.map(v => if (v > 0) 1.0 else 0.0))
        (raw, prefs)
      } else {
        (userMatrix * productMatrix, null)
      }

    val sampledRatings = {
      for (u <- 0 until users; p <- 0 until products if rand.nextDouble() < samplingRate)
        yield Rating(u, p, trueRatings(u, p))
    }

    (sampledRatings, trueRatings, truePrefs)
  }
}

// scalastyle:off
class ALSExplainSuite extends SparkFunSuite with MLlibTestSparkContext {
    test("rank-5 explain") {
        testALSExplain(50,100,5,5,0.7,0.3)
    }
  /**
   * Test if ALS model matches explain
   *
   * @param users number of users
   * @param products number of products
   * @param features number of features (rank of problem)
   * @param iterations number of iterations to run
   * @param samplingRate what fraction of the user-product pairs are known
   * @param matchThreshold max difference allowed to consider a predicted rating correct
   * @param bulkPredict flag to test bulk prediction
   * @param negativeWeights whether the generated data can contain negative values
   * @param numUserBlocks number of user blocks to partition users into
   * @param numProductBlocks number of product blocks to partition products into
   * @param negativeFactors whether the generated user/product factors can have negative entries
   */
  def testALSExplain(
      users: Int,
      products: Int,
      features: Int,
      iterations: Int,
      samplingRate: Double,
      matchThreshold: Double,
      bulkPredict: Boolean = false,
      negativeWeights: Boolean = false,
      numUserBlocks: Int = -1,
      numProductBlocks: Int = -1,
      negativeFactors: Boolean = true){

    val implicitPrefs: Boolean = true
    val numUserBlocks: Int = -1
    val numProductBlocks: Int = -1

    val (sampledRatings, trueRatings, truePrefs) = ALSSuite.generateRatings(users, products,
      features, samplingRate, implicitPrefs, negativeWeights, negativeFactors)

    val model = new ALS()
      .setUserBlocks(numUserBlocks)
      .setProductBlocks(numProductBlocks)
      .setRank(features)
      .setIterations(iterations)
      .setAlpha(1.0)
      .setImplicitPrefs(implicitPrefs)
      .setLambda(0.00)
      .setSeed(0L)
      // .setNonnegative(!negativeFactors)
      .run(sc.parallelize(sampledRatings))

    val usersProducts = for (u <- 0 until users; p <- 0 until products) yield (u, p)
    val userProductsRDD = sc.parallelize(usersProducts)
    val predictedRatings = model.predict(userProductsRDD)

    val explanation = new ALSExplain().explain(model.productFeatures,
                                         sc.parallelize(sampledRatings),
                                         topExplanation=3,
                                         regParam=0.00,
                                         alpha=1.0)
    val explanation_res = explanation.map(x => (x.productExplanations.map(y => (x.user, y.explainedPid, y.score))))
                                     .flatMap(x => x)
                                     .distinct()
                                     .map(x => ((x._1, x._2), x._3))
    val predictedRatings_res = predictedRatings.map(x => ((x.user, x.product), x.rating))
    
    val final_res = explanation_res.union(predictedRatings_res).groupByKey()
    final_res.take(5).foreach(println)
    // val explain_table = explanation
    //                     .select("user", F.explode("productExplanations"))
    //                     .select("user", "col.explainedPid", "col.score", "col.productInfluence")
    //                     .toDF("user", "explainedPid", "score", "productInfluence")
    //                     .select("user", "explainedPid", "score", F.explode("productInfluence"))
    //                     .select("user", "explainedPid", "score", "col.productId", "col.influenceScore", "col.percentage")
    //                     .toDF("user", "explainedPid", "score", "productId", "influenceScore", "percentage")
    //
    // val compare_explain_table = (
    //     explain_table.
    //     select("user", "explainedPid", "score").
    //     toDF("userId", "movieId", "newPrediction").
    //     distinct()
    // )
    // val compare_table = predictions.join(compare_explain_table, Seq("userId", "movieId"))
    // println(compare_table.stat.corr("prediction", "newPrediction"))
    
  }
}
// scalastyle:on

