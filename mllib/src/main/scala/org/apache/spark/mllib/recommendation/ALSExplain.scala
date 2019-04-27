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

import breeze.linalg.{*, diag, inv, DenseMatrix, DenseVector, Matrix, Vector}
import scala.collection.mutable.ListBuffer

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD



// Nested data structures to hold product influence scores for each user
/**
 * ProductInfluence object holds absolute influrenceScore and relative percentage that productId
 * contributes to a particular product (expainedId in the parent object)
 * @param productId the influencing productId
 * @param influenceScore the absolute influence score (Su_i_j * Cu_j from paper)
 * @param percentage the relative percent this influence score makes up for this item
 */
case class ProductInfluence(productId: Int,
                            influenceScore: Double,
                            percentage: Double)


/**
 * ProductExplanation object holds the explained productId, the overall preference score the
 * original ALSModel reported for this user and item, and an array of ProductInfluence objects
 * containing the absolute and relative influrnece scores that contributed to overall preference
 * @param explainedPid the productId that is being explained
 * @param score the overall preference score of this user to explainedPid (p_ui from the paper)
 * @param productInfluence array of ProductExplaination objects with the explained components
 */
case class ProductExplanation(explainedPid: Int,
                              score: Double,
                              productInfluence: Array[ProductInfluence])


/**
 * UserExplaination object contains all the productExplainations for recommendations of a
   of a particular user.
 * @param user specific userId
 * @param productExplanations array of nested ProductExplanation for all products recommended to
                              this particular user
 */
case class UserExplanation(user: Int,
                           productExplanations: Array[ProductExplanation])


@SerialVersionUID(100L)
class ALSExplain extends Serializable {
  /**
   * Generates explainability scores for recommendations for the  implicit Alternating Least
   * Squares (ALS) matrix factorization model.
   * The algorithm used is based on "Collaborative Filtering for Implicit Feedback Datasets",
   * available at https://doi.org/10.1109/ICDM.2008.22.
   * @param prodFactors itemFactors RDD returned from ALSModel object with implicitPrefs=True
   * @param ratings RDD of Ratings used to fit the implicit ALSModel
   * @param regParam Regularization constant used to fit the implicit ALSModel
   * @param alpha Alpha constant used to fit the implicit ALSModel
   * @param topExplanation Integer limit to the number of explaining items to return (default 10)
   * @return a DataFrame UserExplanation(user, productExplanations) objects with
             influenceScore for each recommendation
   */
  def explain(prodFactors: RDD[(Int, Array[Double])],
              ratings: RDD[Rating],
              regParam: Double,
              alpha: Double,
              topExplanation: Int = 10):
  RDD[UserExplanation] = {
    val sc = prodFactors.context
    // Create row number for each product factor
    // ((prodId, prodFactor), prodRowNum)
    val indexedProdFactors = prodFactors.zipWithIndex()
    // (prodId, (prodRowNum, prodFactor))
    val productToIndex = indexedProdFactors.map( v => (v._1._1, (v._2, v._1._2)))
    // (prodRowNum, prodFactor)
    val indexedFactors = indexedProdFactors.map( v => (v._2, v._1._2))
    // Convert indexedFactors to Spark matricies and compute Y, YT, YTY
    val YIndexMatrix = toIndexedMatrix(indexedFactors)
    val Y = YIndexMatrix.toBlockMatrix()
    val YT = Y.transpose
    val YTY = YT.multiply(Y)

    // Create broadcast variables to send to the executors
    val local_YT = sc.broadcast(YT.toLocalMatrix().asBreeze).value
    val local_Y = sc.broadcast(Y.toLocalMatrix().asBreeze).value
    val local_YTY = sc.broadcast(YTY.toLocalMatrix().asBreeze).value
    // Create maps for lookup
    val indexProdCollect = indexedProdFactors.map(v => (v._1._1, v._2)).collect()
    // (prodId to prodRow)
    val indexMap = sc.broadcast(indexProdCollect
      .map(keyValue => (keyValue._1, keyValue._2)).toMap).value
    // (prodRow to prodId)
    val productMap = sc.broadcast(indexProdCollect
      .map( keyValue => (keyValue._2, keyValue._1)).toMap).value

    // Group ratings by user and send to UserExplaination()
    val userExplanationRDD = ratings
      .map( rating => (rating.product, rating))
      .join(productToIndex) // Join to productRowNumber and factor
      .map{
        case (prodId, (rating, prodRowToFactor)) =>
          (rating.user, (prodRowToFactor, (rating.rating * alpha)))
      }
      .groupByKey()
      .map {
          case (user, cuArray) => UserExplanation(user,
                                   generateUserExplanation(cuArray.toArray,
                                                           local_YT,
                                                           local_Y,
                                                           local_YTY,
                                                           regParam,
                                                           productMap,
                                                           topExplanation)
                                  )
      }
    // Return
    userExplanationRDD
  }

  /**
   * Generates all ProductExplanation based on an individual's product confidence score
   * @param CuArray Array of productRow, productFactor and the adjusted
                    user's confidence value (rating * alpha) for the assiated item
                    ((prodRow, prodFactor), cuValue)
   * @param local_YT Broadcast value of the transposed prodFactor matrix Y
   * @param local_Y Broadcast value of the prodFactor matrix Y
   * @param local_YTY Broadcast value of the inner product of productFactor matrix YTY
   * @param regParam Alpha constant used to fit the implicit ALSModel
   * @param productLookup Map connecting prodRow of Y to original prodId
   * @param topExplaination: Maxiumum number of influence scores to report for each item
   */
  def generateUserExplanation(CuArray: Array[((Long, Array[Double]), Double)],
                              local_YT: Matrix[Double],
                              local_Y: Matrix[Double],
                              local_YTY: Matrix[Double],
                              regParam: Double,
                              productLookup: Map[Long, Int],
                              topExplanation: Int) : Array[ProductExplanation] = synchronized{
    // Construct YTCuMinusI matrix
    val YTCUMinusI = CuArray.map{
        case ((prodRow, prodFactor), cuValue) =>
         (prodRow, prodFactor.map(_ * cuValue)) // Multiply prodFactor by cuValue
        }.
        sortBy(_._1). // sort by prodRow
        map(_._2) // take cuValue
    val YTCUMinusIMatrix = new DenseMatrix(local_YTY.rows,
                                           CuArray.length,
                                           YTCUMinusI.flatten.toArray)

    // Construct YTCu matrix
    val YTCU = CuArray.map{
        case ((prodRow, prodFactor), cuValue) =>
        (prodRow, prodFactor.map( _ * (cuValue + 1))) // Note: original form of Cu
    }.
    sortBy(_._1). // sort by prodRow
    map(_._2) // take cuValue
    val YTCUMatrix = new DenseMatrix(local_YTY.rows, CuArray.length, YTCU.flatten.toArray)

    // Create Map of Cu_index to prodRow
    val cuIndex = CuArray.
      sortBy(_._1._1).
      map(_._1._1).
      zipWithIndex.
      map(row => (row._2, row._1)).toMap

    // Create Y matrix of prodFactors
    val Y = CuArray.map(row => (row._1._1, row._1._2)).sortBy(_._1).map(_._2)
    val YMatrix = new DenseMatrix(local_YTY.rows, CuArray.length, Y.flatten.toArray).t

    // Create diagonal lambda matrix
    val lambdaI_cu = diag(DenseVector(List.fill(local_YTY.rows)
    (regParam *  CuArray.length).toArray))

    // Perform calculation of user-specific influence matrix S
    val A = YTCUMinusIMatrix * YMatrix
    val W = inv(local_YTY.toDenseMatrix + A.toDenseMatrix + lambdaI_cu.toDenseMatrix).toDenseMatrix
    val S = (local_Y * W * YTCUMatrix).toDenseMatrix

    // Generate all product explainations for this user for each row of the influence matrix S
    S(*, ::).map( x => generateProductExplanation(x, cuIndex, productLookup, topExplanation))
      .toArray.zipWithIndex.map(
      row => ProductExplanation(productLookup(row._2), row._1._1, row._1._2))
      .sortBy(_.explainedPid)
  }

  /**
   * Generate Product Explanation score for a given product (productId, influenceScore, percentage)
   * @param suCuInfluenceValues Vector of influrence score for a particular item
   * @param cuIndex: Map of Cu_Index to prodRow
   * @param productLookup: Map from prodRow to original prodId
   * @param topExplaination: Maxiumum number of influence scores to report
   */
  def generateProductExplanation(suCuInfluenceValues: Vector[Double],
                                 cuIndex: Map[Int, Long],
                                 productLookup: Map[Long, Int],
                                 topExplanation: Int): (Double, Array[ProductInfluence]) =
  synchronized{
    val result : ListBuffer[ProductInfluence] = new ListBuffer[ProductInfluence]

    // Denominator for precent
    var sum: Double = 0.0
    for ( key <- cuIndex ) {
      val score = suCuInfluenceValues(key._1)
      sum = sum + score
    }
    // Calculate influence scores of items that the user interacted with
    for ( key <- cuIndex ) {
      val score = suCuInfluenceValues(key._1)
      val item = new ProductInfluence(
        productId = productLookup(key._2),
        influenceScore = score,
        percentage = score/sum
      )
      result += item
    }
    // Sort and return by influence
    (sum, result.sortBy(-_.percentage).take(topExplanation).toArray)
  }

  /**
   * Helper function to confern a features Rdd to IndexedRowMatrix
   */
  def toIndexedMatrix(features: RDD[(Long, Array[Double])]): IndexedRowMatrix = {
    val rows = features.map { case (i, xs) => IndexedRow(i, Vectors.dense(xs)) }
    new IndexedRowMatrix(rows)
  }

}
